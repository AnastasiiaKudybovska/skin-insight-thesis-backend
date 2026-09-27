"""Authenticated inference API for the HAM10000 checkpoints."""

import base64
import json
from collections import OrderedDict
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.experiments.inference import available_configurations, classify, crop_from_mask, remove_hair, resize_for_classifier, segment
from app.experiments.xai import XAI_BY_MODEL, explain_classifier

experiment_router = APIRouter()

MODELS = {"efficientnetb0", "vit", "deit", "swin"}
COMPARISON_SEGMENTATIONS = ("deeplabv3plus", "otsu", "gradcam", "unet", "segnet")
XAI_METHODS = set().union(*XAI_BY_MODEL.values())
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_RUNS = 20
_runs = OrderedDict()
_runs_lock = Lock()
_inference_lock = Lock()


class ExplainRequest(BaseModel):
    run_id: str
    models: list[str] | None = None
    configurations: list[str] | None = None
    methods: list[str] = Field(min_length=1)


def _png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _preview(image: np.ndarray, is_mask: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 512 / max(height, width))
    if scale < 1:
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    return image


def _stages(image_rgb: np.ndarray, working_rgb: np.ndarray, mask: np.ndarray, crop_rgb: np.ndarray) -> dict[str, str]:
    original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    working_bgr = cv2.cvtColor(working_rgb, cv2.COLOR_RGB2BGR)
    if mask.all():
        overlay = working_bgr
    else:
        color_mask = np.zeros_like(working_bgr)
        color_mask[:, :, 1] = mask_uint8
        overlay = cv2.addWeighted(working_bgr, 0.7, color_mask, 0.3, 0)
    stages = {
        "original": _png_base64(_preview(original_bgr)),
        "hair_removed": _png_base64(_preview(working_bgr)),
        "mask": _png_base64(_preview(mask_uint8, is_mask=True)),
        "overlay": _png_base64(_preview(overlay)),
        "roi": _png_base64(_preview(crop_bgr)),
    }
    return stages


def _analyze_image(image_rgb, model_ids, segmentation_ids, supported):
    with _inference_lock:
        working_rgb = remove_hair(image_rgb)
        crops = {}
        stages = {}
        for segmentation_id in segmentation_ids:
            mask = segment(working_rgb, segmentation_id)
            crop = working_rgb if segmentation_id == "none" else crop_from_mask(working_rgb, mask)
            crops[segmentation_id] = crop
            stages[segmentation_id] = _stages(image_rgb, working_rgb, mask, crop)

        results = []
        for model_id in model_ids:
            for segmentation_id in segmentation_ids:
                if segmentation_id not in supported[model_id]:
                    continue
                probabilities = classify(crops[segmentation_id], model_id, segmentation_id)
                predicted_class = max(probabilities, key=probabilities.get)
                results.append({
                    "configuration_id": f"{model_id}:{segmentation_id}",
                    "model_id": model_id,
                    "segmentation_id": segmentation_id,
                    "predicted_class": predicted_class,
                    "confidence": probabilities[predicted_class],
                    "probabilities": probabilities,
                })
        prepared_crops = {key: resize_for_classifier(value) for key, value in crops.items()}
        return results, stages, prepared_crops


def _overlay(source_bgr, heat):
    heat = np.clip(heat, 0, 1).astype(np.float32)
    colored = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
    # Opacity follows the heat value so unimportant regions keep the original skin visible.
    alpha = (0.15 + 0.5 * heat)[..., None]
    overlay = source_bgr.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return colored, np.clip(overlay, 0, 255).astype(np.uint8)


def _explain_images(crops, configurations, methods):
    with _inference_lock:
        explanations = []
        unavailable = []
        for configuration_id, (model_id, segmentation_id) in configurations.items():
            supported = [method for method in methods if method in XAI_BY_MODEL[model_id]]
            unavailable.extend(f"{configuration_id}:{method}" for method in methods if method not in supported)
            if not supported:
                continue
            source = np.clip(crops[segmentation_id], 0, 255).astype(np.uint8)
            source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            predicted_class, maps = explain_classifier(crops[segmentation_id], model_id, segmentation_id, supported)
            for method, heat in maps.items():
                colored, overlay = _overlay(source_bgr, heat)
                explanations.append({
                    "configuration_id": configuration_id,
                    "model_id": model_id,
                    "segmentation_id": segmentation_id,
                    "method": method,
                    "predicted_class": predicted_class,
                    "overlay_image": _png_base64(overlay),
                    "heatmap_image": _png_base64(colored),
                })
        return explanations, unavailable


@experiment_router.get("/capabilities")
async def capabilities(user: dict = Depends(get_current_user)):
    return {"configurations": available_configurations(), "xai_methods": XAI_BY_MODEL}


@experiment_router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    models: str = Form(...),
    segmentation_enabled: bool = Form(True),
    compare_segmentations: bool = Form(True),
    user: dict = Depends(get_current_user),
):
    try:
        model_ids = json.loads(models)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=422, detail="models must be a JSON array")
    if not isinstance(model_ids, list) or not model_ids or len(model_ids) > len(MODELS) or any(not isinstance(model, str) or model not in MODELS for model in model_ids) or len(set(model_ids)) != len(model_ids):
        raise HTTPException(status_code=422, detail="Invalid classification models")
    segmentation_ids = (list(COMPARISON_SEGMENTATIONS) if compare_segmentations
                        else ["deeplabv3plus"]) if segmentation_enabled else ["none"]

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 10 MB)")
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Invalid image")
    if image_bgr.shape[0] * image_bgr.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions are too large")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    supported = available_configurations()
    available_pairs = [(model_id, segmentation_id) for model_id in model_ids
                       for segmentation_id in segmentation_ids if segmentation_id in supported[model_id]]
    if not available_pairs:
        raise HTTPException(status_code=422, detail="No checkpoints for the selected model and segmentation combinations")
    unavailable = [f"{model_id}:{segmentation_id}" for model_id in model_ids
                   for segmentation_id in segmentation_ids if segmentation_id not in supported[model_id]]
    try:
        results, stages, crops = await run_in_threadpool(_analyze_image, image_rgb, model_ids, segmentation_ids, supported)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {error}") from error

    run_id = str(uuid4())
    with _runs_lock:
        _runs[run_id] = {
            "user_id": str(user["_id"]),
            "models": model_ids,
            "segmentations": segmentation_ids,
            "crops": crops,
        }
        while len(_runs) > MAX_RUNS:
            _runs.popitem(last=False)

    return {
        "run_id": run_id,
        "demo": False,
        "segmentations": segmentation_ids,
        "remove_hair_artifacts": True,
        "results": results,
        "unavailable_configurations": unavailable,
        "stages_by_segmentation": stages,
    }


@experiment_router.post("/explain")
async def explain(payload: ExplainRequest, user: dict = Depends(get_current_user)):
    with _runs_lock:
        run = _runs.get(payload.run_id)
    if run is None or run["user_id"] != str(user["_id"]):
        raise HTTPException(status_code=404, detail="Experiment run not found")
    available = available_configurations()
    valid_configurations = {
        f"{model_id}:{segmentation_id}": (model_id, segmentation_id)
        for model_id in run["models"] for segmentation_id in run["segmentations"]
        if segmentation_id in available[model_id]
    }
    if payload.configurations is not None:
        configuration_ids = payload.configurations
    elif payload.models is not None and len(run["segmentations"]) == 1:
        configuration_ids = [f"{model_id}:{run['segmentations'][0]}" for model_id in payload.models]
    else:
        configuration_ids = []
    if (not configuration_ids or len(configuration_ids) > len(valid_configurations)
            or len(set(configuration_ids)) != len(configuration_ids)
            or any(item not in valid_configurations for item in configuration_ids)):
        raise HTTPException(status_code=422, detail="Invalid configurations for this run")
    if len(payload.methods) > len(XAI_METHODS) or len(set(payload.methods)) != len(payload.methods) or any(method not in XAI_METHODS for method in payload.methods):
        raise HTTPException(status_code=422, detail="Invalid XAI methods")

    chosen = {key: valid_configurations[key] for key in configuration_ids}
    if not any(method in XAI_BY_MODEL[model_id] for model_id, _ in chosen.values() for method in payload.methods):
        raise HTTPException(status_code=422, detail="None of the selected XAI methods supports the selected models")
    try:
        explanations, unavailable = await run_in_threadpool(_explain_images, run["crops"], chosen, payload.methods)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=f"XAI inference failed: {error}") from error
    return {"demo": False, "explanations": explanations, "unavailable_explanations": unavailable}
