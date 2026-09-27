from fastapi import APIRouter, UploadFile, File, Depends, Form
from app.classification_models.model_loader import require_model
from fastapi.responses import JSONResponse
from gridfs import GridFS
from app.auth.dependencies import get_current_user_optional
from app.db.mongo import get_mongo_db
from app.utils.getters_services import get_image_from_gridfs
from app.utils.history_cleanup import delete_all_images
from app.utils.exceptions import (
    invalid_image_id_exception, 
    image_not_found_exception,
    invalid_lime_image_exception,
    invalid_image_exception
)
from app.xai.models import Explanation, ExplanationItem
from app.xai.schemas import XAIResponse
from app.xai.service import ( 
    explain_image_with_gradcam, explain_image_with_integrated_gradients, explain_image_with_lime, 
    explain_image_with_anchor, explain_image_with_shap,
    handle_authenticated_user, handle_unknown_user
)
import cv2
from typing import Optional
from pymongo.database import Database
from bson import ObjectId
from dataclasses import asdict

xai_router = APIRouter()


@xai_router.post("/gradcam", response_model=XAIResponse, dependencies=[Depends(require_model)])
async def gradcam_explanation(
    file: UploadFile = File(...),
    history_id: Optional[str] = Form(None),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    result = await explain_image_with_gradcam(file)

    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)
    if user:
        explanation_item = handle_authenticated_user(db, file, "gradcam", overlay_bgr, heatmap_bgr, history_id)
    else:
        explanation_item = handle_unknown_user("gradcam", overlay_bgr, heatmap_bgr)

    explanation_response = Explanation(
        history_id=history_id,
        explanations=[explanation_item]
    )
    return XAIResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=asdict(explanation_response)
    )


@xai_router.post("/lime", response_model=XAIResponse, dependencies=[Depends(require_model)])
async def lime_explanation(
    file: UploadFile = File(...),
    history_id: Optional[str] = Form(None),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    result = await explain_image_with_lime(file)

    if result is None:
        raise invalid_lime_image_exception
    
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)

    if user:
        explanation_item = handle_authenticated_user(db, file, "lime", overlay_bgr, heatmap_bgr, history_id)
    else:
        explanation_item = handle_unknown_user("lime", overlay_bgr, heatmap_bgr)

    explanation_response = Explanation(
        history_id=history_id,
        explanations=[explanation_item]
    )
    return XAIResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=asdict(explanation_response)
    )
    

@xai_router.post("/anchor", response_model=XAIResponse, dependencies=[Depends(require_model)])
async def anchor_explanation(
    file: UploadFile = File(...),
    history_id: Optional[str] = Form(None),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    result = await explain_image_with_anchor(file)

    if result is None:
        raise invalid_image_exception
    
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)

    if user:
        explanation_item = handle_authenticated_user(db, file, "anchor", overlay_bgr, heatmap_bgr, history_id)
    else:
        explanation_item = handle_unknown_user("anchor", overlay_bgr, heatmap_bgr)

    explanation_response = Explanation(
        history_id=history_id,
        explanations=[explanation_item]
    )
    return XAIResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=asdict(explanation_response)
    )

     
@xai_router.post("/shap", response_model=XAIResponse, dependencies=[Depends(require_model)])
async def shap_explanation(
    file: UploadFile = File(...),
    history_id: Optional[str] = Form(None),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    result = await explain_image_with_shap(file)

    if result is None:
        raise invalid_image_exception
    
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)

    if user:
        explanation_item = handle_authenticated_user(db, file, "shap", overlay_bgr, heatmap_bgr, history_id)
    else:
        explanation_item = handle_unknown_user("shap", overlay_bgr, heatmap_bgr)

    explanation_response = Explanation(
        history_id=history_id,
        explanations=[explanation_item]
    )
    return XAIResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=asdict(explanation_response)
    )
    

@xai_router.post("/ig", response_model=XAIResponse, dependencies=[Depends(require_model)])
async def integrated_gradients_explanation(
    file: UploadFile = File(...),
    history_id: Optional[str] = Form(None),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    result = await explain_image_with_integrated_gradients(file)

    if result is None:
        raise invalid_image_exception
    
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)

    if user:
        explanation_item = handle_authenticated_user(db, file, "integrated gradients", overlay_bgr, heatmap_bgr, history_id)
    else:
        explanation_item = handle_unknown_user("integrated gradients", overlay_bgr, heatmap_bgr)

    explanation_response = Explanation(
        history_id=history_id,
        explanations=[explanation_item]
    )
    return XAIResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=asdict(explanation_response)
    )


@xai_router.get("/images/{image_id}")
async def get_image(image_id: str, db: Database = Depends(get_mongo_db)):
    try:
        object_id = ObjectId(image_id)
    except Exception:
        raise invalid_image_id_exception
    
    try:
        return get_image_from_gridfs(db, image_id)
    except Exception:
        raise image_not_found_exception
    

@xai_router.get("/images/all_images")
async def list_all_images(db: Database = Depends(get_mongo_db)):
    fs = GridFS(db)
    files = fs.find()
    result = []

    for file in files:
        result.append({
            "filename": file.filename,
            "file_id": str(file._id),
            "upload_date": file.upload_date.isoformat() 
        })

    return JSONResponse(content=result)


@xai_router.delete("/images/delete_all_images")
async def delete_all_images_endpoint(db: Database = Depends(get_mongo_db)):
    try:
        delete_all_images(db)
        return JSONResponse(content={"message": "All images have been deleted successfully."})
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)
