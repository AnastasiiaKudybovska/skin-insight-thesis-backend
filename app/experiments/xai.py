"""Checkpoint-backed image explanations on the exact classifier input crop."""

import math
from contextlib import contextmanager

import cv2
import keras
import numpy as np
import tensorflow as tf

from app.experiments.inference import (
    CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, build_classifier, model_logits, release_model,
)


COMMON_METHODS = ("integrated_gradients", "occlusion_sensitivity")
ATTENTION_METHODS = ("attention_rollout", "transformer_attribution", "transition_attention_maps")
XAI_BY_MODEL = {
    "efficientnetb0": (*COMMON_METHODS, "gradcam"),
    "vit": (*ATTENTION_METHODS, *COMMON_METHODS),
    "deit": (*ATTENTION_METHODS, *COMMON_METHODS),
    "swin": (*COMMON_METHODS, "swin_transformer_attribution"),
}


def _logits(model, images):
    return model_logits(model, images)


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _clip_outliers(values, quantile):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.minimum(values, float(np.quantile(values, quantile)))


def _resize_map(values, quantile=1.0):
    values = tf.convert_to_tensor(_clip_outliers(values, quantile)[..., None])
    return _normalize(tf.image.resize(values, (224, 224), method="bilinear")[..., 0].numpy())


def _pixel_map(values, sigma=4.0, quantile=0.99):
    # Raw input-gradient maps are dominated by a few pixels; smooth, then clip before scaling.
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize(_clip_outliers(cv2.GaussianBlur(values, (0, 0), sigma), quantile))


def _log_odds(logits, class_index):
    others = tf.concat([logits[:, :class_index], logits[:, class_index + 1:]], axis=-1)
    return logits[:, class_index] - tf.reduce_logsumexp(others, axis=-1)


def _iter_layers(root):
    seen, stack, found = set(), [root], []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        found.append(node)
        children = list(node.layers) if isinstance(node, keras.Model) else []
        for value in vars(node).values():
            if isinstance(value, keras.layers.Layer):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                children.extend(item for item in value if isinstance(item, keras.layers.Layer))
        stack.extend(children)
    return found


def _attention_layers(model):
    layers = [layer for layer in _iter_layers(model) if isinstance(layer, keras.layers.MultiHeadAttention)]
    if not layers:
        raise RuntimeError("Transformer attention layers not found")
    return layers


@contextmanager
def _capture_attention(attention_layers, scores, tape=None):
    originals = []
    try:
        for layer in attention_layers:
            original = layer.call
            originals.append((layer, original))

            def wrapped(*args, _original=original, **kwargs):
                kwargs.pop("return_attention_scores", None)
                output, attention = _original(*args, return_attention_scores=True, **kwargs)
                if tape is not None:
                    tape.watch(attention)
                scores.append(attention)
                return output

            layer.call = wrapped
        yield
    finally:
        for layer, original in originals:
            layer.call = original


def _attention_map(model, image, class_index, method):
    attention_layers = _attention_layers(model)
    scores = []
    weighted = method != "attention_rollout"
    if weighted:
        with tf.GradientTape() as tape:
            with _capture_attention(attention_layers, scores, tape):
                target = _logits(model, image[None, ...])[0, class_index]
        gradients = tape.gradient(target, scores)
    else:
        with _capture_attention(attention_layers, scores):
            _logits(model, image[None, ...])
        gradients = [None] * len(scores)
    if not scores:
        raise RuntimeError("No attention matrices captured")

    relevance = None
    for matrix, gradient in zip(scores, gradients):
        if method == "transformer_attribution":
            # Chefer et al. 2021 (generic attention explainability): R <- R + mean_h(relu(grad * A)) @ R.
            if gradient is None:
                continue
            attention = tf.nn.relu(matrix * gradient).numpy()[0].mean(axis=0)
            if relevance is None:
                relevance = np.eye(attention.shape[0], dtype=np.float32)
            relevance = relevance + attention @ relevance
            continue
        if weighted:
            attention = matrix[0] if gradient is None else matrix[0] * tf.nn.relu(gradient[0])
            attention = np.maximum(tf.reduce_mean(attention, axis=0).numpy(), 0)
        else:
            attention = matrix.numpy()[0].mean(axis=0)
        attention = attention + np.eye(attention.shape[0], dtype=np.float32)
        attention = attention / np.maximum(attention.sum(axis=-1, keepdims=True), 1e-12)
        relevance = attention if relevance is None else attention @ relevance

    if relevance is None:
        raise RuntimeError("Attention gradients are unavailable")

    tokens = relevance.shape[0]
    prefix, side = next(
        ((count, math.isqrt(tokens - count)) for count in range(1, 4)
         if math.isqrt(tokens - count) ** 2 == tokens - count),
        (None, None),
    )
    if prefix is None:
        raise RuntimeError(f"Cannot map {tokens} tokens to image patches")
    # DeiT-distilled predicts from both CLS and distillation tokens, so average their rows.
    patch_relevance = relevance[:prefix, prefix:].mean(axis=0)
    return _resize_map(patch_relevance.reshape(side, side), quantile=0.99)


def _integrated_gradients(model, image, class_index, baselines, steps=32):
    # Averaging black and white baselines avoids zero attribution for pixels close to a
    # single baseline colour (dark lesions vs. a black baseline), per Sturmfels et al. 2020.
    attribution = tf.zeros(image.shape[:2])
    alphas = tf.linspace(0.0, 1.0, steps)
    for baseline in baselines:
        baseline = tf.broadcast_to(tf.cast(baseline, tf.float32), tf.shape(image))
        delta = image - baseline
        gradients = []
        for start in range(0, steps, 4):
            interpolated = baseline[None, ...] + alphas[start:start + 4, None, None, None] * delta[None, ...]
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                target = tf.reduce_sum(_logits(model, interpolated)[:, class_index])
            gradient = tape.gradient(target, interpolated)
            if gradient is None:
                raise RuntimeError("Integrated Gradients input gradients are unavailable")
            gradients.append(gradient)
        integrated = delta * tf.reduce_mean(tf.concat(gradients, axis=0), axis=0)
        attribution += tf.reduce_sum(tf.abs(integrated), axis=-1)
    return _pixel_map(attribution.numpy() / len(baselines), sigma=6.0)


def _occlusion_sensitivity(model, image, class_index, window=32, stride=16):
    # Log-odds stays informative when the softmax is saturated near 1.0.
    baseline = float(_log_odds(_logits(model, image[None, ...]), class_index)[0])
    source = image.numpy()
    fill = source.mean(axis=(0, 1))
    offsets = list(range(0, 224 - window + 1, stride))
    variants, windows = [], []
    for top in offsets:
        for left in offsets:
            occluded = source.copy()
            occluded[top:top + window, left:left + window] = fill
            variants.append(occluded)
            windows.append((top, left))
    drops = np.zeros((224, 224), dtype=np.float32)
    counts = np.zeros((224, 224), dtype=np.float32)
    for start in range(0, len(variants), 8):
        scores = _log_odds(_logits(model, tf.convert_to_tensor(np.stack(variants[start:start + 8]))), class_index)
        for (top, left), score in zip(windows[start:start + 8], scores.numpy()):
            drops[top:top + window, left:left + window] += baseline - float(score)
            counts[top:top + window, left:left + window] += 1
    drops = np.maximum(drops / np.maximum(counts, 1), 0)
    return _normalize(cv2.GaussianBlur(drops, (0, 0), stride / 2))


def _gradcam(model, image, class_index):
    grad_model = keras.Model(model.inputs, [model.get_layer("top_activation").output, model.output])
    with tf.GradientTape() as tape:
        features, output = grad_model(image[None, ...], training=False)
        if isinstance(output, dict):
            output = output.get("logits", next(iter(output.values())))
        target = output[0, class_index]
    gradients = tape.gradient(target, features)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heat = tf.reduce_sum(features[0] * weights[0][None, None, :], axis=-1)
    return _resize_map(tf.nn.relu(heat).numpy())


def _swin_class_activation(model, image, class_index):
    # Swin logits are Dense(mean over final-stage tokens), so W_c . token is each 7x7 cell's
    # exact share of the logit (CAM; identical to Grad-CAM for this linear head).
    converter = getattr(getattr(model, "preprocessor", None), "image_converter", None)
    batch = image[None, ...]
    tokens = model.backbone(batch if converter is None else converter(batch), training=False)[0]
    contributions = tf.linalg.matvec(tokens, model.output_dense.kernel[:, class_index]).numpy()
    side = math.isqrt(contributions.shape[0])
    if side * side != contributions.shape[0]:
        raise RuntimeError(f"Cannot map {contributions.shape[0]} Swin tokens to a square grid")
    return _resize_map(np.maximum(contributions, 0).reshape(side, side))


def explain_classifier(image_rgb, model_id, segmentation_id, methods):
    """Return per-method heatmaps and the class each explanation targets."""
    unsupported = set(methods) - set(XAI_BY_MODEL[model_id])
    if unsupported:
        raise ValueError(f"XAI method unavailable for {model_id}: {', '.join(sorted(unsupported))}")
    model = build_classifier(model_id, segmentation_id)
    try:
        def to_model_input(pixels):
            return (pixels / 255.0 - IMAGENET_MEAN) / IMAGENET_STD if model_id == "deit" else pixels

        model_input = to_model_input(tf.convert_to_tensor(image_rgb, dtype=tf.float32))
        class_index = int(tf.argmax(_logits(model, model_input[None, ...])[0]))
        maps = {}
        for method in methods:
            if method == "attention_rollout":
                maps[method] = _attention_map(model, model_input, class_index, method)
            elif method == "transformer_attribution":
                maps[method] = _attention_map(model, model_input, class_index, method)
            elif method == "transition_attention_maps":
                maps[method] = _attention_map(model, model_input, class_index, method)
            elif method == "integrated_gradients":
                baselines = [to_model_input(np.full(3, value, dtype=np.float32)) for value in (0.0, 255.0)]
                maps[method] = _integrated_gradients(model, model_input, class_index, baselines)
            elif method == "occlusion_sensitivity":
                maps[method] = _occlusion_sensitivity(model, model_input, class_index)
            elif method == "gradcam":
                maps[method] = _gradcam(model, model_input, class_index)
            elif method == "swin_transformer_attribution":
                maps[method] = _swin_class_activation(model, model_input, class_index)
        return CLASS_NAMES[class_index], maps
    finally:
        release_model(model)
