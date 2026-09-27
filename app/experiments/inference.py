"""Inference matching the checkpoints and preprocessing in prep-main-pipeline-X.ipynb."""

import gc
from pathlib import Path

import cv2
import keras_hub
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASSIFIER_DIR = Path(__file__).resolve().parents[1] / "classification_models"
SEGMENTER_DIR = Path(__file__).resolve().parents[1] / "segmentation_models"
SEGMENTATION_NAMES = {
    "otsu": "otsu", "gradcam": "gradcam", "unet": "unet",
    "segnet": "segnet", "deeplabv3plus": "deeplab",
}
MODEL_NAMES = {"efficientnetb0": "efficientnet", "deit": "deit", "vit": "vit", "swin": "swin"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def classifier_checkpoint(model_id: str, segmentation_id: str) -> Path:
    name = MODEL_NAMES[model_id]
    if segmentation_id == "none":
        suffix = "original"
    elif segmentation_id == "deeplabv3plus":
        suffix = "segmented_deeplab"
    else:
        suffix = f"ablation_{SEGMENTATION_NAMES[segmentation_id]}"
    return CLASSIFIER_DIR / f"{name}_{suffix}.weights.h5"


def available_configurations() -> dict[str, list[str]]:
    result = {}
    for model_id in MODEL_NAMES:
        result[model_id] = [
            segmentation_id for segmentation_id in ("none", *SEGMENTATION_NAMES)
            if classifier_checkpoint(model_id, segmentation_id).is_file()
        ]
    return result


def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return layers.Conv2D(filters, 3, padding="same", activation="relu")(x)


def build_unet():
    inputs = layers.Input((256, 256, 3))
    c1 = _conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = _conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = _conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    bridge = _conv_block(p3, 256)
    x = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(bridge)
    x = _conv_block(layers.Concatenate()([x, c3]), 128)
    x = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(x)
    x = _conv_block(layers.Concatenate()([x, c2]), 64)
    x = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(x)
    x = _conv_block(layers.Concatenate()([x, c1]), 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return models.Model(inputs, outputs, name="unet")


class MaxPoolWithArgmax(layers.Layer):
    def call(self, inputs):
        return tf.nn.max_pool_with_argmax(inputs, ksize=2, strides=2, padding="SAME", include_batch_in_index=True)

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        shape = (batch, None if height is None else (height + 1) // 2,
                 None if width is None else (width + 1) // 2, channels)
        return shape, shape


class MaxUnpooling2D(layers.Layer):
    def call(self, inputs):
        updates, indices = inputs
        pooled_shape = tf.shape(updates)
        output_shape = tf.stack([pooled_shape[0], pooled_shape[1] * 2, pooled_shape[2] * 2, pooled_shape[3]])
        flat_output = tf.scatter_nd(
            tf.reshape(tf.cast(indices, tf.int64), [-1, 1]), tf.reshape(updates, [-1]),
            tf.cast(tf.reduce_prod(output_shape)[None], tf.int64),
        )
        return tf.reshape(flat_output, output_shape)

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape[0]
        return (batch, None if height is None else height * 2,
                None if width is None else width * 2, channels)


def build_segnet():
    inputs = layers.Input((256, 256, 3), name="image")
    x = layers.Lambda(lambda images: vgg16_preprocess(tf.cast(images, tf.float32)),
                      dtype="float32", name="vgg16_preprocess")(inputs)
    encoder_blocks = (("block1", 64, 2), ("block2", 128, 2), ("block3", 256, 3),
                      ("block4", 512, 3), ("block5", 512, 3))
    decoder_filters = {"block5": (512, 512, 512), "block4": (512, 512, 256),
                       "block3": (256, 256, 128), "block2": (128, 64), "block1": (64,)}
    pooling_indices = {}
    for block_name, filters, count in encoder_blocks:
        for index in range(count):
            x = layers.Conv2D(filters, 3, padding="same", name=f"{block_name}_conv{index + 1}")(x)
            x = layers.BatchNormalization(name=f"{block_name}_bn{index + 1}")(x)
            x = layers.Activation("relu", name=f"{block_name}_relu{index + 1}")(x)
        x, pooling_indices[block_name] = MaxPoolWithArgmax(name=f"{block_name}_pool")(x)
    for block_name, _, _ in reversed(encoder_blocks):
        x = MaxUnpooling2D(name=f"{block_name}_unpool")([x, pooling_indices[block_name]])
        for index, filters in enumerate(decoder_filters[block_name]):
            x = layers.Conv2D(filters, 3, padding="same", name=f"decoder_{block_name}_conv{index + 1}")(x)
            x = layers.BatchNormalization(name=f"decoder_{block_name}_bn{index + 1}")(x)
            x = layers.Activation("relu", name=f"decoder_{block_name}_relu{index + 1}")(x)
    outputs = layers.Conv2D(2, 1, activation="softmax", dtype="float32", name="segmentation")(x)
    return models.Model(inputs, outputs, name="segnet_vgg16")


def build_segmenter(method: str):
    if method == "unet":
        model = build_unet()
    elif method == "segnet":
        model = build_segnet()
    elif method == "deeplabv3plus":
        model = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
            "deeplab_v3_plus_resnet50_pascalvoc", load_weights=False,
            num_classes=2, activation="softmax",
        )
        if model.preprocessor is not None:
            model.preprocessor.image_size = (256, 256)
    else:
        raise ValueError(f"Unknown segmenter: {method}")
    checkpoint = SEGMENTER_DIR / f"segmenter_{SEGMENTATION_NAMES[method]}.weights.h5"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_weights(checkpoint)
    return model


def build_classifier(model_id: str, segmentation_id: str):
    if model_id == "efficientnetb0":
        base = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(base.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(7, activation=None, dtype="float32", name="ham10000_classifier")(x)
        model = models.Model(base.input, outputs)
    elif model_id == "deit":
        model = keras_hub.models.DeiTImageClassifier.from_preset(
            "deit_small_distilled_patch16_224_imagenet", load_weights=False,
            num_classes=7, preprocessor=None, head_dtype="float32",
        )
    elif model_id == "vit":
        model = keras_hub.models.ViTImageClassifier.from_preset(
            "vit_base_patch16_224_imagenet21k", load_weights=False,
            num_classes=7, head_dtype="float32",
        )
    elif model_id == "swin":
        model = keras_hub.models.SwinTransformerImageClassifier.from_preset(
            "swin_tiny_patch4_window7_224", load_weights=False,
            num_classes=7, head_dtype="float32",
        )
    else:
        raise ValueError(f"Unknown classifier: {model_id}")
    checkpoint = classifier_checkpoint(model_id, segmentation_id)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_weights(checkpoint)
    return model


def release_model(model):
    del model
    gc.collect()
    tf.keras.backend.clear_session()


def model_logits(model, batch):
    """Call a KerasHub model with the same image conversion used by predict()."""
    converter = getattr(getattr(model, "preprocessor", None), "image_converter", None)
    output = model(batch if converter is None else converter(batch), training=False)
    if isinstance(output, dict):
        output = output.get("logits", next(iter(output.values())))
    return output


def _model_output(model, batch):
    return np.asarray(model_logits(model, batch))


def resize_for_classifier(image_rgb: np.ndarray) -> np.ndarray:
    return tf.image.resize(image_rgb, (224, 224), antialias=True).numpy().astype("float32")


def remove_hair(image_rgb: np.ndarray) -> np.ndarray:
    """Black Hat detection and Navier-Stokes inpainting from the training notebook."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    cleaned_bgr = cv2.inpaint(image_bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)


def segment(image_rgb: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.ones(image_rgb.shape[:2], dtype=bool)
    if method == "otsu":
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask > 0
    if method == "gradcam":
        model = build_classifier("efficientnetb0", "none")
        try:
            grad_model = models.Model(model.inputs, [model.get_layer("top_activation").output, model.output])
            resized = tf.image.resize(tf.cast(image_rgb, tf.float32), (224, 224))[None, ...]
            with tf.GradientTape() as tape:
                activations, logits = grad_model(resized, training=False)
                target = logits[:, tf.argmax(logits[0])]
            grads = tape.gradient(target, activations)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            heat = tf.nn.relu(tf.reduce_sum(activations[0] * pooled, axis=-1))
            heat = heat / (tf.reduce_max(heat) + 1e-8)
            heat = tf.image.resize(heat[..., None], image_rgb.shape[:2], method="bilinear")[..., 0].numpy()
            return heat >= np.quantile(heat, 0.65)
        finally:
            release_model(model)
    model = build_segmenter(method)
    try:
        resized = tf.image.resize(image_rgb, (256, 256)).numpy().astype("float32")
        prediction = _model_output(model, resized[None, ...])[0]
        channel = 0 if method == "unet" else 1
        mask = tf.image.resize(prediction[..., channel:channel + 1], image_rgb.shape[:2],
                               method="nearest")[..., 0].numpy()
        return mask > 0.5
    finally:
        release_model(model)


def crop_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (np.argmax(sizes) + 1)
    ys, xs = np.where(mask)
    if not len(xs):
        return image_rgb
    height, width = image_rgb.shape[:2]
    pad_y, pad_x = int(height * 0.08), int(width * 0.08)
    y0, y1 = max(0, ys.min() - pad_y), min(height, ys.max() + pad_y + 1)
    x0, x1 = max(0, xs.min() - pad_x), min(width, xs.max() + pad_x + 1)
    return image_rgb[y0:y1, x0:x1]


def classify(image_rgb: np.ndarray, model_id: str, segmentation_id: str) -> dict[str, float]:
    model = build_classifier(model_id, segmentation_id)
    try:
        resized = resize_for_classifier(image_rgb)
        if model_id == "deit":
            resized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        logits = _model_output(model, resized[None, ...])[0]
        probabilities = tf.nn.softmax(logits).numpy()
        return {name: float(value) for name, value in zip(CLASS_NAMES, probabilities)}
    finally:
        release_model(model)
