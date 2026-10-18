"""Grad-CAM: highlight the leaf regions that drove the prediction."""
import io

import numpy as np
from PIL import Image


def make_gradcam(model, x_pre: np.ndarray, class_idx=None):
    """x_pre: pre-processed batch of shape (1, H, W, 3).
    Returns (cam in [0,1] with shape (h, w), probabilities, class_idx)."""
    import tensorflow as tf
    from src.models import backbone_of

    backbone = backbone_of(model)
    head = model.layers[model.layers.index(backbone) + 1:]
    x = tf.convert_to_tensor(x_pre, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv = backbone(x, training=False)       # last conv feature map
        tape.watch(conv)
        h = conv
        for layer in head:
            h = layer(h, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(h[0]))
        score = h[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))                  # (1, C)
    cam = tf.einsum("bhwc,bc->bhw", conv, weights)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), h.numpy()[0], class_idx


def heatmap_rgba(cam: np.ndarray, size, max_alpha: float = 0.85) -> Image.Image:
    """Colour the CAM (jet) and use its intensity as the alpha channel."""
    from matplotlib import colormaps
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(size, Image.BICUBIC)
    c = np.asarray(cam_img, dtype="float32") / 255.0
    rgba = (colormaps["jet"](c) * 255).astype("uint8")
    rgba[..., 3] = (c * max_alpha * 255).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Static overlay for reports (heat-map blended on the leaf photo)."""
    heat = heatmap_rgba(cam, img.size, max_alpha=1.0)
    out = img.convert("RGBA")
    heat.putalpha(heat.getchannel("A").point(lambda a: int(a * alpha)))
    return Image.alpha_composite(out, heat).convert("RGB")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
