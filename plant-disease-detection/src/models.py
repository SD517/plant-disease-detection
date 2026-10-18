"""Transfer-learning models: MobileNetV2 and ResNet50 with a new softmax head.

The networks take *pre-processed* images (see ``get_preprocess``). Keeping the
model free of custom pre-processing layers makes it load anywhere and keeps
Grad-CAM simple.
"""
from tensorflow import keras
from tensorflow.keras import layers

import config


def get_preprocess(kind: str):
    """ImageNet pre-processing function that matches the backbone's weights."""
    kind = kind.lower()
    if kind == "mobilenetv2":
        return keras.applications.mobilenet_v2.preprocess_input   # scales to [-1, 1]
    if kind == "resnet50":
        return keras.applications.resnet50.preprocess_input       # BGR + mean subtraction
    raise ValueError(f"Unknown model '{kind}'. Choose from {config.MODEL_NAMES}")


def get_backbone(kind: str, img_size=config.IMG_SIZE, weights="imagenet"):
    shape = (*img_size, 3)
    kind = kind.lower()
    if kind == "mobilenetv2":
        return keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=shape)
    if kind == "resnet50":
        return keras.applications.ResNet50(include_top=False, weights=weights, input_shape=shape)
    raise ValueError(f"Unknown model '{kind}'. Choose from {config.MODEL_NAMES}")


def build_model(kind: str, num_classes: int, img_size=config.IMG_SIZE,
                dropout=config.DROPOUT, weights="imagenet") -> keras.Model:
    """Frozen ImageNet backbone + GlobalAveragePooling + Dropout + Dense(softmax)."""
    base = get_backbone(kind, img_size, weights)
    base.trainable = False
    inputs = keras.Input(shape=(*img_size, 3), name="image")
    x = base(inputs, training=False)          # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"{kind.lower()}_plant")


def backbone_of(model: keras.Model) -> keras.Model:
    """Return the nested pre-trained backbone."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No backbone found in model")


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")])


def unfreeze_top(model: keras.Model, n_layers: int) -> None:
    """Make the last ``n_layers`` of the backbone trainable (BatchNorm stays frozen).
    The model must be re-compiled afterwards."""
    base = backbone_of(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
