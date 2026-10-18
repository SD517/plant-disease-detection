"""Two-phase transfer-learning training with early stopping."""
import json
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from src import data
from src.models import build_model, compile_model, unfreeze_top


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def train_model(kind: str, head_epochs=config.HEAD_EPOCHS, ft_epochs=config.FINE_TUNE_EPOCHS,
                batch_size=config.BATCH_SIZE, use_class_weights=config.USE_CLASS_WEIGHTS):
    """Train one model. Returns (model, history, class_names, test_ds, seconds)."""
    set_seed()
    train_ds, val_ds, test_ds, class_names = data.make_datasets(kind, batch_size)
    cw = data.class_weights(class_names) if use_class_weights else None
    model = build_model(kind, num_classes=len(class_names))

    t0 = time.time()
    # Phase 1: train the new head only
    print(f"\n--- {kind}: phase 1 (frozen backbone, {head_epochs} epochs max) ---")
    compile_model(model, config.HEAD_LR)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=cw, callbacks=_callbacks(), verbose=2)

    # Phase 2: fine-tune the top of the backbone with a much smaller learning rate
    n_unfreeze = config.UNFREEZE_LAYERS[kind]
    print(f"\n--- {kind}: phase 2 (fine-tune last {n_unfreeze} layers, {ft_epochs} epochs max) ---")
    unfreeze_top(model, n_unfreeze)
    compile_model(model, config.FINE_TUNE_LR)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                   class_weight=cw, callbacks=_callbacks(), verbose=2)
    seconds = time.time() - t0

    history = {}
    for key in h1.history:
        history[key] = [float(v) for v in h1.history[key] + h2.history.get(key, [])]
    history["phase_boundary"] = len(h1.history["loss"])
    return model, history, class_names, test_ds, seconds


def save_class_names(class_names) -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    (config.MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))
