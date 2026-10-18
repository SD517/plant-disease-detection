"""tf.data input pipelines (augmentation only on the training set)."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from src.models import get_preprocess
from src.splitting import count_images

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation() -> keras.Sequential:
    """Light geometric / photometric augmentation that mimics real photos."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
    ], name="augmentation")


def _load(split: str, batch_size: int, shuffle: bool):
    return keras.utils.image_dataset_from_directory(
        config.SPLIT_DIR / split, labels="inferred", label_mode="int",
        image_size=config.IMG_SIZE, batch_size=batch_size,
        shuffle=shuffle, seed=config.SEED)


def make_datasets(kind: str, batch_size: int = config.BATCH_SIZE):
    """Return train_ds, val_ds, test_ds, class_names (all already pre-processed)."""
    pre = get_preprocess(kind)
    train_raw = _load("train", batch_size, True)
    val_raw = _load("val", batch_size, False)
    test_raw = _load("test", batch_size, False)
    class_names = train_raw.class_names
    assert class_names == val_raw.class_names == test_raw.class_names, \
        "Class folders differ between train/val/test"

    aug = build_augmentation()
    train = (train_raw.map(lambda x, y: (pre(aug(x, training=True)), y), num_parallel_calls=AUTOTUNE)
             .prefetch(AUTOTUNE))
    val = val_raw.map(lambda x, y: (pre(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test = test_raw.map(lambda x, y: (pre(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train, val, test, class_names


def class_weights(class_names) -> dict:
    """'Balanced' weights: n_samples / (n_classes * n_class_samples)."""
    counts = count_images(config.SPLIT_DIR / "train")
    n = np.array([counts[c] for c in class_names], dtype="float64")
    w = n.sum() / (len(n) * n)
    return {i: float(v) for i, v in enumerate(w)}
