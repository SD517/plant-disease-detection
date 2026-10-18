"""Load trained models lazily and run a prediction on a single PIL image."""
import base64
import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src.class_info import get_info
from src.gradcam import heatmap_rgba, make_gradcam, png_bytes


class Predictor:
    def __init__(self, models_dir=config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._models = {}
        self._lock = threading.Lock()
        self.class_names = None

    # -- discovery ------------------------------------------------------
    def available_models(self) -> list[str]:
        if not (self.models_dir / "class_names.json").exists():
            return []
        return [m for m in config.MODEL_NAMES if (self.models_dir / f"{m}.keras").exists()]

    def _load_class_names(self):
        if self.class_names is None:
            self.class_names = json.loads((self.models_dir / "class_names.json").read_text())
        return self.class_names

    def _load(self, kind: str):
        if kind not in self._models:
            from tensorflow import keras
            from src.models import get_preprocess
            model = keras.models.load_model(self.models_dir / f"{kind}.keras", compile=False)
            self._models[kind] = (model, get_preprocess(kind))
        return self._models[kind]

    # -- inference ------------------------------------------------------
    def predict(self, img: Image.Image, kind: str, topk: int = 3, cam: bool = True) -> dict:
        if kind not in self.available_models():
            raise FileNotFoundError(f"Model '{kind}' has not been trained yet.")
        names = self._load_class_names()
        with self._lock:
            model, pre = self._load(kind)
            rgb = img.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR)
            x = pre(np.asarray(rgb, dtype="float32")[None].copy())
            t0 = time.perf_counter()
            if cam:
                heat, probs, _ = make_gradcam(model, x)
            else:
                probs, heat = model.predict(x, verbose=0)[0], None
            ms = 1000 * (time.perf_counter() - t0)

        order = np.argsort(probs)[::-1][:topk]
        top = []
        for i in order:
            info = get_info(names[i])
            top.append({"crop": info["crop"], "condition": info["condition"],
                        "healthy": info["healthy"], "confidence": float(probs[i])})
        best = get_info(names[order[0]])
        best["confidence"] = float(probs[order[0]])
        result = {
            "model": kind,
            "prediction": best,
            "top": top,
            "low_confidence": best["confidence"] < config.CONFIDENCE_THRESHOLD,
            "inference_ms": round(ms, 1),
            "heatmap": None,
        }
        if heat is not None:
            png = png_bytes(heatmap_rgba(heat, config.IMG_SIZE))
            result["heatmap"] = "data:image/png;base64," + base64.b64encode(png).decode()
        return result
