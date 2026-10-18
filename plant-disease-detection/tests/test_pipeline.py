"""Tests that run without TensorFlow or the real dataset."""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import class_info, splitting  # noqa: E402


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 120, 30)).save(path)


@pytest.fixture()
def raw(tmp_path):
    root = tmp_path / "raw" / "plantvillage dataset" / "color"
    for cls, n in [("Tomato___Late_blight", 60), ("Tomato___healthy", 40)]:
        for i in range(n):
            _img(root / cls / f"{i:04d}-uuid___RS_X {i}.JPG")
            if i % 3 == 0:                                   # augmented sibling of the same photo
                _img(root / cls / f"{i:04d}-uuid___RS_X {i}_flipTB.JPG")
    return tmp_path / "raw"


def test_find_class_root_descends_wrappers(raw):
    assert splitting.find_class_root(raw).name == "color"


def test_split_is_disjoint_stratified_and_group_aware(raw, tmp_path):
    root = splitting.find_class_root(raw)
    out = tmp_path / "split"
    summary = splitting.split_dataset(root, out, seed=1)
    total = sum(sum(v.values()) for v in summary.values())
    assert total == sum(1 for _ in root.rglob("*.JPG"))
    for split in ("train", "val", "test"):
        assert set(summary[split]) == {"Tomato___Late_blight", "Tomato___healthy"}
        assert all(n > 0 for n in summary[split].values())
    # No original photo (or its augmented copy) appears in two splits.
    # Group keys are only unique within a class, so include the class name.
    seen = {}
    for split in ("train", "val", "test"):
        for f in (out / split).rglob("*.JPG"):
            key = (f.parent.name, splitting.group_key(f.name))
            assert seen.setdefault(key, split) == split, f"leak: {key}"
    train_share = sum(summary["train"].values()) / total
    assert 0.6 < train_share < 0.8


def test_split_is_reproducible(raw, tmp_path):
    root = splitting.find_class_root(raw)
    a = splitting.split_dataset(root, tmp_path / "a", seed=7)
    b = splitting.split_dataset(root, tmp_path / "b", seed=7)
    assert a == b


def test_split_refuses_non_empty_output(raw, tmp_path):
    root = splitting.find_class_root(raw)
    splitting.split_dataset(root, tmp_path / "s")
    with pytest.raises(FileExistsError):
        splitting.split_dataset(root, tmp_path / "s")


def test_group_key_strips_augmentation_suffixes():
    base = "abc-123___RS_HL 5856"
    assert splitting.group_key(base + ".JPG") == "abc-123"
    assert splitting.group_key(base + "_flipTB.JPG") == "abc-123"
    assert splitting.group_key(base + "_90deg.JPG") == "abc-123"


@pytest.mark.parametrize("raw_name,crop,cond,healthy", [
    ("Tomato___Late_blight", "Tomato", "Late blight", False),
    ("Apple___healthy", "Apple", "Healthy", True),
    ("Pepper,_bell___Bacterial_spot", "Pepper, bell", "Bacterial spot", False),
    ("Corn_(maize)___Common_rust_", "Corn (maize)", "Common rust", False),
])
def test_parse_class_name(raw_name, crop, cond, healthy):
    assert class_info.parse_class_name(raw_name) == (crop, cond, healthy)


ALL_38 = """Apple___Apple_scab Apple___Black_rot Apple___Cedar_apple_rust Apple___healthy Blueberry___healthy
Cherry_(including_sour)___Powdery_mildew Cherry_(including_sour)___healthy
Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot Corn_(maize)___Common_rust_ Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy Grape___Black_rot Grape___Esca_(Black_Measles) Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy Orange___Haunglongbing_(Citrus_greening) Peach___Bacterial_spot Peach___healthy
Pepper,_bell___Bacterial_spot Pepper,_bell___healthy Potato___Early_blight Potato___Late_blight Potato___healthy
Raspberry___healthy Soybean___healthy Squash___Powdery_mildew Strawberry___Leaf_scorch Strawberry___healthy
Tomato___Bacterial_spot Tomato___Early_blight Tomato___Late_blight Tomato___Leaf_Mold Tomato___Septoria_leaf_spot
Tomato___Spider_mites_Two-spotted_spider_mite Tomato___Target_Spot Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus Tomato___healthy""".split()


def test_every_plantvillage_class_has_specific_info():
    assert len(ALL_38) == 38
    for name in ALL_38:
        info = class_info.get_info(name)
        if not info["healthy"]:
            assert info["cause"] != "Unknown", name
        assert info["advice"] and info["description"]


# ---------------------------------------------------------------- web app
@pytest.fixture()
def client():
    import app as webapp

    class Stub:
        def __init__(self, models): self.models = models
        def available_models(self): return self.models
        def predict(self, img, kind, **kw):
            info = class_info.get_info("Tomato___Late_blight")
            info["confidence"] = 0.93
            return {"model": kind, "prediction": info, "top": [], "low_confidence": False,
                    "inference_ms": 1.0, "heatmap": None}

    webapp.app.config["TESTING"] = True
    return webapp, webapp.app.test_client(), Stub


def _png():
    buf = io.BytesIO(); Image.new("RGB", (30, 30), (0, 150, 0)).save(buf, "PNG"); buf.seek(0); return buf


def test_pages_render(client):
    _, c, _ = client
    for url in ("/", "/models", "/about", "/api/status"):
        assert c.get(url).status_code == 200


def test_predict_without_model_returns_503(client, monkeypatch):
    webapp, c, Stub = client
    monkeypatch.setattr(webapp, "predictor", Stub([]))
    r = c.post("/api/predict", data={"image": (_png(), "leaf.png")})
    assert r.status_code == 503 and "error" in r.get_json()


def test_predict_validates_input(client, monkeypatch):
    webapp, c, Stub = client
    monkeypatch.setattr(webapp, "predictor", Stub(["mobilenetv2"]))
    assert c.post("/api/predict", data={}).status_code == 400
    assert c.post("/api/predict", data={"image": (io.BytesIO(b"x"), "a.txt")}).status_code == 400
    assert c.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "a.png")}).status_code == 400
    assert c.post("/api/predict", data={"image": (_png(), "a.png"), "model": "nope"}).status_code == 400


def test_predict_success(client, monkeypatch):
    webapp, c, Stub = client
    monkeypatch.setattr(webapp, "predictor", Stub(["mobilenetv2"]))
    r = c.post("/api/predict", data={"image": (_png(), "leaf.png"), "model": "mobilenetv2"})
    assert r.status_code == 200
    assert r.get_json()["prediction"]["condition"] == "Late blight"
