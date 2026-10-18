"""LeafCheck - Flask website for plant leaf disease detection.

Run:   python app.py     then open http://127.0.0.1:5000
"""
import csv
import io

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

import config
from src.predictor import Predictor

ALLOWED = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
predictor = Predictor()

MODEL_LABELS = {"mobilenetv2": "MobileNetV2", "resnet50": "ResNet50"}


def _comparison():
    path = config.RESULTS_DIR / "comparison.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@app.route("/")
def index():
    models = predictor.available_models()
    return render_template("index.html", models=models, labels=MODEL_LABELS,
                           max_mb=config.MAX_UPLOAD_MB, threshold=int(config.CONFIDENCE_THRESHOLD * 100))


@app.route("/models")
def models_page():
    rows = _comparison()
    figures = [(m, f) for m in config.MODEL_NAMES
               for f in ("training_curves.png", "confusion_matrix.png", "per_class_f1.png")
               if (config.RESULTS_DIR / m / f).exists()]
    has_cmp_fig = (config.RESULTS_DIR / "model_comparison.png").exists()
    return render_template("models.html", rows=rows, figures=figures, labels=MODEL_LABELS,
                           has_cmp_fig=has_cmp_fig)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(config.RESULTS_DIR, filename)


@app.route("/api/status")
def status():
    return jsonify({"models": predictor.available_models(), "ready": bool(predictor.available_models())})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    available = predictor.available_models()
    if not available:
        return jsonify(error="No trained model found. Run `python main.py train` first."), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Please choose a leaf photo."), 400
    if file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type. Use JPG, PNG, BMP or WEBP."), 400

    kind = request.form.get("model") or available[0]
    if kind not in available:
        return jsonify(error=f"Model '{kind}' is not available."), 400

    raw = file.read()
    try:
        Image.open(io.BytesIO(raw)).verify()                      # reject corrupt / fake images
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="That file could not be read as an image."), 400

    try:
        return jsonify(predictor.predict(img, kind))
    except Exception as exc:                                       # pragma: no cover
        app.logger.exception("prediction failed")
        return jsonify(error=f"Prediction failed: {exc}"), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image is too large (max {config.MAX_UPLOAD_MB} MB)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
