# LeafCheck - Plant Leaf Disease Detection with Transfer Learning

Classifies leaf photos into **38 crop / disease classes** of the PlantVillage dataset using
**MobileNetV2** and **ResNet50** (ImageNet transfer learning), compares the two models, explains each
prediction with **Grad-CAM**, and serves everything through a **Flask website**.

```
plant-disease-detection/
|-- main.py              # CLI: prepare | train | predict
|-- app.py               # Flask website (python app.py)
|-- config.py            # all settings in one place
|-- requirements.txt
|-- src/
|   |-- splitting.py     # stratified, group-aware 70/15/15 split (no TensorFlow needed)
|   |-- data.py          # tf.data pipelines + augmentation + class weights
|   |-- models.py        # MobileNetV2 / ResNet50 + new softmax head, fine-tuning helper
|   |-- train.py         # two-phase training (frozen head -> fine-tune), early stopping
|   |-- evaluate.py      # accuracy, top-3, precision/recall/F1, confusion matrix, speed
|   |-- visualize.py     # training curves, confusion matrix, per-class F1, comparison
|   |-- gradcam.py       # Grad-CAM heat-maps
|   |-- predictor.py     # loads saved models and predicts one image
|   `-- class_info.py    # readable names + short, general advice for every class
|-- templates/ static/   # website (HTML, CSS, JS)
|-- tests/               # pytest (runs without TensorFlow or the dataset)
|-- data/  models/  results/
```

## 1. Setup

Python 3.9 - 3.12 is recommended.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The first training run downloads the ImageNet weights, so it needs an internet connection.

## 2. Get the dataset

Download PlantVillage (about 54,000 images, 38 classes), for example from Kaggle:
`abdallahalidev/plantvillage-dataset` (use the **color** folder).

```bash
pip install kaggle        # needs your Kaggle API token
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```

Or download it in the browser and unzip it into `data/raw/`. The script finds the class folders by itself
(it looks inside wrapper folders such as `plantvillage dataset/color`).

## 3. Split the data

```bash
python main.py prepare --raw data/raw          # add --link to save disk space
```

This creates `data/split/{train,val,test}` (70 / 15 / 15) and `split_manifest.csv`.
The split is **stratified** (every class in every set) and **group-aware**: an original photo and its
augmented copies always stay in the same set, so the test score is not inflated by near-duplicates.

## 4. Train and evaluate

```bash
python main.py train                           # both models
python main.py train --models mobilenetv2      # just one
python main.py train --head-epochs 3 --ft-epochs 3   # quick trial run
```

Training is two-phase: (1) the new classification head is trained with the backbone frozen,
(2) the top backbone layers are unfrozen and fine-tuned with a 100x smaller learning rate.
A GPU is strongly recommended (see Colab below); on CPU expect hours.

Outputs:

* `models/<model>.keras` and `models/class_names.json`
* `results/comparison.csv` and `results/model_comparison.png`
* `results/<model>/` with `metrics.json`, `classification_report.csv`, `confusion_matrix.csv/.png`,
  `training_curves.png`, `per_class_f1.png`, `history.json`

All numbers in your report must come from these files.

## 5. Run the website

```bash
python app.py
```

Open http://127.0.0.1:5000 - drop a leaf photo, choose the model, read the diagnosis, confidence,
advice and the Grad-CAM overlay (use the slider to blend it). The **Models** page shows the comparison
table and plots from `results/`. Without a trained model the page shows a clear "train first" message.

Command-line prediction: `python main.py predict --image leaf.jpg --model mobilenetv2`

## 6. Tests

```bash
python -m pytest -q
```

## Training on Google Colab (free GPU)

1. Runtime > Change runtime type > GPU. Upload the project folder (zip) and the dataset zip to Drive.
2. In a notebook:
```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/plant-disease-detection.zip -d /content
!unzip -q /content/drive/MyDrive/plantvillage-dataset.zip -d /content/plant-disease-detection/data/raw
%cd /content/plant-disease-detection
!pip install -q flask
!python main.py prepare --raw data/raw --link
!python main.py train
```
3. Download the `models/` and `results/` folders and copy them into your local project, then `python app.py`.

## Configuration

Everything (image size, batch size, epochs, learning rates, layers to unfreeze, split ratios, seed) is in
`config.py`.

## Known limitations (state these in your report)

* PlantVillage photos are single leaves on plain backgrounds; accuracy on real field photos is usually much lower.
* Several photos can show the same physical leaf from different angles. They have different file ids, so they
  cannot be grouped automatically, and the test accuracy may still be somewhat optimistic.
* The model only knows 38 classes and does not check whether the picture is a leaf.
* The advice texts are general guidance, not agronomic prescriptions.
