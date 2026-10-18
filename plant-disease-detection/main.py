"""Command-line interface.

  python main.py prepare --raw data/raw            # split dataset 70/15/15
  python main.py train --models mobilenetv2 resnet50
  python main.py predict --image leaf.jpg --model mobilenetv2
"""
import argparse
import json
import sys
from pathlib import Path

import config


def cmd_prepare(a):
    from src import splitting
    root = splitting.find_class_root(Path(a.raw))
    n_cls = len([p for p in root.iterdir() if p.is_dir()])
    print(f"[prepare] class folders found in {root}: {n_cls}")
    if n_cls != 38:
        print("[prepare] WARNING: PlantVillage normally has 38 classes. Check your download.")
    summary = splitting.split_dataset(
        root, config.SPLIT_DIR,
        ratios=(config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO),
        seed=config.SEED, mode="link" if a.link else "copy", clean=a.clean)
    for split, per_class in summary.items():
        print(f"[prepare] {split:5s}: {sum(per_class.values()):6d} images")
    print(f"[prepare] split written to {config.SPLIT_DIR}")


def cmd_train(a):
    import pandas as pd
    from src import evaluate, train, visualize
    rows, comparison = {}, []
    for kind in a.models:
        out = config.RESULTS_DIR / kind
        out.mkdir(parents=True, exist_ok=True)
        model, hist, class_names, test_ds, secs = train.train_model(
            kind, head_epochs=a.head_epochs, ft_epochs=a.ft_epochs,
            batch_size=a.batch_size, use_class_weights=not a.no_class_weights)

        config.MODELS_DIR.mkdir(exist_ok=True)
        model_path = config.MODELS_DIR / f"{kind}.keras"
        model.save(model_path)
        train.save_class_names(class_names)

        y_true, probs, ms = evaluate.predict_dataset(model, test_ds)
        m = evaluate.compute_metrics(y_true, probs, class_names)

        pd.DataFrame(m["report"]).T.to_csv(out / "classification_report.csv")
        pd.DataFrame(m["confusion_matrix"], index=class_names, columns=class_names).to_csv(out / "confusion_matrix.csv")
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        visualize.plot_history(hist, kind, out / "training_curves.png")
        visualize.plot_confusion(m["confusion_matrix"], class_names, kind, out / "confusion_matrix.png")
        visualize.plot_per_class_f1(m["report"], class_names, kind, out / "per_class_f1.png")

        row = {
            "model": kind,
            "accuracy": round(m["accuracy"], 4), "top3_accuracy": round(m["top3_accuracy"], 4),
            "precision_macro": round(m["precision_macro"], 4), "recall_macro": round(m["recall_macro"], 4),
            "f1_macro": round(m["f1_macro"], 4), "f1_weighted": round(m["f1_weighted"], 4),
            "params": int(model.count_params()),
            "params_millions": round(model.count_params() / 1e6, 3),
            "size_mb": round(evaluate.model_size_mb(model_path), 1),
            "train_time_s": round(secs, 1), "epochs_run": len(hist["loss"]),
            "inference_ms": round(ms, 2),
        }
        comparison.append(row)
        rows[kind] = row
        (out / "metrics.json").write_text(json.dumps(row, indent=2))
        print(f"\n[{kind}] test accuracy {row['accuracy']:.4f} | macro-F1 {row['f1_macro']:.4f}")
        print("Most frequent confusions:")
        for t, p, c in evaluate.top_confusions(m["confusion_matrix"], class_names):
            print(f"   {t}  ->  {p}: {c}")

    # merge with earlier runs so training one model at a time still gives a table
    cmp_path = config.RESULTS_DIR / "comparison.csv"
    new = pd.DataFrame(comparison)
    if cmp_path.exists():
        old = pd.read_csv(cmp_path)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(cmp_path, index=False)
    if len(new) > 1:
        visualize.plot_comparison({r["model"]: r for r in new.to_dict("records")},
                                  config.RESULTS_DIR / "model_comparison.png")
    print("\n", new.to_string(index=False))
    print(f"\nSaved: {cmp_path}")


def cmd_predict(a):
    from PIL import Image, ImageOps
    from src.predictor import Predictor
    p = Predictor()
    if a.model not in p.available_models():
        sys.exit(f"Model '{a.model}' not found in {config.MODELS_DIR}. Run `python main.py train` first.")
    img = ImageOps.exif_transpose(Image.open(a.image))
    r = p.predict(img, a.model, cam=False)
    best = r["prediction"]
    print(f"{best['crop']} - {best['condition']}  ({best['confidence']:.1%})")
    for t in r["top"]:
        print(f"   {t['crop']:<22s} {t['condition']:<40s} {t['confidence']:.1%}")


def main():
    ap = argparse.ArgumentParser(description="Plant leaf disease detection (transfer learning)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="split the raw dataset into train/val/test")
    pp.add_argument("--raw", default=str(config.RAW_DIR))
    pp.add_argument("--clean", action="store_true", help="delete an existing split first")
    pp.add_argument("--link", action="store_true", help="hard-link files instead of copying")
    pp.set_defaults(fn=cmd_prepare)

    pt = sub.add_parser("train", help="train, evaluate and save the models")
    pt.add_argument("--models", nargs="+", default=config.MODEL_NAMES, choices=config.MODEL_NAMES)
    pt.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    pt.add_argument("--ft-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    pt.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    pt.add_argument("--no-class-weights", action="store_true")
    pt.set_defaults(fn=cmd_train)

    pr = sub.add_parser("predict", help="classify one image from the command line")
    pr.add_argument("--image", required=True)
    pr.add_argument("--model", default="mobilenetv2", choices=config.MODEL_NAMES)
    pr.set_defaults(fn=cmd_predict)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
