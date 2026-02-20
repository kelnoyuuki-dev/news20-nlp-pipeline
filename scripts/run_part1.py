# scripts/run_part1.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.data_loader import load_20newsgroups_train_test
from src.part1_classic import build_classic_pipeline, get_part1_models
from src.metrics import (
    evaluate_classifier,
    save_confusion_matrix_png,
    top_confusion_pairs,
)
from src.utils import ensure_dir, save_csv, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part 1: BoW/TF-IDF + classic classifiers on 20 Newsgroups"
    )
    p.add_argument(
        "--vectorizer",
        choices=["bow", "tfidf"],
        default="tfidf",
        help="Vectorizer type: bow=CountVectorizer, tfidf=TfidfVectorizer",
    )
    p.add_argument("--max_features", type=int, default=60000)
    p.add_argument("--ngram_max", type=int, default=2, choices=[1, 2])
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument(
        "--stop_words",
        choices=["none", "english"],
        default="english",
        help="Stopword setting for vectorizer",
    )
    p.add_argument(
        "--remove",
        default="headers,footers,quotes",
        help="Comma-separated: headers,footers,quotes, or empty for none. "
             "Example: --remove headers,quotes",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Base outputs directory",
    )
    p.add_argument(
        "--save_confusion_png",
        action="store_true",
        help="If set, saves a confusion matrix PNG for best model",
    )
    return p.parse_args()


def pretty_print_results(df: pd.DataFrame) -> None:
    # Sort by Macro-F1 desc
    df2 = df.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)

    # Simple console table without extra deps
    cols = ["model", "accuracy", "macro_f1", "train_seconds", "predict_seconds"]
    show = df2[cols].copy()
    show["accuracy"] = show["accuracy"].map(lambda x: f"{x:.4f}")
    show["macro_f1"] = show["macro_f1"].map(lambda x: f"{x:.4f}")
    show["train_seconds"] = show["train_seconds"].map(lambda x: f"{x:.2f}")
    show["predict_seconds"] = show["predict_seconds"].map(lambda x: f"{x:.2f}")

    print("\n=== Part 1 Results (sorted by Macro-F1) ===")
    print(show.to_string(index=False))
    print("")


def main() -> None:
    args = parse_args()

    outputs_base = Path(args.outputs_dir)
    ensure_dir(outputs_base)

    # Per spec: save summary to outputs/part1_metrics.csv (exact path)
    metrics_csv_path = outputs_base / "part1_metrics.csv"

    # Also store artifacts under outputs/part1/
    part_dir = outputs_base / "part1"
    ensure_dir(part_dir)

    remove_tuple: tuple[str, ...] = tuple(
        [x.strip() for x in args.remove.split(",") if x.strip()]
    )

    # ---------- Load data ----------
    data = load_20newsgroups_train_test(remove=remove_tuple)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    target_names = data["target_names"]

    # Confirm dataset size (>= 10,000 rows)
    total_docs = len(X_train) + len(X_test)
    print(f"Loaded 20 Newsgroups:")
    print(f"  Train docs: {len(X_train)}")
    print(f"  Test docs : {len(X_test)}")
    print(f"  Total     : {total_docs} (requirement: >= 10,000)")
    print(f"  Classes   : {len(target_names)} (requirement: > 5)\n")

    # ---------- Build vectorizer settings ----------
    stop_words = None if args.stop_words == "none" else "english"
    vectorizer_cfg: Dict[str, Any] = dict(
        vectorizer_type=args.vectorizer,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        stop_words=stop_words,
    )

    # ---------- Train/Eval each model ----------
    results: List[Dict[str, Any]] = []
    best = None  # (macro_f1, model_name, y_pred, confusion, pipeline)

    models = get_part1_models(seed=args.seed)

    for model_name, estimator in models.items():
        pipe = build_classic_pipeline(
            estimator=estimator,
            **vectorizer_cfg,
        )

        metrics = evaluate_classifier(
            model=pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=list(range(len(target_names))),
        )

        row = {
            "model": model_name,
            "vectorizer": args.vectorizer,
            "max_features": args.max_features,
            "ngram_range": f"(1,{args.ngram_max})",
            "min_df": args.min_df,
            "stop_words": args.stop_words,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "train_seconds": metrics["train_seconds"],
            "predict_seconds": metrics["predict_seconds"],
        }
        results.append(row)

        if best is None or metrics["macro_f1"] > best["macro_f1"]:
            best = {
                "model_name": model_name,
                "macro_f1": metrics["macro_f1"],
                "y_pred": metrics["y_pred"],
                "confusion": metrics["confusion"],
                "pipeline": pipe,
            }

    df = pd.DataFrame(results)
    pretty_print_results(df)

    # ---------- Save summary CSV ----------
    save_csv(df, metrics_csv_path)
    print(f"Saved metrics CSV: {metrics_csv_path}")

    # ---------- Best model artifacts ----------
    assert best is not None
    best_model_name = best["model_name"]
    confusion = best["confusion"]
    y_pred = best["y_pred"]

    # Top 10 confusion pairs
    conf_pairs = top_confusion_pairs(confusion, target_names=target_names, top_k=10)
    conf_pairs_path = part_dir / "top_confusion_pairs.json"
    save_json(conf_pairs, conf_pairs_path)
    print(f"Saved top confusion pairs: {conf_pairs_path}")

    # Print top confusion pairs to console too
    print("\n=== Top 10 Confusion Pairs (off-diagonal) for Best Model ===")
    for item in conf_pairs:
        print(
            f"{item['true_label']} -> {item['pred_label']}: {item['count']} "
            f"(true_idx={item['true_idx']}, pred_idx={item['pred_idx']})"
        )
    print("")

    # Confusion matrix PNG (optional, per your flag)
    if args.save_confusion_png:
        cm_path = part_dir / f"confusion_matrix_best_{best_model_name}.png"
        save_confusion_matrix_png(
            confusion=confusion,
            class_names=target_names,
            out_path=cm_path,
            title=f"Part 1 Best Model: {best_model_name}",
            max_classes_to_show=20,
        )
        print(f"Saved confusion matrix PNG: {cm_path}")

    # Small metadata file for reproducibility
    meta_path = part_dir / "run_metadata.json"
    save_json(
        {
            "best_model": best_model_name,
            "best_macro_f1": best["macro_f1"],
            "vectorizer_cfg": vectorizer_cfg,
            "remove": list(remove_tuple),
            "seed": args.seed,
            "total_docs": total_docs,
            "num_classes": len(target_names),
        },
        meta_path,
    )
    print(f"Saved run metadata: {meta_path}\n")


if __name__ == "__main__":
    main()
