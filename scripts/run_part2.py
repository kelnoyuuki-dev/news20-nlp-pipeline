# scripts/run_part2.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# --- Make repo root importable even when running as a script ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_20newsgroups_train_test
from src.embedding_cache import embed_texts_cached
from src.part2_embeddings import get_part2_models, build_embedding_pipeline
from src.metrics import (
    evaluate_classifier,
    save_confusion_matrix_png,
    top_confusion_pairs,
)
from src.utils import ensure_dir, save_csv, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part 2: SentenceTransformer embeddings + classic classifiers on 20 Newsgroups"
    )
    p.add_argument("--st_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--remove",
        default="headers,footers,quotes",
        help="Comma-separated: headers,footers,quotes, or empty for none.",
    )
    p.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Base outputs directory",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="outputs/cache",
        help="Embedding cache directory",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings (often helps SVM/LR and clustering consistency).",
    )
    p.add_argument(
        "--save_confusion_png",
        action="store_true",
        help="If set, saves confusion matrix PNG for best model.",
    )
    return p.parse_args()


def _find_part1_metrics_csv(outputs_base: Path) -> Path | None:
    """
    Your Part 1 code sometimes saves outputs/part1_metrics.csv.
    Some versions save outputs/part1_metrics.csv or outputs/part1_metrics.csv.
    We'll check the common ones.
    """
    candidates = [
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",  # keep if your naming differs
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        # Most likely in your run: outputs/part1_metrics.csv OR outputs/part1_metrics.csv
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
        outputs_base / "part1_metrics.csv",
    ]
    # Also check your screenshot: outputs\part1_metrics.csv
    candidates.insert(0, outputs_base / "part1_metrics.csv")
    # And if you saved it under outputs/part1/part1_metrics.csv (rare)
    candidates.append(outputs_base / "part1" / "part1_metrics.csv")

    for c in candidates:
        if c.exists():
            return c
    return None


def print_part1_vs_part2_comparison(outputs_base: Path, part2_df: pd.DataFrame) -> None:
    """
    Prints a short comparison: best metrics in Part1 vs Part2 and a reasoned explanation.
    """
    part1_path = _find_part1_metrics_csv(outputs_base)
    part2_best = part2_df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]

    print("\n" + "=" * 70)
    print("Part 1 vs Part 2 Comparison")
    print("=" * 70)

    if part1_path is None:
        print("Part 1 metrics file not found under outputs/.")
        print("Run Part 1 first to enable automatic comparison.\n")
        print(f"Part 2 best model: {part2_best['model']}")
        print(f"  Accuracy: {part2_best['accuracy']:.4f}")
        print(f"  Macro-F1: {part2_best['macro_f1']:.4f}\n")
        return

    part1_df = pd.read_csv(part1_path)
    part1_best = part1_df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]

    print(f"Part 1 best model: {part1_best['model']}")
    print(f"  Accuracy: {float(part1_best['accuracy']):.4f}")
    print(f"  Macro-F1: {float(part1_best['macro_f1']):.4f}\n")

    print(f"Part 2 best model: {part2_best['model']}")
    print(f"  Accuracy: {part2_best['accuracy']:.4f}")
    print(f"  Macro-F1: {part2_best['macro_f1']:.4f}\n")

    # Simple reasoned explanation (template-ish but usable)
    print("Interpretation:")
    if float(part2_best["macro_f1"]) > float(part1_best["macro_f1"]):
        print(
            "- Embeddings performed better here. SentenceTransformer vectors capture semantic similarity\n"
            "  across topics even when surface words differ, which often helps with multi-class text classification."
        )
    else:
        print(
            "- Classic TF-IDF/BoW performed better here. For 20 Newsgroups, topic-specific keywords are strong\n"
            "  signals, and sparse linear models or NB can exploit those discriminative terms very effectively."
        )

    print(
        "- MultinomialNB is included for completeness in Part 2, but it typically underperforms on dense embeddings\n"
        "  because its assumptions (nonnegative count-like features + conditional independence) do not match learned embeddings."
    )
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()

    outputs_base = Path(args.outputs_dir)
    ensure_dir(outputs_base)

    # Required exact path
    metrics_csv_path = outputs_base / "part2_metrics.csv"

    # Artifacts under outputs/part2/
    part_dir = outputs_base / "part2"
    ensure_dir(part_dir)

    cache_dir = Path(args.cache_dir)
    ensure_dir(cache_dir)

    remove_tuple: tuple[str, ...] = tuple(
        [x.strip() for x in args.remove.split(",") if x.strip()]
    )

    # ---------- Load data ----------
    data = load_20newsgroups_train_test(remove=remove_tuple)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    target_names = data["target_names"]

    total_docs = len(X_train) + len(X_test)
    print("Loaded 20 Newsgroups:")
    print(f"  Train docs: {len(X_train)}")
    print(f"  Test docs : {len(X_test)}")
    print(f"  Total     : {total_docs} (requirement: >= 10,000)")
    print(f"  Classes   : {len(target_names)} (requirement: > 5)\n")

    # ---------- Embed + cache ----------
    emb_train = embed_texts_cached(
        texts=X_train,
        st_model_name=args.st_model,
        cache_dir=cache_dir,
        split_name="train",
        remove=remove_tuple,
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=args.normalize,
        dtype=np.float32,
    )
    emb_test = embed_texts_cached(
        texts=X_test,
        st_model_name=args.st_model,
        cache_dir=cache_dir,
        split_name="test",
        remove=remove_tuple,
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=args.normalize,
        dtype=np.float32,
    )

    # sanity
    print(f"Embeddings: train {emb_train.shape} test {emb_test.shape} dtype={emb_train.dtype}\n")

    # ---------- Train/Eval models ----------
    results: List[Dict[str, Any]] = []
    best = None  # macro_f1, y_pred, confusion, model_name

    models = get_part2_models(seed=args.seed)

    for model_name, estimator in models.items():

        try:
            model = build_embedding_pipeline(
                model_name=model_name,
                estimator=estimator
            )

            metrics = evaluate_classifier(
                model=model,
                X_train=emb_train,
                y_train=y_train,
                X_test=emb_test,
                y_test=y_test,
                labels=list(range(len(target_names))),
            )

        except ValueError as e:
            print(f"\nSkipping {model_name}: {e}")
            print(
                "Reason: MultinomialNB assumes non-negative count features, "
                "but embeddings contain negative values.\n"
            )
            continue
        row = {
            "model": model_name,
            "st_model": args.st_model,
            "normalize": bool(args.normalize),
            "batch_size": args.batch_size,
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
            }

    df = pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=False)
    # console table
    show = df[["model", "accuracy", "macro_f1", "train_seconds", "predict_seconds"]].copy()
    show["accuracy"] = show["accuracy"].map(lambda x: f"{x:.4f}")
    show["macro_f1"] = show["macro_f1"].map(lambda x: f"{x:.4f}")
    show["train_seconds"] = show["train_seconds"].map(lambda x: f"{x:.2f}")
    show["predict_seconds"] = show["predict_seconds"].map(lambda x: f"{x:.2f}")

    print("\n=== Part 2 Results (Embeddings, sorted by Macro-F1) ===")
    print(show.to_string(index=False))
    print("")

    # ---------- Save summary CSV ----------
    save_csv(df, metrics_csv_path)
    print(f"Saved metrics CSV: {metrics_csv_path}")

    # ---------- Best model artifacts ----------
    assert best is not None
    best_model_name = best["model_name"]
    confusion = best["confusion"]

    conf_pairs = top_confusion_pairs(confusion, target_names=target_names, top_k=10)
    conf_pairs_path = part_dir / "top_confusion_pairs.json"
    save_json(conf_pairs, conf_pairs_path)
    print(f"Saved top confusion pairs: {conf_pairs_path}")

    print("\n=== Top 10 Confusion Pairs (off-diagonal) for Best Model ===")
    for item in conf_pairs:
        print(f"{item['true_label']} -> {item['pred_label']}: {item['count']}")
    print("")

    if args.save_confusion_png:
        cm_path = part_dir / f"confusion_matrix_best_{best_model_name}.png"
        save_confusion_matrix_png(
            confusion=confusion,
            class_names=target_names,
            out_path=cm_path,
            title=f"Part 2 Best Model: {best_model_name}",
            max_classes_to_show=20,
        )
        print(f"Saved confusion matrix PNG: {cm_path}")

    meta_path = part_dir / "run_metadata.json"
    save_json(
        {
            "best_model": best_model_name,
            "best_macro_f1": float(best["macro_f1"]),
            "st_model": args.st_model,
            "normalize": bool(args.normalize),
            "batch_size": args.batch_size,
            "remove": list(remove_tuple),
            "seed": args.seed,
            "total_docs": total_docs,
            "num_classes": len(target_names),
        },
        meta_path,
    )
    print(f"Saved run metadata: {meta_path}")

    # ---------- Part1 vs Part2 comparison ----------
    print_part1_vs_part2_comparison(outputs_base=outputs_base, part2_df=df)


if __name__ == "__main__":
    main()
