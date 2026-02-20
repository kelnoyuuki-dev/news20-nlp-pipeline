# scripts/run_part3.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# --- Make repo root importable even when running as a script ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_20newsgroups_train_test
from src.embedding_cache import embed_texts_cached
from src.cluster_utils import (
    compute_elbow_inertia,
    choose_k_by_max_curvature,
    fit_kmeans,
    representatives_closest_to_centroid,
    tfidf_fallback_label,     # <-- uses your improved version in cluster_utils.py
    save_elbow_plot,
)
from src.llm_packets import (
    write_cluster_label_packet,
    write_subcluster_label_packet,
)
from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part 3: KMeans clustering + 2-level topic tree with LLM labeling packets"
    )
    p.add_argument("--st_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--remove",
        default="headers,footers,quotes",
        help="Comma-separated: headers,footers,quotes, or empty for none.",
    )

    p.add_argument("--outputs_dir", type=str, default="outputs")
    p.add_argument("--cache_dir", type=str, default="outputs/cache")

    # clustering
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=9)
    p.add_argument(
        "--k_override",
        type=int,
        default=None,
        help="If set, skip elbow selection and use this K (must be < 10).",
    )
    p.add_argument("--rep_docs", type=int, default=8, help="Representative docs per cluster/subcluster")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings before clustering (recommended).",
    )

    # labeling mode
    p.add_argument(
        "--label_mode",
        choices=["packet", "fallback"],
        default="packet",
        help="packet = write LLM prompt packets + also compute fallback TF-IDF label; "
             "fallback = only TF-IDF labels (no packets).",
    )

    # optional viz
    p.add_argument(
        "--umap",
        action="store_true",
        help="If set, save UMAP scatter plots (requires umap-learn).",
    )

    return p.parse_args()


def _tree_to_text(tree: Dict[str, Any]) -> str:
    lines: List[str] = []
    for cl in tree["clusters"]:
        lines.append(f"Cluster {cl['cluster_id']}: {cl['label']}")
        if cl.get("subclusters"):
            for sub in cl["subclusters"]:
                lines.append(f"  - Sub{sub['subcluster_id']}: {sub['label']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outputs_base = Path(args.outputs_dir) / "part3"
    ensure_dir(outputs_base)

    cache_dir = Path(args.cache_dir)
    ensure_dir(cache_dir)

    # Part 3 outputs (ALL inside outputs/part3/)
    elbow_path = outputs_base / "part3_elbow.png"
    top_clusters_json = outputs_base / "part3_top_clusters.json"
    subclusters_json = outputs_base / "part3_subclusters.json"
    tree_txt_path = outputs_base / "topic_tree.txt"
    packets_dir = outputs_base / "llm_packets"
    ensure_dir(packets_dir)

    remove_tuple: Tuple[str, ...] = tuple(
        [x.strip() for x in args.remove.split(",") if x.strip()]
    )

    # ---------- Load data (use all docs = train + test) ----------
    data = load_20newsgroups_train_test(remove=remove_tuple)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    target_names = data["target_names"]

    texts_all = list(X_train) + list(X_test)

    print("Loaded 20 Newsgroups for clustering:")
    print(f"  Train docs: {len(X_train)}")
    print(f"  Test docs : {len(X_test)}")
    print(f"  Total     : {len(texts_all)} (requirement: >= 10,000)")
    print(f"  Classes   : {len(target_names)} (requirement: > 5)\n")

    # ---------- Embed + cache ----------
    emb_all = embed_texts_cached(
        texts=texts_all,
        st_model_name=args.st_model,
        cache_dir=cache_dir,
        split_name="all",
        remove=remove_tuple,
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=args.normalize,
        dtype=np.float32,
    )
    print(f"Embeddings: all {emb_all.shape} dtype={emb_all.dtype} normalize={bool(args.normalize)}\n")

    # ---------- Elbow: K=2..9 ----------
    k_min = int(args.k_min)
    k_max = int(args.k_max)
    if not (2 <= k_min <= k_max <= 9):
        raise ValueError("K range must be within [2..9].")

    ks = list(range(k_min, k_max + 1))
    inertias = compute_elbow_inertia(X=emb_all, ks=ks, seed=args.seed)
    save_elbow_plot(ks, inertias, elbow_path)
    print(f"Saved elbow plot: {elbow_path}")

    if args.k_override is not None:
        K = int(args.k_override)
        if K < 2 or K > 9:
            raise ValueError("--k_override must be in [2..9].")
        print(f"Selected K (override): {K}\n")
    else:
        K = choose_k_by_max_curvature(ks, inertias)
        print(f"Selected K (elbow max-curvature heuristic): {K}\n")

    # ---------- Top-level clustering ----------
    kmeans_top = fit_kmeans(emb_all, n_clusters=K, seed=args.seed)
    top_labels = kmeans_top.labels_
    centers = kmeans_top.cluster_centers_

    # sizes
    cluster_sizes: List[Tuple[int, int]] = []
    for cid in range(K):
        idx = np.where(top_labels == cid)[0]
        cluster_sizes.append((cid, int(idx.size)))

    # Build top cluster summaries
    top_cluster_summaries: List[Dict[str, Any]] = []
    for cid, size in sorted(cluster_sizes, key=lambda x: x[0]):
        idx = np.where(top_labels == cid)[0]

        reps = representatives_closest_to_centroid(
            X=emb_all,
            indices=idx,
            centroid=centers[cid],
            texts=texts_all,
            top_n=args.rep_docs,
        )

        fallback = tfidf_fallback_label([texts_all[i] for i in idx], top_k=8)
        label = fallback["label"]

        packet_path = None
        if args.label_mode == "packet":
            packet_path = write_cluster_label_packet(
                out_dir=packets_dir,
                cluster_id=cid,
                representative_docs=[r["text"] for r in reps],
            )

        top_cluster_summaries.append(
            {
                "cluster_id": cid,
                "size": size,
                "label_mode": args.label_mode,
                "label": label,
                "fallback_keywords": fallback["keywords"],
                "representatives": reps,
                "llm_packet_path": str(packet_path) if packet_path else None,
            }
        )

    save_json(top_cluster_summaries, top_clusters_json)
    print(f"Saved top-level cluster summaries: {top_clusters_json}")

    # ---------- Optional UMAP viz (top-level) ----------
    if args.umap:
        from src.cluster_viz import plot_umap_clusters

        viz_path = outputs_base / "cluster_scatter_umap_top.png"
        plot_umap_clusters(
            X=emb_all,
            labels=top_labels,
            out_path=viz_path,
            seed=args.seed,
            title=f"UMAP: Top-level KMeans (K={K})",
        )
        print(f"Saved UMAP cluster scatter: {viz_path}")

    # ---------- Identify 2 largest clusters ----------
    largest_two = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)[:2]
    largest_two_ids = [largest_two[0][0], largest_two[1][0]]
    print(f"Two largest clusters: {largest_two_ids} (sizes: {[largest_two[0][1], largest_two[1][1]]})\n")

    # ---------- Second-level clustering for 2 biggest clusters ----------
    subcluster_results: Dict[str, Any] = {"parent_clusters": []}
    tree = {"clusters": []}

    top_by_id = {c["cluster_id"]: c for c in top_cluster_summaries}

    for cid in range(K):
        node = {
            "cluster_id": cid,
            "label": top_by_id[cid]["label"],
            "size": top_by_id[cid]["size"],
            "subclusters": [],
        }

        # Only split the two biggest clusters
        if cid not in largest_two_ids:
            tree["clusters"].append(node)
            continue

        idx_parent = np.where(top_labels == cid)[0]
        X_parent = emb_all[idx_parent]

        kmeans_sub = fit_kmeans(X_parent, n_clusters=3, seed=args.seed)
        sub_labels = kmeans_sub.labels_
        sub_centers = kmeans_sub.cluster_centers_

        parent_entry = {
            "cluster_id": cid,
            "parent_label": top_by_id[cid]["label"],
            "parent_size": int(idx_parent.size),
            "subclusters": [],
        }

        for sid in range(3):
            local_idx = np.where(sub_labels == sid)[0]
            global_idx = idx_parent[local_idx]  # map back to original doc indices

            reps = representatives_closest_to_centroid(
                X=emb_all,
                indices=global_idx,
                centroid=sub_centers[sid],
                texts=texts_all,
                top_n=args.rep_docs,
            )

            fallback = tfidf_fallback_label([texts_all[i] for i in global_idx], top_k=8)
            sub_label = fallback["label"]

            packet_path = None
            if args.label_mode == "packet":
                packet_path = write_subcluster_label_packet(
                    out_dir=packets_dir,
                    cluster_id=cid,
                    subcluster_id=sid,
                    representative_docs=[r["text"] for r in reps],
                )

            parent_entry["subclusters"].append(
                {
                    "subcluster_id": sid,
                    "size": int(global_idx.size),
                    "label_mode": args.label_mode,
                    "label": sub_label,
                    "fallback_keywords": fallback["keywords"],
                    "representatives": reps,
                    "llm_packet_path": str(packet_path) if packet_path else None,
                }
            )

            node["subclusters"].append(
                {
                    "subcluster_id": sid,
                    "label": sub_label,
                    "size": int(global_idx.size),
                }
            )

        subcluster_results["parent_clusters"].append(parent_entry)
        tree["clusters"].append(node)

        # ---------- Optional UMAP viz (subclusters) ----------
        if args.umap:
            from src.cluster_viz import plot_umap_clusters

            sub_viz_path = outputs_base / f"cluster_scatter_umap_sub_parent{cid}.png"
            plot_umap_clusters(
                X=X_parent,           # IMPORTANT: use parent subset embeddings
                labels=sub_labels,     # labels are 0..2 within this parent
                out_path=sub_viz_path,
                seed=args.seed,
                title=f"UMAP: Subclusters for parent cluster {cid} (k=3)",
            )
            print(f"Saved UMAP subcluster scatter: {sub_viz_path}")

    # Save subclusters
    save_json(subcluster_results, subclusters_json)
    print(f"Saved subcluster summaries: {subclusters_json}")

    # ---------- Display + save tree ----------
    tree_text = _tree_to_text(tree)
    print("\n" + "=" * 70)
    print("Topic Tree (fallback labels; replace with LLM labels after you paste packets)")
    print("=" * 70)
    print(tree_text)

    tree_txt_path.write_text(tree_text, encoding="utf-8")
    print(f"Saved topic tree text: {tree_txt_path}\n")

    if args.label_mode == "packet":
        print(f"LLM labeling packets written to: {packets_dir}")
        print("Paste each packet into ChatGPT and save the returned JSON outputs for reporting.\n")


if __name__ == "__main__":
    main()
