# src/llm_packets.py
from __future__ import annotations

from pathlib import Path
from typing import List


_PACKET_HEADER = """You are labeling topics for a document clustering project.

TASK:
Given representative documents from ONE cluster, generate an interpretable short topic label.

IMPORTANT OUTPUT FORMAT:
Return STRICT JSON ONLY (no markdown, no commentary) with exactly these fields:
{
  "cluster_id": <int>,
  "label": <string>,
  "rationale": <string>,
  "keywords": <array of strings>
}

RULES:
- label: 3 to 8 words, human-readable, not too broad, not too technical.
- rationale: 1-2 sentences explaining why the label fits the documents.
- keywords: 5-10 keywords or keyphrases (strings), derived from the content.
- If the cluster is mixed, pick the dominant theme and mention mixed nature in rationale.
"""


def _format_docs(representative_docs: List[str], max_chars_each: int = 1200) -> str:
    blocks = []
    for i, doc in enumerate(representative_docs, start=1):
        text = " ".join(doc.split())
        text = text[:max_chars_each]
        blocks.append(f"[Doc {i}]\n{text}")
    return "\n\n".join(blocks)


def write_cluster_label_packet(out_dir: Path, cluster_id: int, representative_docs: List[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / f"cluster_{cluster_id}_label_packet.txt"

    body = (
        _PACKET_HEADER
        + "\n\n"
        + f"CLUSTER_ID: {cluster_id}\n\n"
        + "REPRESENTATIVE DOCUMENTS:\n\n"
        + _format_docs(representative_docs)
        + "\n\n"
        + "NOW RETURN STRICT JSON ONLY.\n"
    )

    packet_path.write_text(body, encoding="utf-8")
    return packet_path


def write_subcluster_label_packet(
    out_dir: Path,
    cluster_id: int,
    subcluster_id: int,
    representative_docs: List[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / f"cluster_{cluster_id}_sub{subcluster_id}_label_packet.txt"

    header = _PACKET_HEADER.replace(
        '"cluster_id": <int>',
        '"cluster_id": <int>,\n  "subcluster_id": <int>'
    )

    body = (
        header
        + "\n\n"
        + f"CLUSTER_ID: {cluster_id}\n"
        + f"SUBCLUSTER_ID: {subcluster_id}\n\n"
        + "REPRESENTATIVE DOCUMENTS:\n\n"
        + _format_docs(representative_docs)
        + "\n\n"
        + "NOW RETURN STRICT JSON ONLY.\n"
    )

    packet_path.write_text(body, encoding="utf-8")
    return packet_path
