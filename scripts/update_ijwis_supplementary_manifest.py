"""Hash supplementary IJWIS experiment sources and publication assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/ijwis_single_gpu_3090/analysis"
OUTPUT = ANALYSIS / "supplementary_asset_manifest.json"
PATHS = (
    "results/ijwis_single_gpu_3090/analysis/bilingual_index_ablation.json",
    "results/ijwis_single_gpu_3090/analysis/governance_history_audit.json",
    "results/ijwis_single_gpu_3090/analysis/rag_evidence_support.json",
    "results/ijwis_single_gpu_3090/analysis/web_load_test.json",
    "paper/ijwis/figures/system_architecture.pdf",
    "paper/ijwis/figures/system_architecture.png",
    "paper/ijwis/figures/knowledge_governance_lifecycle.pdf",
    "paper/ijwis/figures/knowledge_governance_lifecycle.png",
    "paper/ijwis/figures/supplementary_system_validation.png",
    "paper/ijwis/tables/table10_bilingual_index_ablation.csv",
    "paper/ijwis/tables/table11_governance_audit.csv",
    "paper/ijwis/tables/table12_rag_faithfulness.csv",
    "paper/ijwis/tables/table13_web_load_test.csv",
)


def main() -> None:
    artifacts = []
    for relative_path in PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "scope": "IJWIS automated supplementary validation, 2026-08-01",
        "interpretation_boundary": (
            "Expert metadata and human semantic ratings are not included. "
            "BGE-M3 claim support is an automated proxy."
        ),
        "artifacts": artifacts,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
