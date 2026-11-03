"""Reproduce figures and PCA audit artifacts from the immutable laboratory CSVs.

Run from the project root: python scripts/export_ftir_analysis.py
Only the selected output directory is written; original spectra are read only.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".test-cache" / "matplotlib"))

import numpy as np
import ftir_reference as ftir


def export(output):
    output = Path(output).resolve()
    if output == ftir.DATA_DIR or ftir.DATA_DIR in output.parents:
        raise ValueError("Derived artifacts must not be written inside the original data/ftir directory.")
    library = ftir.load_reference_library()
    if not library.complete:
        raise ValueError(ftir.library_status(library))
    output.mkdir(parents=True, exist_ok=True)
    scores, ratios, loadings = ftir.calculate_pca(library)
    figures = {
        "raw_ripe_absorbance.png": ftir.plot_ripeness_overlay(library),
        "peel_pulp_absorbance.png": ftir.plot_tissue_comparison(library),
        "fingerprint_1200_900.png": ftir.plot_ripeness_overlay(library, fingerprint=True),
        "pca_scores.png": ftir.plot_pca(library)[0],
    }
    for name, figure in figures.items():
        figure.savefig(output / name, dpi=160)
    with (output / "pca_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crf_id", "banana_id", "ripeness", "tissue", "PC1", "PC2"])
        for spectrum, row in zip(library.spectra, scores):
            writer.writerow([spectrum.crf_id, spectrum.banana_id, spectrum.ripeness, spectrum.tissue, *row])
    with (output / "pca_loadings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavenumber_cm-1", "PC1_loading", "PC2_loading"])
        writer.writerows(zip(ftir.EXPECTED_GRID, *loadings))
    audit = {
        "protocol": ftir.PROTOCOL,
        "conversion": "A = -log10(%T/100); values >100% retained and flagged",
        "pca": "Full 4000–400 cm-1 absorbance; column mean-centering; no scaling/smoothing/baseline correction; SVD",
        "pca_sign": "Largest absolute loading in each component is positive",
        "explained_variance_ratio": ratios.tolist(),
        "numpy_version": np.__version__,
        "spectra_count": len(library.spectra),
        "biological_bananas": 5,
        "mapping_provenance": "Investigators' supplied mapping; CSV headers verify CRF IDs, not tissue or ripeness",
        "treatment_history": "Unverified; no treatment classifier",
        "historical_reported_pca": {"PC1_pct": 66.18, "PC2_pct": 27.12, "reproduced": False},
        "historical_reported_pulp_classification": {"correct": 4, "total": 5, "upper_tail_proportion": 0.30, "recomputed": False},
        "source_files": library.audit_records(),
        "manifest_sha256": hashlib.sha256((ftir.DATA_DIR / "manifest.csv").read_bytes()).hexdigest(),
    }
    (output / "validation.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(figures)} authentic-data figures, PCA scores/loadings and validation audit to {output}")
    print(f"PCA: {ratios[0]*100:.4f}%, {ratios[1]*100:.4f}%")
    print(ftir.library_status(library))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "ftir")
    export(parser.parse_args().output)
