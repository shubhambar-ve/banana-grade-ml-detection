"""Read-only access to historical FTIR exports; no current-sample spectroscopy.

Protocol absorbance-centered-v1 deliberately uses no undocumented baseline,
smoothing, derivative, interpolation, or intensity normalization. It is a new
descriptive PCA, not a reconstruction of the investigators' earlier analysis.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import hashlib
import io
import re

import numpy as np
from matplotlib.figure import Figure


DATA_DIR = Path(__file__).resolve().parent / "data" / "ftir"
PROTOCOL = "absorbance-centered-v1"
EXPECTED_GRID = np.arange(4000.0, 399.0, -1.0)
# Biological labels are supplied by the investigators, not encoded in the CSVs.
EXPERIMENT = {
    "CRF164": ("B1", "Raw", "Peel", 69.0),
    "CRF165": ("B1", "Raw", "Pulp", 69.0),
    "CRF166": ("B3", "Ripe", "Peel", 97.0),
    "CRF167": ("B3", "Ripe", "Pulp", 97.0),
    "CRF168": ("B2", "Raw", "Peel", 68.0),
    "CRF169": ("B2", "Raw", "Pulp", 68.0),
    "CRF170": ("B4", "Ripe", "Peel", 99.0),
    "CRF171": ("B4", "Ripe", "Pulp", 99.0),
    "CRF172": ("B5", "Ripe", "Peel", 119.0),
    "CRF173": ("B5", "Ripe", "Pulp", 119.0),
}
SAMPLE_COLORS = {"B1": "#146348", "B2": "#309b8a", "B3": "#c38008",
                 "B4": "#e27527", "B5": "#865019"}
CAPTION = "Historical B1–B5 laboratory measurements. No spectrum of the current photographed banana is shown."


@dataclass(frozen=True)
class Spectrum:
    crf_id: str
    banana_id: str
    ripeness: str
    tissue: str
    weight_g: float
    filename: str
    sha256: str
    header: str
    wavenumber: np.ndarray
    transmittance: np.ndarray
    absorbance: np.ndarray
    warnings: tuple

    @property
    def label(self):
        return f"{self.banana_id} · {self.crf_id} · {self.ripeness} · {self.tissue}"


@dataclass(frozen=True)
class ReferenceLibrary:
    spectra: tuple = ()
    errors: tuple = ()
    warnings: tuple = ()

    @property
    def complete(self):
        return not self.errors and {s.crf_id for s in self.spectra} == set(EXPERIMENT)

    def audit_records(self):
        return [{
            "Spectrum": s.crf_id, "Banana": s.banana_id,
            "Ripeness": s.ripeness, "Tissue": s.tissue,
            "Historical weight (g)": s.weight_g, "Points": len(s.wavenumber),
            "Grid (cm⁻¹)": "4000 → 400; step −1",
            "Source": "%T", "Min %T": float(s.transmittance.min()),
            "Max %T": float(s.transmittance.max()),
            "Audit": "; ".join(s.warnings) or "Validated",
            "Filename": s.filename, "SHA-256": s.sha256,
        } for s in self.spectra]


def transmittance_to_absorbance(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("%T must be finite and strictly positive; no clipping or imputation is applied.")
    # Values above 100 produce negative absorbance and are retained and flagged.
    return -np.log10(values / 100.0)


def read_spectrum(path, metadata):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("SHA-256 differs from the original export recorded in manifest.csv; review provenance.")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) < 3:
        raise ValueError("Missing metadata, unit header, or numerical data.")
    header = ",".join(rows[0])
    match = re.search(r"\bSample\s+(\d+)\b", header, re.IGNORECASE)
    if not match or f"CRF{match.group(1)}" != metadata["crf_id"]:
        raise ValueError("Instrument-header sample ID disagrees with the manifest.")
    if [v.strip() for v in rows[1]] != ["cm-1", "%T"]:
        raise ValueError("Expected explicit 'cm-1,%T' header; units will not be guessed.")
    if any(len(row) != 2 for row in rows[2:]):
        raise ValueError("Every numerical row must contain exactly two columns; blank rows are invalid.")
    try:
        values = np.asarray(rows[2:], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Non-numerical values in the spectral columns.") from exc
    if values.shape != (3601, 2) or not np.all(np.isfinite(values)):
        raise ValueError("Expected 3601 complete, finite numerical pairs.")
    if not np.array_equal(values[:, 0], EXPECTED_GRID):
        raise ValueError("Expected unique descending 4000–400 cm⁻¹ grid at 1 cm⁻¹; no resampling is performed.")
    absorbance = transmittance_to_absorbance(values[:, 1])
    above = int(np.count_nonzero(values[:, 1] > 100))
    warnings = ()
    if above:
        warnings = (f"{above} point(s) above 100% T (max {values[:, 1].max():.2f}%); negative absorbance retained",)
    for array in (values, absorbance):
        array.setflags(write=False)
    return Spectrum(
        metadata["crf_id"], metadata["banana_id"], metadata["ripeness"],
        metadata["tissue"], float(metadata["weight_g"]), path.name, digest,
        header, values[:, 0], values[:, 1], absorbance, warnings,
    )


def _load_library(directory):
    directory = Path(directory)
    manifest = directory / "manifest.csv"
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["crf_id"] for row in rows]
        if len(ids) != 10 or len(set(ids)) != 10 or set(ids) != set(EXPERIMENT):
            raise ValueError("Manifest must map CRF164–CRF173 exactly once each.")
        filenames = [row["filename"] for row in rows]
        if len(set(filenames)) != 10:
            raise ValueError("Manifest contains duplicate filenames.")
        for row in rows:
            supplied = (row["banana_id"], row["ripeness"], row["tissue"], float(row["weight_g"]))
            if supplied != EXPERIMENT[row["crf_id"]]:
                raise ValueError(f"{row['crf_id']}: biological mapping conflicts with the investigators' supplied mapping.")
            if row["wavenumber_unit"] != "cm-1" or row["source_unit"] != "%T":
                raise ValueError(f"{row['crf_id']}: manifest units are unsupported.")
            if row["mapping_source"] != "investigator_supplied":
                raise ValueError("Mapping provenance must be explicitly investigator_supplied.")
            if Path(row["filename"]).name != row["filename"] or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ValueError("Invalid manifest filename or original-file checksum.")
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        return ReferenceLibrary(errors=(f"Experimental FTIR reference data not loaded: {exc}",))

    spectra, errors, warnings = [], [], []
    declared = set(filenames)
    for extra in sorted(directory.glob("*.csv")):
        if extra.name not in declared | {"manifest.csv"}:
            warnings.append(f"Unmapped CSV not used: {extra.name}")
    for row in sorted(rows, key=lambda item: item["crf_id"]):
        try:
            spectra.append(read_spectrum(directory / row["filename"], row))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{row['crf_id']} ({row['filename']}): {exc}")
    return ReferenceLibrary(tuple(spectra), tuple(errors), tuple(warnings))


@lru_cache(maxsize=4)
def _cached_library(directory, signature):
    return _load_library(directory)


def load_reference_library(directory=DATA_DIR):
    """Cache only while file size/mtime signatures match; reload after edits/removal."""
    directory = Path(directory).resolve()
    try:
        signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                          for p in sorted(directory.glob("*.csv")))
    except OSError as exc:
        return ReferenceLibrary(errors=(f"Experimental FTIR reference data not loaded: {exc}",))
    return _cached_library(str(directory), signature)


def library_status(library):
    if not library.spectra:
        title = "**Experimental FTIR reference data not loaded.**"
    elif library.complete:
        title = "**10/10 authentic exports loaded · 5 biological bananas · paired peel and pulp.**"
    else:
        title = f"**Partial reference library: {len(library.spectra)}/10 spectra loaded. PCA is unavailable.**"
    issues = [*library.errors, *library.warnings,
              *(f"{s.crf_id}: {warning}" for s in library.spectra for warning in s.warnings)]
    return title + ("\n\n" + "\n".join(f"- {issue}" for issue in issues) if issues else "")


def calculate_pca(library):
    """Deterministic SVD on all ten full-range, column-mean-centered absorbances."""
    if not library.complete:
        raise ValueError("PCA requires all ten validated, paired reference spectra on the same grid.")
    matrix = np.stack([s.absorbance for s in library.spectra])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, loadings = np.linalg.svd(centered, full_matrices=False)
    if not np.isfinite(singular).all() or np.count_nonzero(singular > singular[0] * 1e-12) < 2:
        raise ValueError("Insufficient spectral variation for a two-component PCA.")
    scores = u[:, :2] * singular[:2]
    # Resolve the otherwise arbitrary SVD sign for reproducible exported scores.
    for component in range(2):
        pivot = np.argmax(np.abs(loadings[component]))
        if loadings[component, pivot] < 0:
            scores[:, component] *= -1
            loadings[component] *= -1
    return scores, singular[:2] ** 2 / np.sum(singular ** 2), loadings[:2]


def _figure(title, ncols=1, height=4.8):
    fig = Figure(figsize=(12, height), facecolor="white")
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, CAPTION, ha="center", fontsize=8, color="#526171")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.84, wspace=0.25)
    return fig, axes


def _spectrum_axis(ax, limits=(4000, 400)):
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance, A = −log₁₀(%T/100) (dimensionless)", fontsize=9)
    ax.set_xlim(*limits)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _emphasis(spectrum, groups):
    return (1.7, 0.95, 3) if spectrum.ripeness in groups else (0.8, 0.35, 1)


def plot_ripeness_overlay(library, groups=("Raw", "Ripe"), fingerprint=False):
    if not library.spectra:
        return None
    title = ("1200–900 cm⁻¹ carbohydrate fingerprint region" if fingerprint
             else "Measured raw versus ripe references")
    fig, axes = _figure(title + " · emphasis: " + " + ".join(groups), ncols=2)
    for ax, tissue in zip(axes, ("Peel", "Pulp")):
        for s in library.spectra:
            if s.tissue != tissue:
                continue
            width, alpha, zorder = _emphasis(s, groups)
            selected = ((s.wavenumber >= 900) & (s.wavenumber <= 1200)) if fingerprint else slice(None)
            ax.plot(s.wavenumber[selected], s.absorbance[selected], color=SAMPLE_COLORS[s.banana_id],
                    linewidth=width, alpha=alpha, zorder=zorder, label=s.label)
        _spectrum_axis(ax, (1200, 900) if fingerprint else (4000, 400))
        if not fingerprint:
            ax.axvspan(900, 1200, color="#86a7c0", alpha=0.12)
        ax.set_title(tissue)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
    return fig


def plot_tissue_comparison(library, groups=("Raw", "Ripe"), specimen="All bananas"):
    spectra = [s for s in library.spectra if specimen == "All bananas" or s.banana_id == specimen]
    if not spectra:
        return None
    fig, (ax,) = _figure("Paired peel / pulp references · " + specimen)
    for s in spectra:
        width, alpha, zorder = _emphasis(s, groups)
        ax.plot(s.wavenumber, s.absorbance, label=s.label,
                color=SAMPLE_COLORS[s.banana_id], linestyle="-" if s.tissue == "Peel" else "--",
                linewidth=width, alpha=alpha, zorder=zorder)
    _spectrum_axis(ax)
    ax.axvspan(900, 1200, color="#86a7c0", alpha=0.12)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    return fig


def plot_pca(library, groups=("Raw", "Ripe")):
    scores, ratios, _ = calculate_pca(library)
    fig, (ax,) = _figure("Exploratory PCA · ten spectra from five paired bananas", height=5.2)
    span = np.maximum(np.ptp(scores, axis=0), 1e-12)
    for i, s in enumerate(library.spectra):
        _, alpha, zorder = _emphasis(s, groups)
        ax.scatter(*scores[i], color=SAMPLE_COLORS[s.banana_id], marker="o" if s.tissue == "Peel" else "^",
                   s=80, alpha=alpha, zorder=zorder, label=s.label)
        near_previous = i > 0 and np.any(np.linalg.norm((scores[:i] - scores[i]) / span, axis=1) < 0.055)
        offset = (5, -14) if near_previous else (5, 5)
        ax.annotate(f"{s.banana_id}/{s.crf_id}", scores[i], xytext=offset,
                    textcoords="offset points", fontsize=7, alpha=max(alpha, 0.6))
    ax.axhline(0, color="#d1d5db", linewidth=0.6)
    ax.axvline(0, color="#d1d5db", linewidth=0.6)
    ax.set_xlabel(f"PC1 score ({ratios[0]*100:.2f}% variance; centered absorbance units)")
    ax.set_ylabel(f"PC2 score ({ratios[1]*100:.2f}% variance; centered absorbance units)")
    ax.margins(0.22)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=6.5, loc="best", ncol=2)
    return fig, ratios


def pca_description(ratios):
    return (
        f"**Recalculated exploratory PCA — {PROTOCOL}:** PC1 **{ratios[0]*100:.2f}%**, "
        f"PC2 **{ratios[1]*100:.2f}%**, combined **{sum(ratios)*100:.2f}%**. "
        "All 3,601 wavenumbers; %T → absorbance; mean-center each wavenumber across ten spectra; "
        "deterministic SVD. No smoothing, baseline correction, derivatives, interpolation, "
        "or variance scaling. This is descriptive: tissue, overall intensity and baseline "
        "differences can dominate; ten spectra represent only five biological units. "
        "The photographed banana has no PCA coordinate.\n\n"
        "**Earlier investigator-reported analysis:** PC1 66.18%, PC2 27.12%, total 93.30%. "
        "Its preprocessing code and scores were not supplied, so those numbers have not been "
        "reproduced or substituted into this graph. The reported exploratory pulp result "
        "was 4/5 (80%), with exact label-arrangement upper-tail proportion 0.30: "
        "not statistically significant. Those classification results are historical reports, "
        "not recalculated here, and are unrelated to the original 93.44% grade-model evaluation."
    )
