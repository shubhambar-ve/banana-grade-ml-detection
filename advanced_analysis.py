"""Sample-specific descriptions and historical reference selection, not classification."""

from dataclasses import dataclass
import math

import pandas as pd

import ftir_reference as ftir
from research_insights import FINGERPRINT_CONTEXT, contextual_research


COMPARISON_MODES = ["Automatic emphasis", "Compare all equally"]
SPECIMENS = ["All bananas", "B1", "B2", "B3", "B4", "B5"]


@dataclass(frozen=True)
class ReferenceSelection:
    pattern: str
    title: str
    groups: tuple
    reason: str


def select_reference(features):
    """Transparent presentation rules, not learned/validated ripeness thresholds.

    Priority: failed quality → both; brown >=25 → ripe context; green/yellow
    both >=25 → both; green >=50 → raw; yellow >=50 → ripe; otherwise both.
    Weight, grade, and sensor values never choose a historical specimen.
    """
    needed = ("peel_green_pct", "peel_yellow_pct", "peel_brown_spot_pct", "peel_other_pct",
              "image_quality_score", "background_uniformity_score", "banana_area_fraction")
    if any(key not in features or not math.isfinite(float(features[key])) for key in needed):
        raise ValueError("Reference selection requires finite image features.")
    green, yellow, brown, other = [float(features[key]) for key in needed[:4]]
    if any(value < 0 or value > 100 for value in (green, yellow, brown, other)) or abs(green+yellow+brown+other-100) > 0.1:
        raise ValueError("Peel percentages must lie in 0–100 and sum to 100.")
    quality_ok = (features["image_quality_score"] >= 0.50
                  and features["background_uniformity_score"] >= 0.30
                  and 0.03 <= features["banana_area_fraction"] <= 0.65)
    if not quality_ok:
        return ReferenceSelection("uncertain", "Image needs review", ("Raw", "Ripe"),
                                  "The original image-quality checks did not all pass; both reference groups remain visible without a ripeness emphasis.")
    if brown >= 25:
        return ReferenceSelection("dark", "Substantial brown / dark coverage", ("Ripe",),
                                  f"Brown/dark coverage is {brown:.1f}% (presentation rule ≥25%). Ripe references provide limited context; no separately validated overripe class exists in B1–B5.")
    if green >= 25 and yellow >= 25:
        return ReferenceSelection("mixed", "Mixed green / yellow appearance", ("Raw", "Ripe"),
                                  f"Green {green:.1f}% and yellow {yellow:.1f}% both meet the 25% display rule. Show raw and ripe together; no intermediate-stage spectra were measured in this dataset.")
    if green >= 50:
        return ReferenceSelection("green", "Green-dominant appearance", ("Raw",),
                                  f"Green peel is {green:.1f}% (presentation rule ≥50%); emphasize measured raw B1/B2 references, with ripe spectra retained for comparison.")
    if yellow >= 50:
        return ReferenceSelection("yellow", "Yellow-dominant appearance", ("Ripe",),
                                  f"Yellow peel is {yellow:.1f}% (presentation rule ≥50%); emphasize measured ripe B3/B4/B5 references, with raw spectra retained for comparison.")
    return ReferenceSelection("uncertain", "No dominant reference pattern", ("Raw", "Ripe"),
                              f"Green {green:.1f}%, yellow {yellow:.1f}%, brown/dark {brown:.1f}% and other {other:.1f}% do not meet a specific display rule; compare both groups.")


def describe_current(features, grade, selection):
    f = features
    if not all(math.isfinite(float(value)) for value in f.values()):
        raise ValueError("Extended analysis requires finite input values.")
    interpretation = {
        "green": "The visible surface is predominantly green and looks externally immature. This describes appearance, not internal chemistry.",
        "yellow": "The visible surface is predominantly yellow and looks ripe-like. Review the measured dark coverage and texture alongside that colour impression.",
        "mixed": "Substantial green and yellow regions suggest a visually transitional or unevenly coloured surface; internal ripening is not measured by this image.",
        "dark": "Substantial dark regions may accompany advanced ripening or surface deterioration. The threshold also includes shadows and other dark foreground pixels; cause and spoilage are not established.",
        "uncertain": "A single visual-ripeness description is not supported by the current colour mix or image-quality checks. Inspect the original mask and lighting.",
    }[selection.pattern]
    grade_note = (
        "Grade C can represent either unripe or overripe/poor-quality patterns; the colour observations above distinguish the visible context without redefining that class."
        if grade == "C" else
        "The grade combines all 21 original inputs. Its decision can differ from this colour-based reference-selection rule."
    )
    delta = f["surface_temp_C"] - f["ambient_temp_C"]
    return (
        f"### Observed in current input · {selection.title}\n"
        f"**Original result: Grade {grade}.** {interpretation}\n\n"
        f"- **Peel:** green {f['peel_green_pct']:.2f}%, yellow {f['peel_yellow_pct']:.2f}%, "
        f"brown/dark {f['peel_brown_spot_pct']:.2f}%, other {f['peel_other_pct']:.2f}%.\n"
        f"- **Colour distribution:** hue {f['peel_hue_mean_deg']:.2f}° with standard deviation "
        f"{f['peel_hue_std_deg']:.2f}°; saturation {f['peel_saturation_mean']:.2f}/255; "
        f"brightness {f['peel_brightness_mean']:.2f}/255. Hue spread describes colour variation, "
        "not a validated uniformity grade.\n"
        f"- **Texture:** GLCM contrast {f['peel_texture_contrast']:.3f}, homogeneity "
        f"{f['peel_texture_homogeneity']:.3f}. These describe grey-level patterns, not measured firmness.\n"
        f"- **Image checks:** quality {f['image_quality_score']:.3f}, background uniformity "
        f"{f['background_uniformity_score']:.3f}, sharpness score {f['blur_score']:.3f}, "
        f"exposure {f['exposure_score']:.3f}; segmented area {f['banana_area_fraction']*100:.2f}%.\n\n"
        f"{grade_note}\n\n"
        f"**Supplied sensor values:** weight {f['weight_g']:.2f} g; surface "
        f"{f['surface_temp_C']:.2f} °C; ambient {f['ambient_temp_C']:.2f} °C "
        f"(surface − ambient {delta:+.2f} °C); humidity {f['relative_humidity_pct']:.2f}%; "
        f"pressure {f['pressure_hPa']:.2f} hPa; gas resistance {f['gas_resistance_kohm']:.2f} kΩ. "
        "These are the field values submitted for this prediction. They may be manually entered, "
        "defaults, or serial-populated; this callback does not certify acquisition time or calibration. "
        "Environmental values provide context, gas resistance is not a chemical concentration, "
        "and weight does not establish spectral similarity. No treatment thresholds are applied."
    )


def _plot_safely(name, function, errors):
    try:
        return function()
    except Exception as exc:
        errors.append(f"{name} unavailable: {exc}")
        return None


def build_analysis(features=None, grade=None, comparison="Automatic emphasis", specimen="All bananas", library=None):
    """Ten extension outputs; original grading outputs never enter or leave here."""
    if comparison not in COMPARISON_MODES or specimen not in SPECIMENS:
        raise ValueError("Unknown reference comparison option.")
    if features is None:
        selection = ReferenceSelection("uncertain", "Historical laboratory overview", ("Raw", "Ripe"),
                                       "Run Predict Grade to emphasize references using the current banana's image features.")
        observed = "### Observed in current input\nRun **Predict Grade** to generate the individual visual and sensor interpretation."
        research = "### Published research context\nSample-specific commentary appears after prediction. The literature below is background research, not an analysis of a new banana."
        state = None
    else:
        selection = select_reference(features)
        observed = describe_current(features, grade, selection)
        research = contextual_research(selection.pattern, features)
        state = {"features": dict(features), "grade": grade}
    groups = selection.groups if comparison == "Automatic emphasis" else ("Raw", "Ripe")
    library = ftir.load_reference_library() if library is None else library
    reference = (
        "### Experimentally measured reference\n" + ftir.library_status(library) + "\n\n"
        f"**Why this reference:** {selection.reason}\n\n"
        f"**Graph emphasis:** {' + '.join(groups)}. Thick, opaque curves emphasize the selected group; "
        "the opposite group remains visible. The paired comparison can be filtered by historical "
        "banana ID below; it is not a nearest-sample match.\n\n"
        "Reference selection uses transparent display rules, independently of the Random Forest. "
        "No current spectrum is measured, predicted or reconstructed. Biological labels and weights "
        "come from the investigator's supplied mapping; exports verify the CRF IDs and numerical data.\n\n"
        + FINGERPRINT_CONTEXT
    )
    plot_errors = []
    overlay = _plot_safely("Raw/ripe overlay", lambda: ftir.plot_ripeness_overlay(library, groups), plot_errors)
    paired = _plot_safely("Peel/pulp comparison", lambda: ftir.plot_tissue_comparison(library, groups, specimen), plot_errors)
    fingerprint = _plot_safely("Fingerprint comparison", lambda: ftir.plot_ripeness_overlay(library, groups, True), plot_errors)
    pca_plot = None
    pca_text = "PCA unavailable: all ten original spectra must pass validation; no synthetic data or partial-cohort PCA is substituted."
    if library.complete:
        pca_result = _plot_safely("PCA", lambda: ftir.plot_pca(library, groups), plot_errors)
        if pca_result is not None:
            pca_plot, ratios = pca_result
            pca_text = ftir.pca_description(ratios)
        else:
            pca_text = "PCA could not be calculated. See the reference error details; the spectrum plots and original grade remain independent."
    if plot_errors:
        reference += "\n\n**Plot availability:**\n" + "\n".join(f"- {message}" for message in plot_errors)
    return (observed, reference, overlay, paired, fingerprint, pca_plot, pca_text,
            research, pd.DataFrame(library.audit_records()), state)
