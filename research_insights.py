"""Curated literature context. No treatment labels, scores, or chemical detector.

Bibliographic metadata checked against publisher pages/arXiv on 2026-11-03.
The 2019 source is explicitly a review; the 2026 source is a preprint.
"""

REFERENCES = (
    ("Zaukuu et al. (2024)",
     "Combining NIR spectroscopy with chemometrics for discriminating naturally ripened banana and calcium carbide ripened banana",
     "npj Science of Food 8, 86; original NIR study",
     "https://doi.org/10.1038/s41538-024-00327-1"),
    ("Maduwanthi & Marapana (2019)",
     "Induced Ripening Agents and Their Effect on Fruit Quality of Banana",
     "International Journal of Food Science, 2520179; review",
     "https://doi.org/10.1155/2019/2520179"),
    ("Odongo, Kaniu, Ndung’u & Wanjohi (2023)",
     "Direct and rapid screening of calcium carbide in ripened bananas using chemometrics-assisted laser Raman spectroscopy",
     "Applied Physics B 129, 84; original Raman study, published 6 May 2023",
     "https://doi.org/10.1007/s00340-023-08023-w"),
    ("Chaurakoti, H. Kumar, Hani Kumar, Singh & Asrey (2026)",
     "A Multispectral Framework for the Detection of Calcium Carbide-Induced Ripening and Shelf-Life Estimation in Climacteric Fruits",
     "arXiv:2608.13073v1, submitted 13 August 2026; preprint, not treated as peer-reviewed evidence",
     "https://arxiv.org/abs/2608.13073v1"),
    ("Banana ripening ATR-FTIR study (2023)",
     "Quality Assessment of Banana Ripening Stages by Combining Analytical Methods and Image Analysis",
     "Applied Sciences 13(6), 3533; original ATR-FTIR / image-analysis study",
     "https://doi.org/10.3390/app13063533"),
)

FINGERPRINT_CONTEXT = (
    "The highlighted 1200–900 cm⁻¹ interval supports comparison of carbohydrate-related "
    "spectral structure in the historical tissue measurements. Published banana ATR-FTIR "
    "work investigated ripening-associated starch/sugar changes "
    "([2023 study](https://doi.org/10.3390/app13063533)). "
    "This display assigns neither a chemical concentration nor a treatment-specific peak "
    "to the photographed banana."
)


def literature_markdown():
    entries = [f"- **{authors}.** [{title}]({url}). {venue}."
               for authors, title, venue, url in REFERENCES]
    return "\n\n".join(entries) + (
        "\n\n**Scope:** Zaukuu uses NIR; Odongo uses Raman; Chaurakoti uses "
        "visible/NIR multispectral measurements (410–940 nm). None supplies a validated "
        "RGB-to-mid-IR conversion for this application. The 2019 review is background context. "
        "External performance figures are not transferred to our samples or our grade model."
    )


def contextual_research(pattern, features):
    green = features["peel_green_pct"]
    yellow = features["peel_yellow_pct"]
    brown = features["peel_brown_spot_pct"]
    lead = {
        "green": (
            f"The current **{green:.1f}% green peel** supports a green-dominant visual description. "
            "It does not establish an untreated history or reveal whether ripening agents were applied."
        ),
        "yellow": (
            f"The current **{yellow:.1f}% yellow peel** is compatible with a ripe-like external appearance. "
            "Yellow colour alone cannot distinguish natural ripening from externally induced ripening."
        ),
        "mixed": (
            f"With **{green:.1f}% green and {yellow:.1f}% yellow**, the visible surface is mixed. "
            "A visually transitional appearance does not quantify internal biochemical changes "
            "or identify the ripening method."
        ),
        "dark": (
            f"The **{brown:.1f}% brown/dark region** warrants attention to surface condition. "
            "Darkening can accompany advanced ripening or deterioration, but shadows and the "
            "segmentation also affect this measurement; it is not evidence of chemical ripening."
        ),
        "uncertain": (
            f"At **{green:.1f}% green, {yellow:.1f}% yellow and {brown:.1f}% brown/dark**, "
            "the image checks or colour mix do not support a confident visual reference emphasis. "
            "A treatment interpretation would be unsupported."
        ),
    }[pattern]
    return (
        "### Published research context\n" + lead + "\n\n"
        "Ethylene participates in ripening; ethephon can release ethylene, while calcium carbide "
        "in contact with moisture releases acetylene. These mechanisms do not make peel colour "
        "a treatment assay ([Maduwanthi & Marapana, 2019](https://doi.org/10.1155/2019/2520179)).\n\n"
        "Published discrimination experiments used additional measurements: "
        "[NIR spectroscopy](https://doi.org/10.1038/s41538-024-00327-1), "
        "[Raman spectroscopy](https://doi.org/10.1007/s00340-023-08023-w), or "
        "[visible/NIR multispectral sensing (2026 preprint)](https://arxiv.org/abs/2608.13073v1). "
        "Raman shifts cannot be treated as mid-infrared FTIR measurements. Pure ethephon or "
        "carbide spectra are not equivalent to treated banana-tissue spectra.\n\n"
        "**Interpretation boundary:** carbohydrate-related ripening changes do not establish "
        "chemical treatment. B1–B5 have unverified treatment histories. Neither the supplied "
        "gas-resistance value nor Grade A/B/C probabilities measure treatment likelihood. "
        "This application produces no chemical-treatment verdict or score."
    )
