# Authentic laboratory FTIR references

The ten `CRF *.csv` files are original investigator-supplied laboratory exports.
The application and export script read them without modifying or renaming them.
`manifest.csv` records the supplied biological mapping and original SHA-256 hashes.

## Verified export structure

Each file contains:

1. One instrument metadata line identifying `Sample NNN By CRF`, dated Friday,
   May 15, 2026. This line has three comma-separated fields.
2. A two-column unit header: `cm-1,%T`.
3. Exactly 3,601 finite numerical pairs: 4000.00 down to 400.00 cm⁻¹,
   at −1.00 cm⁻¹ steps, with no duplicates, gaps, or missing measurements.

The first column is **wavenumber**, not wavelength. The second is **percentage
transmittance**, not fractional transmission or absorbance. Sampling spacing is
not the same as instrument optical resolution; resolution and acquisition mode
are not specified by these files.

| Export | Banana | Stage | Tissue | Historical weight (g) | Minimum %T | Maximum %T |
|---|---|---|---|---:|---:|---:|
| CRF164 | B1 | Raw | Peel | 69 | 65.46 | 98.66 |
| CRF165 | B1 | Raw | Pulp | 69 | 32.14 | 99.04 |
| CRF166 | B3 | Ripe | Peel | 97 | 94.31 | 100.01 |
| CRF167 | B3 | Ripe | Pulp | 97 | 31.11 | 99.48 |
| CRF168 | B2 | Raw | Peel | 68 | 90.51 | 99.51 |
| CRF169 | B2 | Raw | Pulp | 68 | 30.10 | 99.61 |
| CRF170 | B4 | Ripe | Peel | 99 | 72.37 | 98.00 |
| CRF171 | B4 | Ripe | Pulp | 99 | 30.88 | 99.11 |
| CRF172 | B5 | Ripe | Peel | 119 | 94.47 | 99.58 |
| CRF173 | B5 | Ripe | Pulp | 119 | 66.26 | 99.85 |

**Mapping provenance:** the instrument headers verify the CRF numbers, but do not
encode tissue, biological banana ID, weight, ripeness, or treatment history.
Those labels come from the investigators' explicit experimental mapping. No
independent laboratory notebook or acquisition-method record was supplied.

## Conversion and numerical exception

Absorbance is calculated in memory as:

```text
A = −log10(%T / 100)
```

All source transmittances are strictly positive. CRF166 has **one** point at
100.01% T, at **3736 cm⁻¹**. Its small negative absorbance is mathematically valid
for the recorded value and is retained. The loader reports this exception; it
does not clip, smooth, impute, or alter the source measurement. The export does
not establish the cause of the reading above 100%.

Plots use the converted absorbance without baseline correction, smoothing,
normalization, or chemical-specific peak assignment. Raw/ripe overlays are split
by tissue; paired plots identify both the specimen and CRF number. The
1200–900 cm⁻¹ view contains the original 301 measured points from that interval.

## Reproducible exploratory PCA

Protocol identifier: **`absorbance-centered-v1`**.

1. Require all ten validated exports on the identical sampling grid.
2. Order rows by CRF ID and retain all 3,601 wavenumbers.
3. Convert %T to absorbance, including the flagged negative value.
4. Subtract each wavenumber's mean across the ten spectra.
5. Apply NumPy full-matrix-reduced SVD, without variance scaling.
6. Resolve component signs so the largest absolute loading is positive.
7. Calculate explained variance from squared singular values.

With the supplied data, PC1 explains **98.2582%** and PC2 **1.2985%**
(combined **99.5567%**). This is a new descriptive analysis. Broad intensity,
baseline, tissue, and specimen differences can dominate it. The ten spectra
represent only **five paired biological units**, not ten independent bananas.

The earlier investigator-reported PC1 66.18%, PC2 27.12% (93.30% combined) is
retained as historical context. Its preprocessing protocol and scores were not
provided, so this implementation does not claim to reproduce it. Likewise,
the reported pulp classification result of 4/5 (80%) and exact label-arrangement
upper-tail proportion 0.30 are historical reports, not recalculated classifier
results. They are not statistically significant evidence of treatment detection.

## Reproduce the artifacts

From the project root, using the configured Python environment:

```powershell
python scripts/export_ftir_analysis.py
```

This creates four PNG figures, PCA scores, PCA loadings, and a JSON validation
record under `reports/ftir/`. An alternative directory can be supplied with
`--output`. Writing derived artifacts inside `data/ftir/` is rejected.

Missing or inconsistent files are reported individually. Valid remaining curves
may still be displayed as a clearly labeled partial library, but PCA requires
all ten exports. A manifest mapping conflict prevents all mapped curves from
being used. Checksum changes require explicit provenance review; the loader
never silently accepts altered exports.

These references do not measure the current photographed banana's spectrum,
carbohydrate concentration, chemical residues, or treatment history. B1–B5 lack
verified natural/artificial treatment labels.
