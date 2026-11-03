# FTIR and research extension

## Application boundary

The original model, all 21 features, class meanings, image processing, serial
functions, ESP32 callback, and original fifteen-output `run_prediction()` are
preserved. A new `run_prediction_with_analysis()` adapter calls that function
once, reads its exact unrounded feature DataFrame, and appends ten research
outputs. It does not invoke a second segmentation or RF prediction.

The model's SHA-256 remains:

```text
36e4a97401a96cbfbe27f7f553f906379efa952c79cbce79b989362709dc913a
```

The original approximately 93.44% prototype validation display and its fixed
confusion matrix/importances remain in place, with an explicit provenance note.
They have not been replaced by another experiment's results. All original
function bodies and relevant constants are protected by pre-extension
source-segment hashes in `tests/fixtures/original_contract.json`. The AST locates
each segment, while the hash uses source text to remain stable across Python
versions.

## New panels

1. **Observed in current input:** uses all fifteen image features and six
   submitted sensor values. It describes the actual colour fractions, hue
   distribution, texture and image checks; includes surface-minus-ambient
   temperature; and explicitly identifies uncertain sensor provenance. No
   measured firmness, chemical concentration, or treatment threshold is inferred.
2. **Experimentally measured reference:** shows loader status and the reason for
   contextual raw/ripe reference emphasis, plus an expandable file/mapping audit.
3. **Laboratory comparisons:** four tabs show raw/ripe absorbance overlays,
   paired peel/pulp spectra, the 1200–900 cm⁻¹ region, and exploratory PCA. A
   comparison control restores equal emphasis, and a historical specimen filter
   changes only the paired plot. Both operate on per-session prediction state
   without querying sensors or invoking the RF.
4. **Published research context:** different commentary for green, yellow,
   transitional, dark, and uncertain appearances, with a compact expandable
   bibliography. The exact Odongo reference is verified as Applied Physics B
   129, 84 (2023), DOI `10.1007/s00340-023-08023-w`. The 2026 Chaurakoti work is
   labeled a preprint. NIR, Raman, visible/NIR multispectral, and mid-IR FTIR
   measurements are not equated.

No FTIR upload, new sensor input, artificial-ripening classifier, treatment
probability, risk score, or reconstructed current-sample spectrum was added.

## Contextual selection rules

Rules run in this order and only choose visual emphasis; they are not validated
biological thresholds or a trained model:

| Condition | Emphasis and interpretation |
|---|---|
| Original image-quality checks fail | Both groups; image needs review |
| Brown/dark ≥25% | Ripe references as limited context; no measured overripe class |
| Green and yellow each ≥25% | Both groups; no invented intermediate spectra |
| Green ≥50% | Raw B1/B2 references; ripe references remain visible |
| Yellow ≥50% | Ripe B3/B4/B5 references; raw references remain visible |
| Otherwise | Both groups; no dominant reference pattern |

Weight and RF grade do not choose a specimen. Grade C is explicitly allowed to
represent immature or advanced/poor-quality appearances. Figure emphasis changes
line weight, opacity, or marker emphasis only; numerical spectral curves and PCA
coordinates are never changed to match the image.

## Authentic data and analysis

All CRF164–CRF173 files were found. Each has two numerical columns with explicit
`cm-1,%T` units and 3,601 descending points, 4000–400 cm⁻¹ at 1 cm⁻¹ spacing.
There are no missing, duplicated-grid, or nonfinite numerical entries. Original
CRF files are unchanged. Biological mapping follows the investigators' supplied
table; the CSVs independently verify CRF IDs, not tissue or treatment labels.

CRF166 has one 100.01% T reading at 3736 cm⁻¹. It is flagged and retained,
including its small negative absorbance. `%T` is converted as `−log10(%T/100)`.

The new `absorbance-centered-v1` PCA uses full-range absorbance, column mean
centering, and deterministic SVD, with no undocumented baseline correction,
smoothing, derivative, resampling, or autoscaling. Its PC1/PC2 are 98.2582% /
1.2985%. It does not reproduce the earlier reported 66.18% / 27.12% protocol.
No PLS-DA, SVM, or treatment classifier is fitted. Earlier reported 4/5 pulp
classification and upper-tail proportion 0.30 remain clearly attributed,
non-significant historical context.

See [the full source-data audit and protocol](data/ftir/README.md) and
[machine-readable validation](reports/ftir/validation.json).

## Failure handling

Research imports/loading/rendering have an independent exception boundary. A
failed research module cannot replace the original grade outputs. Missing files
produce explicit availability messages and no fabricated curves; partial curves
identify an incomplete cohort and disable PCA. Individual plotting errors leave
the other plots available. Invalid predictions clear prior research results and
per-session state. All original hardware event handlers remain connected.

## Files

Modified:

- `app.py`: adapter, four-panel UI, state/comparison callbacks, provenance note.
- `requirements.txt`: pins Gradio 5.50.0, scikit-learn 1.6.1, NumPy 2.x.
- `README.md`: launch, analysis, test and export instructions.
- `stm32/README_STM32_INTEGRATION.md`: current two-field serial protocol and
  setup/troubleshooting for a fresh Windows computer; JSON is labeled future work.
- `.gitignore`: excludes local test tools/runtime/caches.

Created:

- `.gitattributes`: prevents Git line-ending conversion of FTIR CSVs, preserving
  the original export checksums across Windows and other systems.
- `advanced_analysis.py`, `ftir_reference.py`, `research_insights.py`.
- `data/ftir/manifest.csv`, `data/ftir/README.md`.
- `tests/test_extension.py`, `tests/fixtures/original_contract.json`,
  `tests/fixtures/original_predictions.json`.
- `scripts/export_ftir_analysis.py`, `scripts/smoke_test.py`.
- `reports/ftir/raw_ripe_absorbance.png`, `peel_pulp_absorbance.png`,
  `fingerprint_1200_900.png`, `pca_scores.png`, `pca_scores.csv`,
  `pca_loadings.csv`, `validation.json`.
- This document.

The supplied ten original CSVs and RF artifact were not modified. Local ignored
`.venv`, `.python`, `.tools` and `.test-cache` directories support testing; they
are not application dependencies or research measurements.

## Reproduce verification

Use Python 3.11 and the project requirements. Tests use standard-library unittest;
the smoke test uses `gradio_client`, already installed with Gradio. No new
scientific dependency is required.

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/smoke_test.py
python -B scripts/export_ftir_analysis.py
```

The regression prediction fixtures were captured by executing the preserved
pre-extension source with UI construction and serial opening omitted. Their
input rows are artificial software test fixtures, not experimental FTIR data.
The tests compare exact source contracts, saved model bytes, feature order,
baseline predictions/probabilities, real-data curve values, PCA algebra, dynamic
selection, missing/bad files, fault isolation, and mocked serial acquisition.

The smoke script launches a temporary loopback-only server, requests its page
and config, submits an artificial RGB software fixture through the real Gradio
API, checks all four real-data plot outputs and per-session comparison controls,
then closes the server. It does not connect to physical hardware. Physical
STM32/ESP32 operation requires the existing external devices and firmware.

## Verification results

- **31 unittest tests passed**, including exact baseline prediction/probability
  comparisons, original source contracts and artifact checksums.
- **Gradio HTTP/API smoke test passed:** page/config returned successfully, a
  prediction returned 24 public outputs (plus internal session state), all four
  historical FTIR plots serialized, and the comparison controls used session
  state without a second RF call. All six event bindings were present.
- **Four authentic-data figures and PCA/audit exports generated successfully.**
- **New-module undefined/unused-name checks passed** (`ruff --select F`). Ruff
  was available transitively in the test environment; it is not required to run
  the application or the documented tests.
- Python **3.11.18**, scikit-learn **1.6.1**, Gradio **5.50.0**, NumPy **2.4.6**.
- A non-failing Windows asyncio `ResourceWarning` about an unclosed event loop
  appeared during the Gradio unit-test run; the API smoke server closed normally.
- Physical hardware was not available for verification. STM32 and ESP32 paths
  were exercised with fake serial data and a mocked HTTP/JPEG transport.

All reference graphs are operational with the supplied data. Independently
reproducing the earlier PCA/classification investigations requires their original
preprocessing code, scores and validation records. Measured intermediate/overripe
comparisons and treatment-history research require additional appropriately
labeled measurements; the current application does not substitute or infer them.
