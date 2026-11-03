# Non-Destructive Banana Grade Prediction App

This project is a local Gradio web app for banana quality/ripeness grading. It accepts manual hardware/sensor values and a banana image, extracts image-processing features, combines them with sensor values, and predicts Grade A/B/C using a trained Random Forest model. An ESP32-CAM on the same Wi-Fi can capture a still image into the app.

The current version is local-first: it does not retrain the model, does not require STM32 serial input, and loads the trusted model package from:

```text
models/banana_grade_model_literature_guided_rf_v2.pkl
```

## Prediction Classes

- A: good-quality / ripe / market-acceptable
- B: intermediate / borderline / partially ripe
- C: unripe, overripe, or poor quality

## Current Inputs

Hardware/sensor values entered manually:

- `weight_g`
- `surface_temp_C`
- `ambient_temp_C`
- `relative_humidity_pct`
- `pressure_hPa`
- `gas_resistance_kohm`

Image-derived values extracted by the OpenCV/skimage pipeline:

- `peel_green_pct`
- `peel_yellow_pct`
- `peel_brown_spot_pct`
- `peel_other_pct`
- hue, saturation, brightness, texture, area, and image quality features

## Repository Structure

```text
banana-grade-ml-detection/
|-- app.py
|-- advanced_analysis.py
|-- ftir_reference.py
|-- research_insights.py
|-- requirements.txt
|-- README.md
|-- IMPLEMENTATION_NOTES.md
|-- .gitignore
|-- models/
|   `-- banana_grade_model_literature_guided_rf_v2.pkl
|-- data/ftir/                 # ten original CRF exports, manifest and audit
|-- reports/ftir/              # derived figures, PCA scores/loadings and validation
|-- scripts/                   # reproducible exports and HTTP/API smoke test
|-- tests/                     # regression and scientific-data checks
|-- sample_inputs/
|   `-- README_SAMPLE_INPUTS.md
`-- stm32/
    `-- README_STM32_INTEGRATION.md
```

## Setup on Windows PowerShell

Clone the repository:

```powershell
git clone https://github.com/shubhambar-ve/banana-grade-ml-detection.git
cd banana-grade-ml-detection
```

Create and activate a Python virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```powershell
python app.py
```

Then open on the computer running the app:

```text
http://127.0.0.1:7860
```

To open the app from a phone or another computer on the same Wi-Fi, use the host computer's LAN address, for example `http://192.168.1.20:7860`. Allow Python through the host firewall for private networks if prompted.

## ESP32-CAM photo capture

1. In Arduino IDE, upload the CameraWebServer sketch with the correct camera model selected in `board_config.h`. Enter your Wi-Fi credentials in the sketch and keep them private.
2. Open Serial Monitor at 115200 baud and note the `Camera Ready!` IP address. Open that address in a browser first to confirm the camera server is reachable.
3. In the app, enter that address in **ESP32-CAM address**, then use **Capture photo with flash**. The returned photo appears in the image box; use **Predict Grade** to analyze it.
4. The app sets the camera server's LED intensity to 180, requests the JPEG, then sets LED intensity back to zero. This needs CameraWebServer firmware that supports `/control?var=led_intensity&val=...` and has `LED_GPIO_NUM` defined. The flash GPIO depends on the board; AI-Thinker ESP32-CAM commonly uses GPIO 4. If the app reports that flash control was rejected, confirm the selected camera model and `LED_GPIO_NUM` in `board_config.h`, and verify that the firmware supports the LED intensity control.

The app does not need browser CORS settings because the Python server performs the camera request. Both the computer running the app and ESP32-CAM must be able to reach each other on the local network. You can set the default camera address with `ESP32_CAMERA_URL`; the app also remains usable with uploaded images if the camera is offline.

The app reads STM32 data from COM13 at 115200 baud by default. Close Arduino IDE Serial Monitor while the app is running because only one program can use the COM port at a time. It recognizes lines like `FINAL -> Weight: 0.00 g | Temp: 31.72 C` and updates the weight and surface temperature fields. **Read latest STM32 measurement** also reads any buffered result immediately. Set `STM32_SERIAL_PORT` if the STM32 uses another port. Install dependencies with `pip install -r requirements.txt`, then restart the app.

For a fresh Windows computer, install Python 3.11 and Git, then run the clone and
environment setup above. You can use the virtual environment directly without
activating it:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:STM32_SERIAL_PORT = "COM5"  # replace COM5 with the port shown in Device Manager
.\.venv\Scripts\python.exe app.py
```

Set the port in the same PowerShell window before launching. `COM5` is only an
example, not a detected device. The repository includes the trained model and
all ten FTIR references, so no retraining or separate research-data download is
needed. Python environments and hardware firmware are not included. See the
[current STM32 connection guide](stm32/README_STM32_INTEGRATION.md) for the exact
supported serial format and troubleshooting.

## Troubleshooting

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If package installation fails, upgrade packaging tools and retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If the model file is missing, place this trusted file inside the `models/` folder:

```text
banana_grade_model_literature_guided_rf_v2.pkl
```

## Model Safety

This app loads a local pickle/joblib model file. Only use the provided trusted `.pkl` file. Do not download or load unknown pickle/joblib files because they can execute code when loaded.

## STM32 Status

STM32 is optional. The active app polls textual weight/surface-temperature lines and fills those two editable fields. The remaining four fields are supplied manually. The six-field JSON protocol in [stm32/README_STM32_INTEGRATION.md](stm32/README_STM32_INTEGRATION.md) is a future plan, not the active parser.

## Advanced Quality Analysis & Experimental Research Insights

After **Predict Grade**, the app now adds individual visual/sensor interpretation,
contextually selected **historical** FTIR references, four scientific plot views,
and cited ripening-research context. The original grade, probabilities, all 21
features, image processing and hardware callbacks are preserved.

All ten genuine laboratory exports, `CRF 164.csv` through `CRF 173.csv`, are loaded
from `data/ftir/`. Their IDs, explicit `%T` units, identical 3,601-point
4000–400 cm⁻¹ grids and original checksums are validated against `manifest.csv`.
The source files are read only. Absorbance is calculated as `−log10(%T/100)`.
CRF166's single 100.01% T reading is flagged and retained, including its negative
absorbance. See [the data audit](data/ftir/README.md).

Green appearance emphasizes raw references, yellow appearance emphasizes ripe
references, mixed appearance compares both, and extensive dark coverage uses ripe
references with an explicit absence of a validated overripe class. Image-quality
failure suppresses selective emphasis. These are transparent display rules,
independent of the RF grade. Current weight never selects a historical specimen.
Use **Compare all equally** or the historical specimen selector to explore the
references without taking another photograph or rerunning the RF.

The plots show actual raw/ripe absorbance, paired peel/pulp, the 1200–900 cm⁻¹
fingerprint region, and reproducible exploratory PCA. The new mean-centered
absorbance PCA gives PC1 98.2582% and PC2 1.2985%; the investigators' earlier
66.18%/27.12% result is separately attributed because its preprocessing protocol
was not supplied. The original 93.44% grade-model evidence is preserved.

The application does not measure or reconstruct the photographed banana's FTIR,
infer its carbohydrate concentration, or classify chemical treatment. Historical
B1–B5 treatment histories are unverified. The reported 4/5 exploratory pulp
classification and upper-tail proportion 0.30 remain non-significant historical
context. Research-module failures do not prevent the original grade prediction.

### Run the configured local environment

From this project directory:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:7860`. For a new environment, follow the Python 3.11 setup
above and install `requirements.txt`. The requirements retain the original
dependency list, with compatibility pins for Gradio 5.50.0, scikit-learn 1.6.1
(the artifact's saved version), and NumPy 2.x. No new scientific packages are
required for the extension.

### Tests and reproducible figure exports

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/smoke_test.py
.\.venv\Scripts\python.exe -B scripts/export_ftir_analysis.py
```

The export script writes four figures, PCA scores/loadings and a validation JSON
under `reports/ftir/`, preserving the original CSVs. The smoke test starts and
closes a temporary local server and requires no physical hardware. Full
implementation details and file changes are documented in
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).
