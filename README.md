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
banana-grade-ml-app/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- models/
|   `-- banana_grade_model_literature_guided_rf_v2.pkl
|-- sample_inputs/
|   `-- README_SAMPLE_INPUTS.md
`-- stm32/
    `-- README_STM32_INTEGRATION.md
```

## Setup on Windows PowerShell

Clone the repository:

```powershell
git clone https://github.com/Shreerang4/banana-grade-ml-app.git
cd banana-grade-ml-app
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

STM32 integration is prepared but not required in this repo version. The app currently uses manual hardware input fields. See [stm32/README_STM32_INTEGRATION.md](stm32/README_STM32_INTEGRATION.md) for the planned serial JSON format.
