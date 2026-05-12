# Non-Destructive Banana Grade Prediction App

This project is a local Gradio web app for banana quality/ripeness grading. It accepts manual hardware/sensor values and a banana image, extracts image-processing features, combines them with sensor values, and predicts Grade A/B/C using a trained Random Forest model.

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

Then open:

```text
http://127.0.0.1:7860
```

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
