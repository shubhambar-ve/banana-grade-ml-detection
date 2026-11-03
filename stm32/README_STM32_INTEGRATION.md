# STM32 connection guide

## Implemented protocol

The Python application reads USB serial at **115200 baud**. The default port is
`COM13`, overridden by the `STM32_SERIAL_PORT` environment variable at startup.
The existing STM32 firmware must emit newline-terminated text such as:

```text
FINAL -> Weight: 99.80 g | Temp: 32.00 C
```

The `FINAL ->` prefix is optional. `Weight` and `Temp` matching is
case-insensitive. Weight is interpreted as grams and temperature as degrees
Celsius. The app buffers partial lines, keeps the latest matching pair, and
updates `weight_g` and `surface_temp_C` approximately once per second. The
**Read latest STM32 measurement** button also checks buffered readings.

The other four fields remain manually supplied:

- `ambient_temp_C`
- `relative_humidity_pct`
- `pressure_hPa`
- `gas_resistance_kohm`

The grade uses the numeric fields' current values when **Predict Grade** is
clicked. There is no implemented image/measurement timestamp synchronization.

## Connect on a friend's Windows computer

1. Clone the repository and install Python 3.11 dependencies using the main
   README. Create a fresh virtual environment; do not copy another computer's
   `.venv` directory.
2. Connect the STM32 by USB. Check its COM number in Device Manager under
   **Ports (COM & LPT)**. If no serial port appears, check the board's USB/driver
   setup using its vendor documentation.
3. Verify the firmware emits the text format above at 115200 baud. Close any
   Arduino IDE, STM32 tool, or other Serial Monitor before running the app.
4. From the cloned repository, set the actual port and launch in the same
   PowerShell window:

   ```powershell
   $env:STM32_SERIAL_PORT = "COM5"  # example: replace with the actual port
   .\.venv\Scripts\python.exe app.py
   ```

5. Open `http://127.0.0.1:7860`. Check STM32 status and the weight/temperature
   updates. Upload a banana photograph or capture one from the optional ESP32,
   review all six fields, and click **Predict Grade**.

The app also starts without connected hardware and allows manual inputs. A
disconnected status does not disable grading or the historical FTIR section.
The timer may replace manually entered weight/temperature when new serial
readings arrive. Environment variables are read at startup; restart after
changing the port. `.env` files are not automatically loaded.

## Troubleshooting

- **Port unavailable / access denied:** confirm the COM number and close other
  applications using it. Reconnect the device and restart the app if necessary.
- **Connected but waiting:** check the baud rate, labels, separators, units, and
  terminating newline. The app does not parse arbitrary firmware log formats.
- **Missing serial dependency:** install the project requirements in the same
  virtual environment used to launch the application.
- **Unexpected values:** verify units and calibration in the hardware firmware;
  the application does not certify sensor calibration or freshness.

No STM32 firmware, board pin map, sensor driver, or flashing project is included.
The B-L475E-IOT01A1 was named in the earlier integration plan; this repository
does not identify or configure a connected board model.

## Future protocol, not implemented

The earlier plan proposed six-field newline-delimited JSON:

```json
{"weight_g":99.8,"surface_temp_C":32.0,"ambient_temp_C":33.0,"relative_humidity_pct":58.0,"pressure_hPa":1008.0,"gas_resistance_kohm":160.0}
```

The active parser does **not** accept this JSON protocol. Supporting it would
require a separate future implementation; use the current two-field text format
for this version.
