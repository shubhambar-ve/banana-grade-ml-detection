# STM32 Integration Plan

## Current Version

- The app uses manual hardware input fields.
- STM32 serial input is not required for the current repo version.
- The trained model prediction path is already prepared to accept the same hardware feature names.

## Future Version

- STM32 B-L475E-IOT01A1 will send readings to the app through USB serial.
- The Python app will read one JSON object per line from the serial port.
- Those values will populate the same feature fields currently entered manually.

Expected sensor readings:

- `weight_g`
- `surface_temp_C`
- `ambient_temp_C`
- `relative_humidity_pct`
- `pressure_hPa`
- `gas_resistance_kohm`

Expected JSON line from STM32:

```json
{"weight_g":99.8,"surface_temp_C":32.0,"ambient_temp_C":33.0,"relative_humidity_pct":58.0,"pressure_hPa":1008.0,"gas_resistance_kohm":160.0}
```

Each JSON line should end with newline `\n`.

Planned Python serial dependency:

```powershell
pip install pyserial
```

STM32 integration will be added after the local app is confirmed working on both machines.
