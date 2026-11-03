"""Launch a temporary loopback-only Gradio server and exercise its real API.

The artificial RGB fixture tests the application only, not scientific accuracy.
No physical hardware is accessed. The server is closed when the test finishes.
"""

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["STM32_SERIAL_PORT"] = "COM_NOT_A_REAL_PORT_TEST"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".test-cache" / "matplotlib"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".test-cache" / "gradio"))

import cv2
import numpy as np
from gradio_client import Client, handle_file
import app


def main():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        app.demo.launch(server_name="127.0.0.1", server_port=port, share=False,
                        prevent_thread_lock=True, quiet=True, inbrowser=False)
        base = f"http://127.0.0.1:{port}"
        with urlopen(base, timeout=10) as response:
            assert response.status == 200, "Gradio page did not load"
        with urlopen(base + "/config", timeout=10) as response:
            config = json.load(response)
        assert len(config["dependencies"]) == 6
        client = Client(base, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((220, 320, 3), 240, dtype=np.uint8)
            cv2.ellipse(image, (160, 110), (100, 34), -15, 0, 360, (35, 190, 225), -1)
            path = Path(directory) / "artificial_rgb_software_test_only.png"
            cv2.imwrite(str(path), image)
            result = client.predict(handle_file(str(path)), 120, 27, 26, 55, 1013.25, 80,
                                    "Automatic emphasis", "All bananas",
                                    api_name="/run_prediction_with_analysis")
            # gr.State is managed server-side and not returned to API clients.
            assert len(result) == 24, len(result)
            assert "Predicted Grade:" in result[0], result[0]
            assert "10/10 authentic exports loaded" in result[16], result[16]
            assert "Observed in current input" in result[15]
            assert "absorbance-centered-v1" in result[21]
            assert "no chemical-treatment verdict or score" in result[22]
            for index in (17, 18, 19, 20):
                assert result[index] is not None, f"Missing plot {index}"
            compared = client.predict("Compare all equally", "B3", api_name="/update_reference_comparison")
            assert len(compared) == 9
            assert "Graph emphasis:** Raw + Ripe" in compared[1]
            # New prediction and comparison are exercised through the Gradio transport,
            # including conversion of all four real-spectra Matplotlib outputs.
            print("PASS: HTTP page/config, prediction, four FTIR plots, per-session comparison state.")
            print(result[0])
            print(f"PASS: {len(result)} public prediction outputs; {len(config['dependencies'])} event bindings.")
    finally:
        app.demo.close()


if __name__ == "__main__":
    main()
