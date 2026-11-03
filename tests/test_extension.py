"""Regression, scientific-provenance and integration tests; no physical hardware.

Artificial RGB fixtures exercise software paths only. All spectral plot tests
use copies of the real CRF exports, never synthetic laboratory spectra.
"""

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["STM32_SERIAL_PORT"] = "COM_NOT_A_REAL_PORT_TEST"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".test-cache" / "matplotlib"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".test-cache" / "gradio"))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ftir_reference as ftir
import advanced_analysis as advanced
import app


def feature_fixture(pattern="green"):
    colors = {
        "green": (80, 15, 2, 3), "yellow": (5, 88, 4, 3),
        "mixed": (40, 50, 5, 5), "dark": (5, 35, 55, 5),
        "uncertain": (5, 5, 5, 85),
    }
    green, yellow, brown, other = colors[pattern]
    return dict(zip(app.EXPECTED_FEATURE_COLS, [
        120, 27, 26, 55, 1013.25, 80,
        green, yellow, brown, other, 90, 20, 140, 180, 5, .8,
        .25, .95, .8, .8, .86,
    ]))


def image_fixture():
    """Artificial test image, not a measured experiment or user specimen."""
    image = np.full((220, 320, 3), 240, dtype=np.uint8)
    cv2.ellipse(image, (160, 110), (100, 34), -15, 0, 360, (225, 190, 35), -1)
    cv2.circle(image, (180, 105), 6, (70, 45, 20), -1)
    return image


class PreservationTests(unittest.TestCase):
    def test_original_functions_and_constants_are_unchanged(self):
        baseline = json.loads((ROOT / "tests/fixtures/original_contract.json").read_text())
        source_text = (ROOT / "app.py").read_text(encoding="utf-8")
        source = ast.parse(source_text)
        functions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
        for name, expected in baseline["functions"].items():
            actual = hashlib.sha256(ast.get_source_segment(source_text, functions[name]).encode()).hexdigest()
            self.assertEqual(actual, expected, name)
        values = {t.id: n.value for n in source.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name)}
        for name, expected in baseline["constants"].items():
            actual = hashlib.sha256(ast.get_source_segment(source_text, values[name]).encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_model_bytes_and_feature_order(self):
        baseline = json.loads((ROOT / "tests/fixtures/original_contract.json").read_text())
        self.assertEqual(hashlib.sha256(app.MODEL_PATH.read_bytes()).hexdigest(), baseline["model_sha256"])
        self.assertEqual(app.MODEL_LOAD_ERROR, "")
        self.assertEqual(app.FEATURE_COLS, app.EXPECTED_FEATURE_COLS)
        self.assertEqual(list(app.MODEL.feature_names_in_), app.FEATURE_COLS)
        self.assertEqual(list(app.MODEL.classes_), ["A", "B", "C"])
        self.assertEqual(len(app.FEATURE_COLS), 21)

    def test_predictions_against_pre_extension_snapshot(self):
        rows = json.loads((ROOT / "tests/fixtures/original_predictions.json").read_text())
        for row in rows:
            vector = pd.DataFrame([row["features"]], columns=app.FEATURE_COLS)
            grade, probabilities = app.predict_grade(vector)
            self.assertEqual(grade, row["grade"])
            np.testing.assert_allclose(probabilities["Probability"], row["probabilities"], rtol=0, atol=1e-14)


class SpectralDataTests(unittest.TestCase):
    def setUp(self):
        self.library = ftir.load_reference_library()

    def test_ten_authentic_files_complete_mapping_and_units(self):
        self.assertTrue(self.library.complete, self.library.errors)
        self.assertEqual(len(self.library.spectra), 10)
        self.assertEqual(len({s.banana_id for s in self.library.spectra}), 5)
        self.assertEqual(sum(s.ripeness == "Raw" for s in self.library.spectra), 4)
        for s in self.library.spectra:
            self.assertEqual((s.banana_id, s.ripeness, s.tissue, s.weight_g), ftir.EXPERIMENT[s.crf_id])
            np.testing.assert_array_equal(s.wavenumber, ftir.EXPECTED_GRID)
            self.assertEqual(len(s.absorbance), 3601)
            self.assertTrue(np.isfinite(s.absorbance).all())
            self.assertFalse(s.absorbance.flags.writeable)
            self.assertEqual(hashlib.sha256((ftir.DATA_DIR / s.filename).read_bytes()).hexdigest(), s.sha256)
            self.assertEqual({t.tissue for t in self.library.spectra if t.banana_id == s.banana_id}, {"Peel", "Pulp"})

    def test_conversion_is_logarithmic_and_does_not_clip(self):
        np.testing.assert_allclose(ftir.transmittance_to_absorbance([100, 10, 50]), [0, 1, np.log10(2)])
        self.assertLess(ftir.transmittance_to_absorbance([100.01])[0], 0)
        for values in ([0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                ftir.transmittance_to_absorbance(values)

    def test_crf166_single_negative_absorbance_retained(self):
        s = next(s for s in self.library.spectra if s.crf_id == "CRF166")
        above = s.transmittance > 100
        self.assertEqual(np.count_nonzero(above), 1)
        self.assertEqual(s.wavenumber[above][0], 3736)
        self.assertEqual(s.transmittance[above][0], 100.01)
        self.assertLess(s.absorbance[above][0], 0)
        self.assertTrue(s.warnings)

    def test_missing_directory_has_no_fabricated_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            library = ftir.load_reference_library(directory)
            result = advanced.build_analysis(feature_fixture(), "C", library=library)
            self.assertIn("reference data not loaded", result[1])
            self.assertEqual(result[2:6], (None, None, None, None))
            self.assertIn("80.00%", result[0])

    def test_missing_spectrum_keeps_partial_data_but_disables_pca(self):
        with tempfile.TemporaryDirectory() as directory:
            for source in ftir.DATA_DIR.glob("*.csv"):
                if source.name != "CRF 173.csv":
                    shutil.copyfile(source, Path(directory) / source.name)
            library = ftir.load_reference_library(directory)
            self.assertEqual(len(library.spectra), 9)
            self.assertFalse(library.complete)
            self.assertTrue(any("CRF173" in message for message in library.errors))
            with self.assertRaises(ValueError):
                ftir.calculate_pca(library)

    def test_conflicting_manifest_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = (ftir.DATA_DIR / "manifest.csv").read_text()
            (Path(directory) / "manifest.csv").write_text(manifest.replace("CRF164,B1,Raw,Peel", "CRF164,B3,Ripe,Peel"))
            library = ftir.load_reference_library(directory)
            self.assertEqual(len(library.spectra), 0)
            self.assertIn("mapping conflicts", library.errors[0])

    def test_duplicate_manifest_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = (ftir.DATA_DIR / "manifest.csv").read_text().replace("CRF165,", "CRF164,")
            (Path(directory) / "manifest.csv").write_text(manifest)
            self.assertIn("exactly once", ftir.load_reference_library(directory).errors[0])

    def test_bad_units_grid_data_and_header_are_not_guessed(self):
        with (ftir.DATA_DIR / "manifest.csv").open(newline="") as handle:
            metadata = next(csv.DictReader(handle))
        source = (ftir.DATA_DIR / metadata["filename"]).read_text()
        bad_sources = [
            source.replace("cm-1,%T", "cm-1,Absorbance"),
            source.replace("3999.00,", "4000.00,", 1),
            source.replace("Sample 164", "Sample 165"),
            "\n".join(source.splitlines()[:-1]),
            source.replace("4000.00,98.61", "4000.00,nan", 1),
            source.replace("4000.00,98.61", "4000.00,0", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / metadata["filename"]
            for bad in bad_sources:
                path.write_text(bad)
                altered = {**metadata, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                with self.assertRaises(ValueError):
                    ftir.read_spectrum(path, altered)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ftir.read_spectrum(path, metadata)


class AnalysisTests(unittest.TestCase):
    def test_distinct_visual_patterns(self):
        expected = {"green": ("Raw",), "yellow": ("Ripe",), "mixed": ("Raw", "Ripe"),
                    "dark": ("Ripe",), "uncertain": ("Raw", "Ripe")}
        descriptions = set()
        research = set()
        for pattern, groups in expected.items():
            f = feature_fixture(pattern)
            selection = advanced.select_reference(f)
            self.assertEqual(selection.pattern, pattern)
            self.assertEqual(selection.groups, groups)
            descriptions.add(advanced.describe_current(f, "C", selection))
            research.add(advanced.contextual_research(pattern, f))
        self.assertEqual(len(descriptions), 5)
        self.assertEqual(len(research), 5)

    def test_quality_failure_suppresses_visual_emphasis(self):
        f = feature_fixture()
        f["image_quality_score"] = .49
        selection = advanced.select_reference(f)
        self.assertEqual(selection.groups, ("Raw", "Ripe"))
        self.assertIn("checks", selection.reason)

    def test_weight_and_grade_do_not_select_reference_specimen(self):
        f = feature_fixture()
        for weight in (68, 69, 97, 99, 119, 500):
            f["weight_g"] = weight
            self.assertEqual(advanced.select_reference(f).groups, ("Raw",))

    def test_grade_c_is_not_equated_with_unripe(self):
        f = feature_fixture("dark")
        selection = advanced.select_reference(f)
        text = advanced.describe_current(f, "C", selection)
        self.assertIn("either unripe or overripe", text)
        self.assertIn("no separately validated overripe", selection.reason)

    def test_mixed_references_have_no_invented_intermediate_spectrum(self):
        self.assertIn("no intermediate-stage spectra", advanced.select_reference(feature_fixture("mixed")).reason)

    def test_sensor_provenance_and_values_are_explicit(self):
        f = feature_fixture("yellow")
        f["surface_temp_C"] = 30.5
        text = advanced.describe_current(f, "A", advanced.select_reference(f))
        self.assertIn("30.50 °C", text)
        self.assertIn("+4.50 °C", text)
        self.assertIn("manually entered", text)
        self.assertIn("gas resistance is not a chemical concentration", text)

    def test_invalid_features_fail_explicitly(self):
        for value in (float("nan"), -1, 120):
            f = feature_fixture()
            f["peel_green_pct"] = value
            with self.assertRaises(ValueError):
                advanced.select_reference(f)

    def test_research_has_sources_and_no_treatment_output(self):
        for pattern in ("green", "yellow", "mixed", "dark", "uncertain"):
            text = advanced.contextual_research(pattern, feature_fixture(pattern))
            self.assertIn("10.1007/s00340-023-08023-w", text)
            self.assertIn("unverified treatment histories", text)
            self.assertIn("no chemical-treatment verdict or score", text)
            self.assertIn("preprint", text)


class PlotAndPCATests(unittest.TestCase):
    def setUp(self):
        self.library = ftir.load_reference_library()

    def test_plotted_curves_equal_original_converted_values(self):
        figure = ftir.plot_ripeness_overlay(self.library, ("Raw",))
        for axis, tissue in zip(figure.axes, ("Peel", "Pulp")):
            spectra = [s for s in self.library.spectra if s.tissue == tissue]
            for line, s in zip(axis.lines, spectra):
                np.testing.assert_array_equal(line.get_xdata(), s.wavenumber)
                np.testing.assert_array_equal(line.get_ydata(), -np.log10(s.transmittance / 100))
                self.assertIn(s.crf_id, line.get_label())
                self.assertEqual(line.get_alpha(), .95 if s.ripeness == "Raw" else .35)

    def test_reference_emphasis_changes_style_not_spectra(self):
        raw = ftir.plot_ripeness_overlay(self.library, ("Raw",))
        ripe = ftir.plot_ripeness_overlay(self.library, ("Ripe",))
        for a, b in zip(raw.axes, ripe.axes):
            for first, second in zip(a.lines, b.lines):
                np.testing.assert_array_equal(first.get_ydata(), second.get_ydata())
                self.assertNotEqual(first.get_alpha(), second.get_alpha())

    def test_fingerprint_grid_and_paired_filter(self):
        figure = ftir.plot_ripeness_overlay(self.library, fingerprint=True)
        for axis in figure.axes:
            for line in axis.lines:
                np.testing.assert_array_equal(line.get_xdata(), np.arange(1200, 899, -1))
        paired = ftir.plot_tissue_comparison(self.library, specimen="B3")
        self.assertEqual(len(paired.axes[0].lines), 2)
        self.assertIn("CRF166", paired.axes[0].lines[0].get_label())
        self.assertIn("CRF167", paired.axes[0].lines[1].get_label())

    def test_pca_reproducible_and_derived_from_absorbance(self):
        scores, ratios, loadings = ftir.calculate_pca(self.library)
        again = ftir.calculate_pca(self.library)
        for first, second in zip((scores, ratios, loadings), again):
            np.testing.assert_array_equal(first, second)
        matrix = np.stack([s.absorbance for s in self.library.spectra])
        matrix -= matrix.mean(axis=0)
        np.testing.assert_allclose(scores, matrix @ loadings.T, atol=1e-12)
        np.testing.assert_allclose(loadings @ loadings.T, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(ratios, np.sum(scores**2, axis=0) / np.sum(matrix**2), atol=1e-12)
        for row in loadings:
            self.assertGreater(row[np.argmax(np.abs(row))], 0)
        self.assertEqual(scores.shape, (10, 2))
        self.assertIn("not been reproduced", ftir.pca_description(ratios))

    def test_plot_failure_is_local(self):
        with patch.object(ftir, "plot_tissue_comparison", side_effect=RuntimeError("test plotting failure")):
            outputs = advanced.build_analysis(feature_fixture(), "C", library=self.library)
        self.assertIsNotNone(outputs[2])
        self.assertIsNone(outputs[3])
        self.assertIsNotNone(outputs[4])
        self.assertIsNotNone(outputs[5])
        self.assertIn("test plotting failure", outputs[1])


class CallbackTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def args(self):
        return (image_fixture(), 120, 27, 26, 55, 1013.25, 80)

    def test_original_15_outputs_preserved_and_one_extraction(self):
        original = app.run_prediction(*self.args())
        with patch.object(app, "extract_image_features", wraps=app.extract_image_features) as extract, \
             patch.object(app, "predict_grade", wraps=app.predict_grade) as predict, \
             patch.object(app, "read_stm32_measurement") as serial_read:
            extended = app.run_prediction_with_analysis(*self.args())
        self.assertEqual(len(original), 15)
        self.assertEqual(len(extended), 25)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(predict.call_count, 1)
        serial_read.assert_not_called()
        for index in (0, 1, 11, 12):
            self.assertEqual(original[index], extended[index])
        for index in (2, 8, 9, 10):
            pd.testing.assert_frame_equal(original[index], extended[index], check_exact=True)
        for index in (4, 5, 6, 7):
            np.testing.assert_array_equal(original[index], extended[index])
        self.assertEqual(extended[24]["features"], original[10].iloc[0].to_dict())

    def test_reference_loader_failure_does_not_change_grade(self):
        original = app.run_prediction(*self.args())
        with patch.object(ftir, "load_reference_library", side_effect=RuntimeError("test library unavailable")):
            result = app.run_prediction_with_analysis(*self.args())
        self.assertEqual(original[0], result[0])
        pd.testing.assert_frame_equal(original[2], result[2])
        self.assertIn("test library unavailable", result[15])
        self.assertIsNone(result[24])

    def test_invalid_image_clears_stale_research_outputs(self):
        result = app.run_prediction_with_analysis(None, 120, 27, 26, 55, 1013.25, 80)
        self.assertIn("Prediction unavailable", result[0])
        self.assertEqual(result[17:21], (None, None, None, None))
        self.assertIsNone(result[24])

    def test_comparison_control_uses_state_without_prediction(self):
        state = {"features": feature_fixture("green"), "grade": "C"}
        with patch.object(app, "run_prediction") as predict:
            result = app.update_reference_comparison(state, "Compare all equally", "B3")
        predict.assert_not_called()
        self.assertEqual(result[9], state)
        self.assertIn("Graph emphasis:** Raw + Ripe", result[1])
        self.assertEqual(len(result[3].axes[0].lines), 2)

    def test_ui_event_contracts_and_hardware_bindings(self):
        dependencies = {d["api_name"]: d for d in app.demo.config["dependencies"]}
        self.assertEqual(len(dependencies["run_prediction_with_analysis"]["outputs"]), 25)
        self.assertEqual(len(dependencies["capture_from_esp32"]["outputs"]), 2)
        self.assertEqual(len(dependencies["read_stm32_measurement"]["outputs"]), 3)
        self.assertEqual(len(dependencies["update_sensor_values"]["outputs"]), 3)
        self.assertIn("update_reference_comparison", dependencies)

    def test_stm32_buffered_parser_with_fake_serial(self):
        class FakeSerial:
            is_open = True
            payload = b"FINAL -> Weight: 99.25 g | Temp: 28.50 C\n"
            @property
            def in_waiting(self):
                return len(self.payload)
            def read(self, count):
                result, self.payload = self.payload[:count], self.payload[count:]
                return result
        with patch.object(app, "ser", FakeSerial()), patch.object(app, "serial_line_buffer", ""):
            weight, temperature, status = app.read_stm32_measurement()
        self.assertEqual(weight["value"], 99.25)
        self.assertEqual(temperature["value"], 28.5)
        self.assertIn("99.25", status)

    def test_esp32_capture_and_flash_with_mock_transport(self):
        class Response:
            status = 200
            def __init__(self, payload=b"", content_type="image/jpeg"):
                self.payload = payload
                self.headers = type("Headers", (), {"get_content_type": lambda _: content_type})()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return self.payload
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_fixture(), cv2.COLOR_RGB2BGR))
        with patch.object(app, "urlopen", side_effect=[Response(), Response(encoded.tobytes()), Response()]) as request:
            image, status = app.capture_from_esp32("192.0.2.1")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(image.shape, image_fixture().shape)
        self.assertIn("flash enabled", status)
        self.assertTrue(request.call_args_list[-1].args[0].full_url.endswith("led_intensity&val=0"))


if __name__ == "__main__":
    unittest.main()
