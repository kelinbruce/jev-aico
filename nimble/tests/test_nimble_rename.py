"""Renaming preserves frozen artifact inputs and published inference packages."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nimble
from nimble.compat import evaluation_settings, model_key, published_model_class, response_path
from nimble.datasets.replay_curation_snapshot import relocated_source


class NimbleRenameTests(unittest.TestCase):
    def test_public_name(self):
        self.assertEqual(nimble.MODEL_NAME, "Bespoke-Nimble-9B")

    def test_old_imports_move_but_prompt_text_stays_identical(self):
        for package in ("openjeff", "openjevons"):
            source = f'from {package}.paths import PROJECT_ROOT\nPROMPT = "{package}"\n'
            self.assertEqual(relocated_source(source),
                             f'from nimble.paths import PROJECT_ROOT\nPROMPT = "{package}"\n')

    def test_metadata_translation_preserves_fingerprints_and_original(self):
        saved = {"openjeff_model": "bespoke-openjeff-9b", "dataset_sha256": "unchanged"}
        current = evaluation_settings(saved)
        self.assertEqual(current, {"nimble_model": nimble.MODEL_NAME, "dataset_sha256": "unchanged"})
        self.assertIn("openjeff_model", saved)
        self.assertEqual(evaluation_settings(current), current)
        with self.assertRaises(ValueError):
            evaluation_settings({**saved, "nimble_model": "conflict"})

    def test_frozen_response_file_is_read_without_renaming(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "openjeff_responses.jsonl"
            legacy.write_bytes(b'original bytes\n')
            self.assertEqual(response_path(directory), legacy)
            self.assertEqual(legacy.read_bytes(), b'original bytes\n')
            current = Path(directory) / "nimble_responses.jsonl"
            current.touch()
            self.assertEqual(response_path(directory), current)

    def test_frozen_and_new_inference_classes_are_supported(self):
        old, new = object(), object()
        self.assertIs(published_model_class(SimpleNamespace(OpenJeffModel=old)), old)
        self.assertIs(published_model_class(SimpleNamespace(NimbleModel=new, OpenJeffModel=old)), new)
        with self.assertRaises(ImportError):
            published_model_class(SimpleNamespace())
        self.assertEqual(model_key("openjeff"), "nimble")
        self.assertEqual(model_key("jev"), "jev")


if __name__ == "__main__":
    unittest.main()
