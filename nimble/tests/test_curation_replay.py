"""Archived releases survive import moves without bypassing gate integrity."""
import tempfile
import unittest
from pathlib import Path

from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.replay_curation_snapshot import relocated_source, same_lineage, verify_snapshot


class ReplayIntegrityTests(unittest.TestCase):
    def test_import_move_is_allowed_but_gate_edits_are_rejected(self):
        recorded = 'from openjevons.datasets.create_tiny_dataset import canonical\nLIMIT = 8\n'
        files = {'minicheck_data.py': recorded}
        digest = fingerprint(files)
        snapshot = {'files': files, 'sha256': digest}
        config = {'implementation_sha256': digest}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            current = directory / 'contrastive_data.py'
            current.write_text(relocated_source(recorded))
            verify_snapshot(snapshot, config, directory)
            current.write_text(current.read_text().replace('LIMIT = 8', 'LIMIT = 80'))
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                verify_snapshot(snapshot, config, directory)

    def test_relocation_does_not_rewrite_prompt_text(self):
        source = 'from openjevons.paths import PROJECT_ROOT\nPROMPT = "openjevons is mentioned here"\n'
        self.assertEqual(relocated_source(source),
                         'from nimble.paths import PROJECT_ROOT\nPROMPT = "openjevons is mentioned here"\n')

    def test_lineage_accepts_directory_alias_but_not_different_plans(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'current'
            directory.mkdir()
            alias = Path(temp) / 'previous'
            alias.symlink_to(directory, target_is_directory=True)
            current = {'directory': str(directory), 'plans_sha256': 'original'}
            recorded = {**current, 'directory': str(alias)}
            self.assertTrue(same_lineage(current, recorded))
            self.assertFalse(same_lineage(current, {**recorded, 'plans_sha256': 'changed'}))

    def test_modified_archive_is_rejected(self):
        files = {'fast_training_dataset.py': 'LIMIT = 8\n'}
        digest = fingerprint(files)
        snapshot = {'files': {**files, 'fast_training_dataset.py': 'LIMIT = 80\n'}, 'sha256': digest}
        with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
            verify_snapshot(snapshot, {'implementation_sha256': digest}, Path('.'))
