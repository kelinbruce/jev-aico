import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from nimble.datasets.curation_serialization import wrap_paragraph_for_object


class ContainerRecoveryTests(unittest.TestCase):
    def test_preserves_every_character_and_shifts_only_verified_root_paths(self):
        paragraph = '  Record A names Mina.\nRecord A reports “blue”.  '
        doc = {'base_state_json': json.dumps(paragraph), 'focus_evidence': [
            {'path': [], 'text': 'Record A names Mina.'},
            {'path': [], 'text': 'Record A reports “blue”.'}]}
        before = copy.deepcopy(doc)
        request = {'payload_json': json.dumps({'original_input': {'state': {'old': 'context'}}})}
        result = wrap_paragraph_for_object(doc, request)
        self.assertEqual(json.loads(result['base_state_json']), {'context': paragraph})
        self.assertEqual([s['path'] for s in result['focus_evidence']], [['context'], ['context']])
        self.assertEqual([s['text'] for s in result['focus_evidence']], [s['text'] for s in doc['focus_evidence']])
        self.assertEqual(doc, before)

    def test_refuses_ambiguous_or_unrelated_repairs(self):
        request = {'payload_json': json.dumps({'original_input': {'state': {}}})}
        doc = {'base_state_json': json.dumps('One sentence. Another sentence.'), 'focus_evidence': [
            {'path': [], 'text': 'One sentence.'}, {'path': [], 'text': 'Another sentence.'}]}
        for raw in ('not encoded JSON', '{broken', '{}', '[]', 'null', '42', '""',
                    json.dumps('One sentence. One sentence. Another sentence.')):
            changed = {**doc, 'base_state_json': raw}
            self.assertEqual(wrap_paragraph_for_object(changed, request), changed)
        changed = copy.deepcopy(doc)
        changed['focus_evidence'][0]['path'] = ['missing']
        self.assertEqual(wrap_paragraph_for_object(changed, request), changed)
        for original in ('paragraph', [], None):
            req = {'payload_json': json.dumps({'original_input': {'state': original}})}
            self.assertEqual(wrap_paragraph_for_object(doc, req), doc)


class RecoveryActivationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_documents_unchanged_until_recorded_threshold(self):
        from nimble.datasets.fast_training_dataset import AsyncStages, parser
        from nimble.datasets.curate_paired_evidence import Document, DocumentGenerator
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.json'
            path.write_text(json.dumps({'version': 1, 'bucket_start_ordinals': {},
                'source_guidance_start_variations': {}, 'object_wrapper_start_variations': {'source-a': 20}}))
            args = parser().parse_args(['--resume-policy', str(path)])
            stage = AsyncStages(Path(tmp), args)
            doc = {'base_state_json': json.dumps('First sentence. Second sentence.'), 'focus_evidence': [
                {'path': [], 'text': 'First sentence.'}, {'path': [], 'text': 'Second sentence.'}]}
            stage.one = AsyncMock(return_value=doc)
            for ordinal in (19, 20):
                req = {'job_id': f'fast-{args.seed}-source-a-{ordinal:03d}', 'payload_json': json.dumps({
                    'original_input': {'state': {}}, 'variation': {'ordinal': ordinal}})}
                result = (await stage.run(DocumentGenerator, Document, [req]))[req['job_id']]
                self.assertEqual(result, doc if ordinal == 19 else wrap_paragraph_for_object(doc, req))
