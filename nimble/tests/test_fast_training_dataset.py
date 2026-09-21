import asyncio
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests import test_evidence_training as fixtures
from nimble.datasets.dataset_io import canonical, write_jsonl
from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.scaled_evidence import build_group
from nimble.datasets.curate_paired_evidence import Pair, PairGenerator
from nimble.datasets.fast_training_dataset import (
    AsyncStages, DraftRejected, RequestBudget, changed_words, curate, load_training_plans, parser, curation_executor,
    normalize_document_serialization,
    select_source,
)
from nimble.datasets.curation_profiles import EVALUATION_MODEL, TRAINING_MODEL
from nimble.training.evidence_data import validate_evidence_data


class FakeClient:
    def __init__(self, source, rows, gate=None):
        self.cert = rows[0]['evidence_certificate']
        self.group = build_group('g', source, self.cert['spec'], self.cert['spec'], self.cert['verified_pair'])
        self.chat = SimpleNamespace(completions=self)
        self.active = self.maximum = 0
        self.gate = gate
        self.fast_advanced = False

    async def create(self, **spec):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            prompt = spec['messages'][0]['content']
            name = spec['response_format']['json_schema']['name']
            if name == 'Pair':
                if self.gate and 'slow-domain' in prompt:
                    await asyncio.wait_for(self.gate.wait(), 3)
                result = self.cert['verified_pair']
            elif name == 'Document':
                if self.gate and 'fast-domain' in prompt:
                    self.fast_advanced = True
                    self.gate.set()
                result = {k: self.cert['spec'][k] for k in ('base_state_json', 'focus_evidence')}
            elif name == 'ContextAudit':
                result = {**self.cert['context_audit'], 'explanation': 'Controlled fixture.'}
            elif name == 'FactAudit':
                payload = json.JSONDecoder().raw_decode(prompt[prompt.index('\n{') + 1:])[0]
                pair = self.cert['verified_pair']
                context = payload['context']
                states = {'left': pair['left'], 'right': pair['right'],
                          'negative_sentence': pair['negative_right'],
                          'positive_pair': pair['left'] + '\n' + pair['right'],
                          'negative_pair': pair['negative_left'] + '\n' + pair['negative_right']}
                matrices = {**self.cert['pair_fact_states'], **self.cert['full_context_fact_states']}
                states.update(self.group['states'])
                case = next(k for k, v in states.items() if v == context)
                values = matrices[case]
                quote = context if isinstance(context, str) else context[0]
                result = {'judgments': [{'atom_id': a['id'], 'state': values[a['id']],
                    'quotes': [] if values[a['id']] == 'unknown' else [quote],
                    'reason': 'Controlled fixture.'} for a in payload['propositions']]}
            else:
                raise AssertionError(name)
            await asyncio.sleep(.001)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                refusal=None, content=json.dumps(result)))], model=spec['model'], usage=None)
        finally:
            self.active -= 1


class FastTrainingTests(unittest.IsolatedAsyncioTestCase):
    def test_rate_windows_use_post_guidance_results_only_after_checkpoint(self):
        pool = ['previously_productive', 'newly_guided']
        sources = {s: {'family': s} for s in pool}
        tried = Counter(previously_productive=160, newly_guided=50)
        accepted = Counter(previously_productive=8)
        old = {'bucket_start_ordinals': {'commerce/score': 1}}
        policy = {**old, 'acceptance_rate_windows': {'start_ordinals': {'commerce/score': 211},
            'sources': {'newly_guided': {'attempted_before': 50, 'accepted_before': 0},
                        'previously_productive': {'attempted_before': 130, 'accepted_before': 8}}}}
        choose = lambda n, p: select_source(pool, ('commerce', 'score'), n, tried, accepted, Counter(), sources, p)
        self.assertEqual(choose(208, policy), choose(208, old))
        self.assertEqual(choose(211, old), 'previously_productive')
        self.assertEqual(choose(211, policy), 'newly_guided')

    def test_exemplars_only_affect_future_drafts_not_blind_verifiers(self):
        from nimble.datasets.curate_paired_evidence import Document, DocumentGenerator
        from nimble.datasets.evidence_stages import FactAudit, FactReviewer
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args([])
            baseline = AsyncStages(Path(tmp), args)
            path = Path(tmp) / 'policy.json'
            path.write_text(json.dumps({'version': 1, 'bucket_start_ordinals': {},
                'source_guidance_start_variations': {}, 'construction_examples': {
                    'source-a': {'start_variation': 10, 'verified_pair': {'left': 'EXAMPLE_PAIR'},
                                 'base_context': 'EXAMPLE_CONTEXT'}}}))
            revised_args = copy.copy(args)
            revised_args.resume_policy = path
            revised = AsyncStages(Path(tmp), revised_args)
            for cls, schema in ((PairGenerator, Pair), (DocumentGenerator, Document), (FactReviewer, FactAudit)):
                for sid, ordinal in (('source-a', 9), ('source-a', 10), ('other-source', 10)):
                    row = {'job_id': f'fast-{args.seed}-{sid}-{ordinal:03d}',
                           'payload_json': canonical({'variation': {'ordinal': ordinal}})}
                    old = baseline.request_spec(cls, schema, row)
                    new = revised.request_spec(cls, schema, row)
                    if cls is FactReviewer or ordinal < 10 or sid != 'source-a':
                        self.assertEqual(old, new)
                    else:
                        content = new['messages'][0]['content']
                        self.assertIn('EXAMPLE_PAIR', content)
                        self.assertEqual('EXAMPLE_CONTEXT' in content, cls is DocumentGenerator)

    def test_adaptive_selection_preserves_checkpoint_and_explores(self):
        pool = ['productive', 'unproductive']
        sources = {s: {'family': s} for s in pool}
        tried = Counter(productive=20, unproductive=19)
        accepted = Counter(productive=5)
        families = Counter()
        policy = {'bucket_start_ordinals': {'media/score': 40}}
        choose = lambda n, p: select_source(pool, ('media', 'score'), n, tried, accepted, families, sources, p)
        self.assertEqual(choose(39, policy), choose(39, {}))
        self.assertEqual(choose(40, policy), 'productive')
        self.assertEqual(choose(43, policy), 'productive')
        self.assertEqual(choose(44, policy), 'unproductive')

    def test_per_source_guidance_preserves_earlier_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(['--guidance-from-variation', '40'])
            baseline = AsyncStages(Path(tmp), args)
            path = Path(tmp) / 'policy.json'
            path.write_text(json.dumps({'version': 1, 'bucket_start_ordinals': {},
                'source_guidance_start_variations': {'source-a': 25}}))
            revised_args = copy.copy(args)
            revised_args.resume_policy = path
            revised = AsyncStages(Path(tmp), revised_args)
            for ordinal in (1, 24, 25, 39, 40, 41):
                row = {'job_id': f'fast-{args.seed}-source-a-{ordinal:03d}',
                       'payload_json': canonical({'variation': {'ordinal': ordinal}})}
                old = baseline.request_spec(PairGenerator, Pair, row)
                new = revised.request_spec(PairGenerator, Pair, row)
                if ordinal < 25 or ordinal >= 40:
                    self.assertEqual(old, new)
                else:
                    self.assertNotEqual(old, new)

    def test_future_guidance_preserves_prior_requests_and_all_verification(self):
        from nimble.datasets.curate_paired_evidence import Document, DocumentGenerator
        from nimble.datasets.evidence_stages import FactAudit, FactReviewer
        args = parser().parse_args([])
        baseline = AsyncStages(Path('/unused'), args)
        revised_args = copy.copy(args)
        revised_args.guidance_from_variation = 40
        revised = AsyncStages(Path('/unused'), revised_args)
        for cls, schema in ((PairGenerator, Pair), (DocumentGenerator, Document),
                            (FactReviewer, FactAudit)):
            for ordinal in (1, 39, 40, 41):
                row = {'job_id': 'sample', 'payload_json': canonical({'variation': {'ordinal': ordinal}})}
                old, new = baseline.request_spec(cls, schema, row), revised.request_spec(cls, schema, row)
                if ordinal < 40 or cls is FactReviewer:
                    self.assertEqual(old, new)
                else:
                    self.assertNotEqual(fingerprint(old), fingerprint(new))
                    self.assertTrue(new['messages'][0]['content'].startswith(old['messages'][0]['content']))
                    self.assertEqual(old['response_format'], new['response_format'])

    def test_literal_string_encoding_preserves_text_and_refuses_ambiguous_repairs(self):
        raw='  Record A identifies Mina.\nRecord A says “blue”.  '
        doc={'base_state_json':raw,'focus_evidence':[
            {'path':[],'text':'Record A identifies Mina.'},
            {'path':[],'text':'Record A says “blue”.'}]}
        request={'payload_json':json.dumps({'original_input':{'state':'Old paragraph.'}})}
        result=normalize_document_serialization(doc,request)
        self.assertEqual(json.loads(result['base_state_json']),raw)
        self.assertEqual(result['focus_evidence'],doc['focus_evidence'])
        self.assertEqual(doc['base_state_json'],raw) # original API response remains raw
        self.assertEqual(normalize_document_serialization(result,request),result)
        for prefix in ('{','[','"','```'):
            damaged={**doc,'base_state_json':prefix+raw}
            self.assertEqual(normalize_document_serialization(damaged,request),damaged)
        nonroot=copy.deepcopy(doc);nonroot['focus_evidence'][0]['path']=['text']
        self.assertEqual(normalize_document_serialization(nonroot,request),nonroot)
        repeated={**doc,'base_state_json':raw+raw}
        self.assertEqual(normalize_document_serialization(repeated,request),repeated)
        for state in ({'text':'old'}, ['old']):
            self.assertEqual(normalize_document_serialization(doc,
                {'payload_json':json.dumps({'original_input':{'state':state}})}),doc)

    async def test_refused_candidate_is_skipped_and_replay_preserves_acceptance(self):
        class RejectOnceClient(FakeClient):
            first = True
            async def create(self, **spec):
                if self.first:
                    self.first = False
                    return SimpleNamespace(model=spec['model'],usage=None,
                        choices=[SimpleNamespace(finish_reason='refusal',
                            message=SimpleNamespace(refusal='refused',content=''))])
                return await super().create(**spec)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args,source,rows,plans,config=self.setup_data(root)
            stage=AsyncStages(root,args,RejectOnceClient(source,rows))
            result=await curate(args,[source],plans,config,stage)
            self.assertEqual(result['training']['examples'],2)
            self.assertEqual(result['attempted_groups'],2)
            rejected=[json.loads(s) for s in (root/'review_queue.jsonl').read_text().splitlines()]
            self.assertEqual(rejected[0]['stage'],'model_response')
            args.offline=True
            self.assertEqual(await curate(args,[source],plans,config,AsyncStages(root,args)),result)

    async def test_bad_responses_are_cached_rejections_not_retried_labels(self):
        for finish, refusal, content in [('length',None,'{}'),('stop','refused','{}'),('stop',None,'invalid JSON')]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, _, _, _, _ = self.setup_data(root)
                create = AsyncMock(return_value=SimpleNamespace(model=args.generator_model,usage=None,
                    choices=[SimpleNamespace(finish_reason=finish,
                        message=SimpleNamespace(refusal=refusal,content=content))]))
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                stage = AsyncStages(root,args,client)
                request={'job_id':'x','payload_json':'{}'}
                for _ in range(2):
                    with self.assertRaises(DraftRejected) as error:
                        await stage.one(PairGenerator,Pair,request)
                    self.assertEqual(error.exception.stage,'model_response')
                create.assert_awaited_once()
                self.assertEqual(stage.api_calls,1)
                self.assertFalse(list((root/'requests').glob('*.json')))
                args.offline=True
                with self.assertRaises(DraftRejected):
                    await AsyncStages(root,args).one(PairGenerator,Pair,request)
                path=next((root/'rejected_requests').glob('*.json'))
                saved=json.loads(path.read_text());saved['rejection']['reason']='tampered'
                path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    await AsyncStages(root,args).one(PairGenerator,Pair,request)

    async def test_full_worker_pool_leaves_room_for_network_resolution(self):
        class DNSClient(FakeClient):
            async def create(self, **spec):
                # asyncio getaddrinfo uses the same executor as to_thread. This
                # used to stall when every worker was occupied by a group bridge.
                await asyncio.wait_for(asyncio.to_thread(lambda: 'resolved'), 2)
                return await super().create(**spec)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, source, rows, plans, config = self.setup_data(root)
            args.group_concurrency = 1
            asyncio.get_running_loop().set_default_executor(curation_executor(1))
            result = await asyncio.wait_for(curate(args,[source],plans,config,
                AsyncStages(root,args,DNSClient(source,rows))), 5)
            self.assertEqual(result['training']['examples'],2)

    def setup_data(self, root, target=2):
        args = parser().parse_args(['--output', str(root), '--target', str(target), '--concurrency', '3'])
        source, rows = fixtures.EvidenceTrainingTests().examples()
        cert = rows[0]['evidence_certificate']
        plan = {k:v for k,v in cert['spec'].items() if k not in ('base_state_json','focus_evidence')}
        plans = {source['id']: {'source_id': source['id'], 'plan': plan, 'audit': cert['rule_audit']}}
        path = root/'source.jsonl'
        write_jsonl(path, [source])
        config = {'source': str(path), 'source_sha256': fingerprint([source])}
        return args, source, rows, plans, config

    async def test_full_pipeline_offline_replay_and_loader_certificates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args, source, rows, plans, config = self.setup_data(root)
            client = FakeClient(source, rows)
            stage = AsyncStages(root, args, client)
            manifest = await curate(args, [source], plans, config, stage)
            exported = [json.loads(s) for s in (root/'train.jsonl').read_text().splitlines()]
            self.assertEqual(manifest['training']['examples'], 2)
            self.assertEqual(len(validate_evidence_data(exported, manifest)), 1)
            self.assertGreater(client.maximum, 1)
            self.assertLessEqual(client.maximum, 3)
            args.offline = True
            replay = AsyncStages(root, args)
            self.assertEqual(await curate(args, [source], plans, config, replay), manifest)
            self.assertEqual(replay.api_calls, 0)
            self.assertGreater(replay.cache_hits, 0)

    async def test_fast_group_advances_while_other_group_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args, source, rows, plans, config = self.setup_data(root, 4)
            sources=[]
            revised={}
            for domain in ('fast-domain','slow-domain'):
                s=copy.deepcopy(source)
                s.update(id=domain, domain=domain, family=domain)
                s['input']['questions']['decision']['instructions'] += ' '+domain
                sources.append(s)
                revised[s['id']]={**plans[source['id']], 'source_id': s['id']}
            client=FakeClient(source, rows, asyncio.Event())
            result=await asyncio.wait_for(curate(args, sources, revised, config, AsyncStages(root,args,client)), 5)
            self.assertTrue(client.fast_advanced)
            self.assertEqual(result['training']['examples'],4)

    async def test_cache_is_bound_to_model_reasoning_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args, source, rows, _, _=self.setup_data(root)
            row={'job_id':'x','payload_json':'{}'}
            stage=AsyncStages(root,args,FakeClient(source,rows))
            await stage.one(PairGenerator,Pair,row)
            args.offline=True
            args.generator_model='different-model'
            with self.assertRaisesRegex(ValueError,'missing a response'):
                await AsyncStages(root,args).one(PairGenerator,Pair,row)
            args.generator_model=TRAINING_MODEL
            path=next((root/'requests').glob('*.json'))
            saved=json.loads(path.read_text());saved['response']['right']='Tampered.'
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                await AsyncStages(root,args).one(PairGenerator,Pair,row)

    async def test_oversized_request_fails_before_network(self):
        with self.assertRaisesRegex(ValueError,'single request'):
            await RequestBudget(10,100).reserve(101)

    def test_small_edits_and_distinct_model_defaults(self):
        pair={'left':'A is 15.', 'right':'Limit is 12.', 'negative_left':'A is 9.', 'negative_right':'Limit is 12.'}
        self.assertEqual(changed_words(pair),1)
        pair['negative_right']='Limit is 20.'
        with self.assertRaises(DraftRejected):changed_words(pair)
        self.assertNotEqual(TRAINING_MODEL,EVALUATION_MODEL)

    def test_reused_plans_reject_heldout_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _, source, _, plans, _=self.setup_data(root)
            source['split']='eval'
            (root/'manifest.json').write_text(json.dumps({'source_sha256':fingerprint([source]),'model':EVALUATION_MODEL}))
            write_jsonl(root/'accepted_plans.jsonl',list(plans.values()))
            with self.assertRaisesRegex(ValueError,'Held-out'):
                load_training_plans(root,[source])


if __name__ == '__main__':
    unittest.main()
