"""Generate and audit evidence-derived schema contrasts from training-only seeds."""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import configure, read_jsonl
from nimble.datasets.dataset_io import canonical, write_jsonl
from nimble.datasets.curation_options import llm_options
from nimble.datasets.contrastive_data import fingerprint, scoring_record
from nimble.datasets.evidence_curation import (
    VERSION, audit_training, build_group, evaluate_group, fact_jobs, rule_job, select_sources,
)


def run_stage(cls, schema, requests, output, model, stage):
    path = output / (stage + '.jsonl')
    saved = {r['job_id']: r for r in read_jsonl(path)} if path.exists() else {}
    for req in requests:
        if req['job_id'] in saved and saved[req['job_id']]['request_sha256'] != fingerprint(req):
            raise ValueError('Saved stage input changed: ' + req['job_id'])
    pending = [r for r in requests if r['job_id'] not in saved]
    if pending:
        print(f'{stage}: {len(pending)} requests ({len(saved)} cached).', flush=True)
        runner = cls(**llm_options(model, schema))
        result = runner(pending, working_dir=str(PROJECT_ROOT / '.cache/curator-evidence-v2' / stage))
        by_id = {r['job_id']: r for r in pending}
        for row in result.dataset:
            saved[row['job_id']] = {'job_id': row['job_id'],
                                    'request_sha256': fingerprint(by_id[row['job_id']]),
                                    'response': json.loads(row['result_json'])}
        write_jsonl(path, sorted(saved.values(), key=lambda r: r['job_id']))
    if not path.exists():
        write_jsonl(path, [])
    if any(r['job_id'] not in saved for r in requests):
        raise ValueError('Incomplete API stage')
    return {r['job_id']: saved[r['job_id']]['response'] for r in requests}


def generate(jobs, output, model):
    from nimble.datasets.evidence_stages import EvidenceSpec, Generation
    requests = [{'job_id': j['id'], 'payload_json': canonical({'method': j['method'], 'input': j['source']['input']})}
                for j in jobs]
    drafts = run_stage(Generation, EvidenceSpec, requests, output, model, 'generation')
    built, failures, retry = {}, [], []
    by_id = {j['id']: j for j in jobs}
    for req in requests:
        try:
            built[req['job_id']] = build_group(by_id[req['job_id']], drafts[req['job_id']])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            failures.append({'group_id': req['job_id'], 'attempt': 1, 'reason': str(error)})
            retry.append({**req, 'structural_error': str(error)})
    if retry:
        repaired = run_stage(Generation, EvidenceSpec, retry, output, model, 'structural_repair')
        for req in retry:
            try:
                built[req['job_id']] = build_group(by_id[req['job_id']], repaired[req['job_id']])
            except (ValueError, KeyError, TypeError, IndexError) as error:
                failures.append({'group_id': req['job_id'], 'attempt': 2, 'reason': str(error)})
    write_jsonl(output / 'generation_failures.jsonl', failures)
    write_jsonl(output / 'constructed_groups.jsonl', sorted(built.values(), key=lambda g: g['id']))
    return built


def export(groups, rule_audits, fact_results, jobs, sources, output, config):
    rows, review = [], []
    for job in jobs:
        gid = job['id']
        if gid not in groups:
            review.append({'group_id': gid, 'reasons': ['structural_generation_failure']})
            continue
        try:
            accepted, excluded = evaluate_group(groups[gid], rule_audits[gid], fact_results)
            rows.extend(accepted)
            review.extend(excluded)
        except (ValueError, KeyError, TypeError) as error:
            review.append({'group_id': gid, 'reasons': ['invalid_audit_output'], 'detail': str(error)})
    rows.sort(key=lambda r: r['id'])
    audit = audit_training(rows, sources)
    write_jsonl(output / 'train.jsonl', rows)
    write_jsonl(output / 'train_scoring.jsonl', [scoring_record(r) for r in rows])
    write_jsonl(output / 'review_queue.jsonl', review)
    manifest = {**config, 'created_at': datetime.now(timezone.utc).isoformat(),
                'constructed_groups': len(groups), 'candidate_contexts': len(groups) * 3,
                'training': audit, 'train_sha256': fingerprint(rows),
                'review_entries': len(review),
                'review_reasons': dict(Counter(reason for r in review for reason in r['reasons'])),
                'reference_human_reviewed': False, 'teacher_calls': 0,
                'notes': ['Targets are derived by deterministic sufficient rules over independently checked facts.',
                          'Each isolated sentence, sentence pair, and full context is checked in a separate API call.',
                          'Rules and context facts are audited separately; all judgments still use the same model.',
                          'Unknown facts never become false by default. Uncovered rule assignments are excluded.',
                          'A group must pass sentence necessity checks and have two different determinate schema labels.',
                          'Source-preserving means synthetic original contexts, not human-written D2C documents.',
                          'No final confidence or calibration targets are invented.',
                          'This is a curation pilot; quality improvement needs human review and held-out evaluation.',
                          'Related examples must stay together by source_family. Original held-out inputs are never sent.',
                          'The v1 trainer assumes five variants; these variable-size v2 groups need a compatible loader.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT_ROOT / 'data/typesafe_diverse_300_gpt56/all.jsonl')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_gpt56_v2')
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    sources = read_jsonl(args.source)
    jobs = select_sources(sources, args.limit)
    from nimble.datasets import evidence_curation, evidence_stages
    implementation_hash = fingerprint({module.__name__: Path(module.__file__).read_text()
                                      for module in (evidence_curation, evidence_stages)})
    config = {'pipeline_version': VERSION, 'source': str(args.source.resolve()),
              'source_sha256': fingerprint(sources), 'model': args.model,
              'source_seeds': len(jobs), 'held_out_seeds_excluded': sum(r['split'] != 'train' for r in sources),
              'implementation_sha256': implementation_hash,
              'selected_sources': [j['source']['id'] for j in jobs]}
    if args.offline:
        saved = json.loads((args.output / 'run_config.json').read_text())
        if saved != config:
            raise ValueError('Source/configuration/implementation changed')
        groups = {r['id']: r for r in read_jsonl(args.output / 'constructed_groups.jsonl')}
        by_id = {j['id']: j for j in jobs}
        for group in groups.values():
            if build_group(by_id[group['id']], group['spec']) != group:
                raise ValueError('Constructed context differs from its source and exact edits')
        rules = {r['job_id']: r['response'] for r in read_jsonl(args.output / 'rule_audits.jsonl')}
        facts = {r['job_id']: r['response'] for r in read_jsonl(args.output / 'fact_audits.jsonl')}
        # Rebuild every accepted example from stored stage artifacts before auditing.
        reconstructed = []
        for group in groups.values():
            try:
                accepted, _ = evaluate_group(group, rules[group['id']], facts)
                reconstructed.extend(accepted)
            except (ValueError, KeyError, TypeError):
                continue
        reconstructed.sort(key=lambda r: r['id'])
        train = read_jsonl(args.output / 'train.jsonl')
        manifest = json.loads((args.output / 'manifest.json').read_text())
        if reconstructed != train or fingerprint(train) != manifest['train_sha256']:
            raise ValueError('Saved training data differs from evidence reconstruction')
        print(json.dumps(audit_training(train, sources), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'run_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Output belongs to another configuration; use a new directory')
    path.write_text(json.dumps(config, indent=2) + '\n')
    write_jsonl(args.output / 'plan.jsonl', [{'group_id': j['id'], 'source_id': j['source']['id'],
                                           'source_family': j['source']['family'], 'method': j['method'],
                                           'domain': j['source']['domain'],
                                           'primitive': j['source']['input']['questions']['decision']['type']}
                                          for j in jobs])
    print(json.dumps({'seeds': len(jobs), 'methods': dict(Counter(j['method'] for j in jobs)),
                      'maximum_schema_examples': 3 * len(jobs), 'fact_checks': 6 * len(jobs)}), flush=True)
    if args.plan_only:
        return
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('OPENAI_API_KEY is required')
    configure()
    from nimble.datasets.evidence_stages import FactAudit, FactReviewer, RuleAudit, RuleReviewer
    groups = generate(jobs, args.output, args.model)
    by_id = {j['id']: j for j in jobs}
    rules = run_stage(RuleReviewer, RuleAudit,
                     [rule_job(g, by_id[g['id']]['source']) for g in groups.values()],
                     args.output, args.model, 'rule_audits')
    facts = run_stage(FactReviewer, FactAudit,
                     [req for g in groups.values() for req in fact_jobs(g)],
                     args.output, args.model, 'fact_audits')
    manifest = export(groups, rules, facts, jobs, sources, args.output, config)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
