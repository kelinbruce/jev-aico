"""Replay a release with its recorded runner source and unchanged gate dependencies."""
import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from nimble.paths import PROJECT_ROOT
from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.create_diverse_dataset import read_jsonl


def relocated_source(source):
    """Translate import locations only; preserve recorded gate logic and prompts."""
    source = re.sub(r'(?m)^(\s*(?:from|import)\s+)(?:openjevons|openjeff)(?=[.\s])',
                    r'\1nimble', source)
    for old, new in (
        ('nimble.datasets.minicheck_data', 'nimble.datasets.contrastive_data'),
        ('nimble.datasets.create_tiny_dataset', 'nimble.datasets.dataset_io'),
        ('nimble.datasets.create_minicheck_dataset import llm_options',
         'nimble.datasets.curation_options import llm_options'),
    ):
        source = source.replace(old, new)
    return source


def same_lineage(current, recorded):
    """A directory move is valid only when all recorded plan metadata matches."""
    return (Path(current['directory']).resolve() == Path(recorded['directory']).resolve()
            and {k: v for k, v in current.items() if k != 'directory'}
            == {k: v for k, v in recorded.items() if k != 'directory'})


def verify_snapshot(snapshot, config, directory):
    if fingerprint(snapshot['files']) != snapshot['sha256'] or snapshot['sha256'] != config['implementation_sha256']:
        raise ValueError('Recorded implementation fingerprint differs')
    for name, content in snapshot['files'].items():
        if name not in ('fast_training_dataset.py', 'benchmark_curation_models.py'):
            current_name = 'contrastive_data.py' if name == 'minicheck_data.py' else name
            if (directory / current_name).read_text() != relocated_source(content):
                raise ValueError('Recorded dependency changed beyond import relocation: ' + name)


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('snapshot',type=Path);opts=p.parse_args()
    cfg=json.loads((opts.directory/'run_config.json').read_text());snapshot=json.loads(opts.snapshot.read_text())
    verify_snapshot(snapshot, cfg, PROJECT_ROOT/'nimble/datasets')
    namespace={'__name__':'archived_fast_training_runner','__file__':str(PROJECT_ROOT/'nimble/datasets/fast_training_dataset.py')}
    exec(compile(relocated_source(snapshot['files']['fast_training_dataset.py']), str(opts.snapshot)+':fast_training_dataset.py', 'exec'),namespace)
    args=namespace['parser']().parse_args(['--offline','--group-concurrency','32'])
    for name in ('target','seed','max_edit_words','attempt_multiplier','generator_model','verifier_model','draft_reasoning','check_reasoning','guidance_from_variation'):
        if name in cfg:setattr(args,name,cfg[name])
    args.output=opts.directory.resolve();args.source=Path(cfg['source']);args.plans=Path(cfg['rule_plan_lineage']['directory'])
    if cfg.get('resume_policy'):
        args.resume_policy=args.output/'offline_replay_policy.json'
        args.resume_policy.write_text(json.dumps(cfg['resume_policy'],indent=2)+'\n')
    sources=read_jsonl(args.source);assert fingerprint(sources)==cfg['source_sha256']
    plans,lineage=namespace['load_training_plans'](args.plans,sources)
    if not same_lineage(lineage, cfg['rule_plan_lineage']):
        raise ValueError('Recorded rule-plan lineage differs')
    stage=namespace['AsyncStages'](args.output,args,None);started=time.monotonic()
    with asyncio.Runner() as runner:
        runner.get_loop().set_default_executor(namespace['curation_executor'](args.group_concurrency))
        manifest=runner.run(namespace['curate'](args,sources,plans,cfg,stage))
    assert stage.api_calls==0
    audit={'status':'passed','at':datetime.now(timezone.utc).isoformat(),'api_calls':stage.api_calls,'cache_hits':stage.cache_hits,'elapsed_seconds':time.monotonic()-started,'implementation_sha256':cfg['implementation_sha256'],'snapshot':str(opts.snapshot.resolve()),'train_sha256':manifest['train_sha256'],'rows':manifest['training']['examples'],'recorded_runner_replayed':True,'gate_dependencies_unchanged':True}
    (args.output/'offline_replay_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__ == "__main__":
    main()
