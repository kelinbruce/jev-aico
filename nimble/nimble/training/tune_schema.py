"""Bounded learning-rate/epoch search with an untouched outer holdout."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from nimble.training.schema_data import read_rows, validate_separation, fingerprint


def choose_candidate(candidates):
    if not candidates:
        raise ValueError('No completed candidates')
    for c in candidates:
        if c['epoch'] not in (1, 2, 3) or not math.isfinite(c['summary']['all']['nll']):
            raise ValueError('Invalid epoch or inner validation NLL')
    return min(candidates, key=lambda c: (c['summary']['all']['nll'], c['summary']['all']['brier'],
                                          c['epoch'], c['learning_rate'], c.get('lora_rank', 16)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def save(name, value):
        path = root / name
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)
    def run(command, name):
        save('status.json', {'phase': name, 'state': 'running', 'elapsed_seconds': time.monotonic() - started})
        remaining = plan['max_runtime_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Bounded tuning runtime exhausted')
        print(json.dumps({'phase': name, 'state': 'started'}), flush=True)
        with (root / f'{name}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
    inner, outer = Path(plan['inner_data']), Path(plan['outer_data'])
    training, validation = read_rows(inner / 'train.jsonl'), read_rows(inner / 'eval.jsonl')
    full_training, holdout = read_rows(outer / 'train.jsonl'), read_rows(outer / 'eval.jsonl')
    assert len(training) == 800 and len(validation) == len(holdout) == 100 and len(full_training) == 900
    assert {r['id'] for r in training + validation} == {r['id'] for r in full_training}
    validate_separation(training, validation)
    validate_separation(full_training, holdout)
    assert fingerprint(holdout) == plan['outer_evaluation_fingerprint']
    assert fingerprint(training) == plan['inner_training_fingerprint']
    assert fingerprint(validation) == plan['inner_validation_fingerprint']
    save('plan.json', plan)
    common = [sys.executable, '-u', '-m', 'nimble.training.schema_train', 'train',
              '--batch-size', str(plan['batch_size']), '--gradient-accumulation', str(plan['gradient_accumulation']),
              '--max-length', '2048', '--seed', '17', '--lora-rank', '16']
    candidates = []
    try:
        for lr in plan['learning_rates']:
            name = 'lr_' + format(lr, '.0e')
            directory = root / name
            if not (directory / 'run_report.json').exists():
                command = common + ['--data-dir', str(inner), '--validation', str(inner / 'eval.jsonl'),
                                    '--output-dir', str(directory), '--learning-rate', str(lr),
                                    '--max-steps', '300', '--warmup-steps', '30', '--save-steps', '100',
                                    '--eval-each-epoch']
                if directory.exists():
                    checkpoints = sorted(directory.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
                    if not checkpoints:
                        raise ValueError('Incomplete trial has no resumable checkpoint: ' + name)
                    command += ['--resume-from-checkpoint', str(checkpoints[-1])]
                run(command, name)
            history = json.loads((directory / 'epoch_metrics.json').read_text())
            assert [r['epoch'] for r in history] == [1, 2, 3], history
            for record in history:
                candidates.append({'trial': name, 'learning_rate': lr, **record})
            save('candidates.json', candidates)
        selected = choose_candidate(candidates)
        save('selection.json', {'criterion': 'Inner validation mean NLL; ties by Brier, fewer epochs, lower LR',
                                'selected': selected, 'outer_holdout_used_for_selection': False})
        print(json.dumps({'selected': selected}), flush=True)
        # Same three-epoch LR horizon; stop at the selected complete epoch.
        final = root / 'refit'
        if not (final / 'run_report.json').exists():
            command = common + ['--data-dir', str(outer), '--validation', str(outer / 'eval.jsonl'),
                                '--output-dir', str(final), '--learning-rate', str(selected['learning_rate']),
                                '--max-steps', '339', '--warmup-steps', '34', '--save-steps', '113',
                                '--stop-after-epochs', str(selected['epoch'])]
            if final.exists():
                checkpoints = sorted(final.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
                if not checkpoints:
                    raise ValueError('Incomplete refit has no resumable checkpoint')
                command += ['--resume-from-checkpoint', str(checkpoints[-1])]
            run(command, 'refit')
        report = json.loads((final / 'run_report.json').read_text())
        assert report['optimizer_steps'] == 113 * selected['epoch']
        assert report['training']['epoch'] == selected['epoch']
        run([sys.executable, '-u', '-m', 'nimble.training.verify_schema_run', '--run', str(final),
             '--data-dir', str(outer), '--validation', str(outer / 'eval.jsonl')], 'verify_refit')
        files = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'chat_template.jinja', 'schema_config.json']
        (final / 'artifact.sha256').write_text(''.join(
            hashlib.sha256((final / f).read_bytes()).hexdigest() + '  ' + f + '\n' for f in files))
        save('status.json', {'phase': 'complete', 'state': 'complete', 'elapsed_seconds': time.monotonic() - started,
                              'selected_learning_rate': selected['learning_rate'], 'selected_epoch': selected['epoch'],
                              'final_run': str(final), 'outer_evaluation': report['after']})
        print((root / 'status.json').read_text(), flush=True)
    except Exception as exc:
        save('status.json', {'phase': 'failed', 'state': 'failed', 'error': str(exc),
                              'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
