#!/usr/bin/env python3
"""Run fixed external comparison suites against any TypeSafe-compatible endpoint."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT.parent / 'decision-v7' / 'run_test.py'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    catalog = load(ROOT / 'catalog.json')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', nargs='+', choices=['all'] + list(catalog), default=['all'])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, help='Smoke test: first N records per suite; baseline comparison is disabled')
    ap.add_argument('--base-url', default='http://127.0.0.1:55733')
    ap.add_argument('--model', default='kev-latest')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--workers', type=int, default=1, help='Maximum concurrent requests per suite (default: 1)')
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be positive')
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be positive')
    if args.list:
        for name, entry in catalog.items():
            print(f"{name:24} {entry['records']:4} records  {entry['description']}")
        return 0
    suites = list(catalog) if 'all' in args.suite else list(dict.fromkeys(args.suite))
    for name in suites:
        directory = ROOT / 'datasets' / name
        manifest = load(directory / 'manifest.json')
        data = (directory / 'data.jsonl').read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['sha256']:
            raise ValueError(f'{name}: dataset checksum differs')
        rows = [json.loads(line) for line in data.split(b'\n') if line.strip()]
        if len(rows) != manifest['records'] or len(rows) != catalog[name]['records']:
            raise ValueError(f'{name}: record count differs')
    out = args.out or ROOT / 'output' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=False)
    comparison = {}
    failed = False
    for name in suites:
        command = [sys.executable, str(RUNNER), '--data', str(ROOT / 'datasets' / name / 'data.jsonl'),
                   '--base-url', args.base_url, '--model', args.model, '--workers', str(args.workers)]
        if args.dry_run: command.append('--dry-run')
        else: command += ['--out', str(out / name)]
        if args.limit is not None: command += ['--limit', str(args.limit)]
        print(f'\n=== {name} ===', flush=True)
        code = subprocess.run(command).returncode
        failed |= code != 0
        if args.dry_run:
            continue
        summary_path = out / name / 'summary.json'
        if summary_path.exists():
            summary = load(summary_path)
            comparable = args.limit is None and summary['complete'] and summary['errors'] == 0
            baselines = []
            for baseline in catalog[name].get('baselines', []):
                value = summary.get(baseline['metric'])
                baselines.append({**baseline, 'local_value': value,
                                  'difference': value - baseline['value'] if comparable and value is not None else None})
            comparison[name] = {'summary': summary, 'full_run_comparable': comparable, 'published_baselines': baselines}
            (out / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            lines = ['# 测试结果与外部基线', '', f'模型：`{args.model}`；地址：`{args.base_url}`。', '',
                     '差值只在全量运行、全部完成且无错误时显示。正值表示本地数值更高；MAE 越低越好。', '',
                     '| 测试集 | 指标 | 本地 | 外部模型 | 已发表值 | 差值 |',
                     '|---|---|---:|---|---:|---:|']
            for suite, result in comparison.items():
                if not result['published_baselines']:
                    lines.append(f"| {suite} | accuracy | {result['summary']['accuracy']:.4f} | — | — | — |")
                for b in result['published_baselines']:
                    diff = '—' if b['difference'] is None else f"{b['difference']:+.4f}"
                    local = '—' if b['local_value'] is None else f"{b['local_value']:.4f}"
                    lines.append(f"| {suite} | {b['metric']} | {local} | {b['model']} | {b['value']:.4f} | {diff} |")
            (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        if code != 0:
            print(f'Stopped after {name}: incomplete run or request errors. Completed outputs are preserved.')
            break
    if not args.dry_run:
        print(f'\nReport: {(out / "REPORT.md").resolve()}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
