"""Summarize recorded per-request timings without making model/API calls."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.compat import model_key as normalized_model_key


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(name, relative_path, count, dataset, runtime, boundary, *, model_key=None, server_timing=False):
    path = PROJECT_ROOT / relative_path
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())['rows']
    if model_key is not None:
        rows = [r for r in rows if normalized_model_key(r['model']) == normalized_model_key(model_key)]
    if model_key is not None and any(r.get('error') for r in rows):
        raise ValueError('Timing run contains failures: ' + name)
    if len(rows) != count or len({r['id'] for r in rows}) != count:
        raise ValueError('Unexpected timing coverage: ' + name)
    milliseconds = [(r['response']['scoring_seconds'] if server_timing else r['elapsed_seconds']) * 1000 for r in rows]
    if any(not math.isfinite(t) or t <= 0 for t in milliseconds):
        raise ValueError('Invalid elapsed time: ' + name)
    return {'name': name, 'count': count, 'dataset': dataset, 'runtime': runtime,
            'timing_boundary': boundary, 'mean_ms': statistics.mean(milliseconds),
            'median_ms': statistics.median(milliseconds), 'p95_ms': percentile(milliseconds, .95),
            'min_ms': min(milliseconds), 'max_ms': max(milliseconds),
            'source': 'Unpublished local evaluation artifact',
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'warmup_exclusion': 'None; all recorded requests retained',
            'repetitions_per_example': 1}


def main():
    models = []
    names = [('gemma3_270m','Gemma 3 270M IT'), ('qwen35_08b','Qwen3.5-0.8B'),
             ('qwen35_4b','Qwen3.5-4B'), ('qwen35_9b','Qwen3.5-9B'), ('qwen38_27b','Qwen3.8-27B')]
    for slug, name in names:
        models.append(summarize(name,
            f'evaluations/openjeff_mixed9b_v1/expanded_baselines/{slug}/results.json', 324,
            '324-example contrastive holdout', 'H100 80GB; batch 1, concurrency 1',
            'In-process adaptation, tokenization, synchronized GPU scoring and metric calculation; excludes model load and network'))
    models.append(summarize('Bespoke-Nimble-9B',
        'evaluations/lint_gaming_eval_v1/openjeff_responses.jsonl', 120,
        '120-example linting/gaming evaluation', 'H100 80GB; batch 1, concurrency 1',
        'Synchronized in-process model.score with the published unmerged LoRA; includes tokenization and probabilities, excludes model load and network'))
    models.append(summarize('Jev 1.13.0',
        'evaluations/openjeff_mixed9b_v1/jev-1.13.0-324/responses.jsonl', 324,
        '324-example contrastive holdout', 'TypeSafe API; client concurrency 4',
        'Client request wall time including network, service processing, response validation and retries/backoff if any; excludes waiting in local worker queue'))
    for slug, name, provider in [('deepseek_v41_flash','DeepSeek-V4.1-Flash','Fireworks'),
                                  ('qwen38_24t','Qwen3.8 2.4T A95B','Modal')]:
        item = summarize(name, f'evaluations/large_models_100_20260917/{slug}/rows.jsonl', 100,
            '100-example general evaluation subset', f'OpenRouter / {provider}',
            'Client HTTP round trip with medium reasoning and regular decoding; includes token-limit failures; excludes earlier superseded transport attempts')
        item['decode_mode'] = 'Regular decoding, medium reasoning, maximum 4096 output/reasoning tokens'
        models.append(item)
    paired_path = 'evaluations/openjeff_diverse9b_v2/latency_324/responses.jsonl'
    unmeasured = [{'name':'nimble-diverse9b-v2',
                  'reason':'Training/evaluation saved quality metrics, but no per-request inference timings'}]
    if (PROJECT_ROOT / paired_path).exists():
        for name, key, scoring, boundary in [
            ('nimble-diverse9b-v2 (GPU scoring)', 'nimble', True,
             'Synchronized model.score including tokenization and probability formatting; excludes loading, HTTP and network'),
            ('nimble-diverse9b-v2 (client round trip)', 'nimble', False,
             'Same local client, full input sent through private SSH tunnel; HTTP request through JSON response parsing'),
            ('Jev 1.13.0 (paired client round trip)', 'jev', False,
             'Same local client, official TypeSafe HTTPS API; HTTP request through JSON response parsing')]:
            item = summarize(name, paired_path, 324, 'Same 324-example contrastive holdout',
                             'H100 80GB' if scoring else 'Same Mac client; alternating provider order, persistent sessions',
                             boundary, model_key=key, server_timing=scoring)
            item['warmup_exclusion'] = 'Three explicit unscored warmups per provider, one per question type'
            item['concurrency'] = 1
            source_keys = {r['model'] for r in [json.loads(line) for line in (PROJECT_ROOT / paired_path).read_text().splitlines()]
                           if normalized_model_key(r['model']) == key}
            if len(source_keys) != 1:
                raise ValueError('Mixed source model identifiers in saved run')
            item['source_model_key'] = source_keys.pop()
            item['source_timing_field'] = 'response.scoring_seconds' if scoring else 'elapsed_seconds'
            models.append(item)
        unmeasured = []
    output = PROJECT_ROOT / 'ignore/docs/assets/model-latency.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'measurement': 'Observed saved-run latency, not a matched serving benchmark',
              'percentile_method': 'Linear interpolation at (n-1)*q; milliseconds',
              'models': models,
              'unmeasured': unmeasured}
    output.write_text(json.dumps(result, indent=2) + '\n')
    for row in models:
        print(f"| {row['name']} | {row['count']} | {row['median_ms']:.1f} | {row['mean_ms']:.1f} | {row['p95_ms']:.1f} | {row['runtime']} |")


if __name__ == '__main__':
    main()
