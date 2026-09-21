"""Compare new models and saved baselines on exactly the same evaluation IDs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from nimble.evaluation.evaluate_models import teacher_assessment
from nimble.paths import PROJECT_ROOT

LABELS={'gemma3_270m':'Gemma 3 270M IT','qwen35_08b':'Qwen3.5-0.8B','qwen35_4b':'Qwen3.5-4B',
        'qwen35_9b':'Qwen3.5-9B','qwen38_27b':'Qwen3.8-27B','deepseek_v41_flash':'DeepSeek-V4.1-Flash',
        'qwen38_24t':'Qwen3.8 2.4T A95B','jev':'Jev-1.13.0'}


def read_rows(path):
    rows=[json.loads(l) for l in path.read_text().splitlines() if l]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs: '+str(path))
    return {r['id']:r for r in rows}


def wilson(correct,n):
    z=1.95996398454;p=correct/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def build(root,data,baselines):
    raw=data.read_bytes();dataset=read_rows(data);ids=list(dataset)
    models={};normalized={}
    for name in LABELS:
        if name=='jev':
            source={rid: {'student':teacher_assessment(row,row['input']['questions']['decision']['type']),
                          'reference':row['reference']} for rid,row in dataset.items()}
            mode='Saved TypeSafe judgments'
        else:
            path=(baselines/name/'rows.jsonl') if name in ('gemma3_270m','qwen35_08b','qwen35_4b','qwen35_9b') else root/name/'rows.jsonl'
            if name in ('qwen38_27b','deepseek_v41_flash','qwen38_24t'):
                settings=json.loads((path.parent/'settings.json').read_text())
                if settings['dataset_sha256']!=hashlib.sha256(raw).hexdigest():
                    raise ValueError('Dataset hash mismatch for '+name)
            source=read_rows(path)
            if not set(ids)<=set(source):raise ValueError('Missing evaluation IDs for '+name)
            if name in ('deepseek_v41_flash','qwen38_24t'):mode='OpenRouter decoding; medium reasoning'
            else:mode='BF16 candidate scoring; no reasoning'
        rows=[]
        for rid,row in dataset.items():
            saved=source[rid]
            if saved['reference']!=row['reference']:raise ValueError('Reference mismatch')
            hard=saved if name in ('deepseek_v41_flash','qwen38_24t') else saved['student']
            valid=hard.get('valid',True);prediction=hard['prediction']
            kind=row['input']['questions']['decision']['type']
            teacher=teacher_assessment(row,kind)['prediction']
            result={'id':rid,'domain':row['domain'],'type':kind,'reference':row['reference']['target'],
                    'prediction':prediction,'valid':valid,'correct':valid and prediction==row['reference']['target'],
                    'teacher_agreement':valid and prediction==teacher}
            rows.append(result)
        normalized[name]=rows;n=len(rows);correct=sum(r['correct'] for r in rows)
        metrics={'name':LABELS[name],'method':mode,'count':n,'correct':correct,'accuracy':correct/n,
                 'wilson_95':wilson(correct,n),'invalid':sum(not r['valid'] for r in rows),
                 'teacher_agreement':sum(r['teacher_agreement'] for r in rows),
                 'by_type':{},'by_domain':{}}
        for key in ('type','domain'):
            for value in sorted({r[key] for r in rows}):
                group=[r for r in rows if r[key]==value]
                metrics['by_'+key][value]={'count':len(group),'correct':sum(r['correct'] for r in group)}
        scores=[r for r in rows if r['type']=='score' and r['valid']]
        metrics['hard_score_mae_on_valid']=statistics.mean(abs(r['prediction']-r['reference']) for r in scores)
        metrics['hard_score_mae_count']=len(scores)
        if name not in ('deepseek_v41_flash','qwen38_24t'):
            native=[source[rid]['student'] for rid in ids]
            metrics['probability_metrics']={
                'mean_negative_log_likelihood':statistics.mean(r['negative_log_likelihood'] for r in native),
                'mean_multiclass_brier':statistics.mean(r['multiclass_brier'] for r in native),
                'expected_score_mae':statistics.mean(source[r['id']]['student']['absolute_score_error'] for r in rows if r['type']=='score')}
        else:
            metrics['probability_metrics']=None
        models[name]=metrics
    paired={}
    base={r['id']:r for r in normalized['qwen35_9b']}
    for name in ('qwen38_27b','deepseek_v41_flash','qwen38_24t'):
        paired[name]=dict(Counter(('both_correct' if base[r['id']]['correct'] else 'new_only_correct') if r['correct']
                                  else ('baseline_only_correct' if base[r['id']]['correct'] else 'both_incorrect')
                                  for r in normalized[name]))
    result={'dataset_sha256':hashlib.sha256(raw).hexdigest(),'count':len(dataset),'models':models,'paired_vs_qwen35_9b':paired}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'comparison_rows.json').write_text(json.dumps(normalized,indent=2)+'\n')
    lines=['# Larger-model evaluation on 100 shared examples','',
           'All models are compared on the exact same 100 IDs, sampled without inspecting model outputs or reference values from the previous 1,000-example evaluation set. Ten examples per domain; 34 Choice, 33 Noul, 33 Score. Seed: 20260917. No training or prompt tuning.',
           '', 'References were generated by GPT-5.6 Sol and have not been human-reviewed. These metrics measure reference agreement, not established ground-truth accuracy. Saved Jev labels are a separate comparator.', '',
           '| Model | Reference matches | 95% Wilson interval | Jev agreement | Invalid / missing answers |',
           '| --- | ---: | ---: | ---: | ---: |']
    for m in models.values():
        lo,hi=m['wilson_95'];lines.append(f"| {m['name']} | {m['correct']}/{m['count']} ({m['accuracy']:.1%}) | {lo:.1%}–{hi:.1%} | {m['teacher_agreement']}/{m['count']} | {m['invalid']} |")
    lines+=['','Intervals illustrate the uncertainty from only 100 examples. Sampling is stratified and all models share the same examples; the intervals do not incorporate synthetic-label error or prove a ranking.','','## By question type','',
            '| Model | Choice (34) | Noul (33) | Score (33) | Hard score MAE ↓ |',
            '| --- | ---: | ---: | ---: | ---: |']
    for m in models.values():
        g=m['by_type'];lines.append(f"| {m['name']} | {g['choice']['correct']}/34 | {g['noul']['correct']}/33 | {g['score']['correct']}/33 | {m['hard_score_mae_on_valid']:.4f} ({m['hard_score_mae_count']} valid) |")
    lines+=['','## Probability metrics for candidate-scored models','',
            '| Model | NLL ↓ | Multiclass Brier ↓ | Expected score MAE ↓ |',
            '| --- | ---: | ---: | ---: |']
    for m in models.values():
        pm=m['probability_metrics']
        if pm is not None:
            lines.append(f"| {m['name']} | {pm['mean_negative_log_likelihood']:.4f} | {pm['mean_multiclass_brier']:.4f} | {pm['expected_score_mae']:.4f} |")
    lines+=['','API decoded answers have no candidate distribution, so their probability metrics are unavailable. NLL uses natural logarithms and clips zero probabilities to 1e-15, as in prior reports. Multiclass Brier sums squared errors over candidates.',
            '','Hard score MAE compares the selected level with the reference level. It differs from the probability-weighted expected-score MAE in earlier reports. Failed outputs count as incorrect in agreement metrics and are excluded from MAE with their count shown.','','## Protocol','',
            '- Qwen3.8-27B: pinned official checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, unquantized BF16, NVIDIA A100 80 GB, existing CUDA candidate-logit scorer, temperature 1, thinking disabled. All 18 checkpoint weight hashes are verified.',
            '- DeepSeek-V4.1-Flash: `deepseek/deepseek-v4.1-flash`, OpenRouter through Fireworks. Qwen3.8 2.4T A95B: `qwen/qwen3.8-2.4t-a95b`, OpenRouter through Modal. Provider fallbacks disabled; existing account privacy controls respected.',
            '- OpenRouter uses ordinary decoding at temperature 0, medium reasoning requested, and a 4,096-token completion limit. The final answer must be exactly one valid uppercase choice code; surrounding whitespace is allowed. Reasoning text is never parsed as the answer. Raw responses, returned model/provider IDs, usage, cost, finish reasons, and timing are retained.',
            '- The system instruction, state, schema, choice descriptions, ordering, and field request match the existing evaluator. API and GPU chat templates and decoding methods differ. Only input.state and input.questions are sent to models; references and teacher answers are excluded.',
            '- For decoded API answers, Brier/NLL and expected-score metrics are unavailable. No self-reported or fabricated probability distribution is used. Local 27B probability metrics remain in its results.json.',
            '- HTTP 429 rate-limit errors are archived and retried with reduced concurrency. Valid answers, incorrect answers, and token-limit completions are never retried based on their content. Failed initial provider routing is retained under the earlier diagnostics directory.',
            '- Earlier Gemma/Qwen/Jev outputs are reused on these exact IDs. They are not the headline percentages from the full 1,000-example set. Local-vs-API latency is not treated as a comparable benchmark.',
            '', '## Per-domain reference matches','', '| Domain | '+ ' | '.join(m['name'] for m in models.values())+' |', '| --- | '+' | '.join('---:' for _ in models)+' |']
    for domain in sorted({r['domain'] for r in dataset.values()}):
        lines.append('| '+domain+' | '+' | '.join(str(m['by_domain'][domain]['correct'])+'/10' for m in models.values())+' |')
    lines+=['','## Artifacts','',
            '- [Dataset](../../data/typesafe_eval_100_gpt56/all.jsonl) and [selection manifest](../../data/typesafe_eval_100_gpt56/manifest.json)',
            '- [Machine-readable comparison](comparison.json), [per-example comparison](comparison_rows.json)',
            '- [Qwen 27B probabilities and metrics](qwen38_27b/results.json), [DeepSeek metrics](deepseek_v41_flash/results.json), [Qwen 2.4T metrics](qwen38_24t/results.json)',
            '- [GPU runtime](runtime.json), [weight integrity](checkpoint_integrity.json), [pod lifecycle](run_metadata.json), [costs](costs.json)',
            '- [Cases needing reference review](review_notes.md)',
            '']
    (root/'report.md').write_text('\n'.join(lines))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'evaluations/large_models_100_20260917')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--baselines',type=Path,default=PROJECT_ROOT/'evaluations/runpod_20260917/expanded_1000')
    a=p.parse_args();r=build(a.output_dir,a.data,a.baselines)
    for m in r['models'].values(): print(m['name'],m['correct'],m['invalid'])

if __name__=='__main__':main()
