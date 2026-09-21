"""Compare frozen domain-evaluation labels with Jev and saved Nimble outputs."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
import urllib.error
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from nimble.paths import PROJECT_ROOT
from nimble.compat import evaluation_settings, response_path
from nimble.datasets.dataset_io import request_for, validate_teacher
from nimble.datasets.typesafe_curator_bridge import call_typesafe
from nimble.evaluation.evaluate_pilot import assess
from nimble.training.schema_data import fingerprint, read_rows


def summarize(rows):
    result={}
    for kind in ('all','choice','noul','score'):
        selected=[r for r in rows if kind=='all' or r['kind']==kind]
        if not selected:continue
        result[kind]={'count':len(selected),'correct':sum(r['correct'] for r in selected)}
        result[kind]['accuracy']=result[kind]['correct']/len(selected)
        for metric in ('nll','brier','score_absolute_error'):
            if all(metric in r for r in selected):result[kind][metric]=sum(r[metric] for r in selected)/len(selected)
    return result


def jev_evaluate(rows, output, model):
    progress=output/'jev_responses.jsonl'
    saved=read_rows(progress) if progress.exists() and progress.stat().st_size else []
    lookup={r['id']:r for r in rows}
    for r in saved:
        assert r['input_fingerprint']==fingerprint(lookup[r['id']]['input'])
        validate_teacher(lookup[r['id']],r['response'],model)
    def request(row):
        started=time.monotonic()
        for attempt in range(4):
            try:
                response=call_typesafe(request_for(row,model),os.environ['TYPESAFE_API_KEY']);break
            except urllib.error.HTTPError as e:
                if e.code not in (429,502,503,504,529) or attempt==3:raise RuntimeError(f'TypeSafe HTTP {e.code}') from None
                time.sleep(2**attempt)
        validate_teacher(row,response,model)
        return {'id':row['id'],'input_fingerprint':fingerprint(row['input']),'response':response,
                'elapsed_seconds':time.monotonic()-started,'created_utc':datetime.now(timezone.utc).isoformat()}
    pending=[r for r in rows if r['id'] not in {s['id'] for s in saved}]
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,progress.open('a') as f:
        for future in concurrent.futures.as_completed([pool.submit(request,r) for r in pending]):
            try:record=future.result()
            except Exception as e:errors.append(str(e));continue
            saved.append(record);f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
            if len(saved)%10==0:print(f'Jev {len(saved)}/{len(rows)}',flush=True)
    if errors:raise RuntimeError('Some Jev calls failed; successful results saved: '+str(errors))
    return {r['id']:r['response'] for r in saved}


def assess_rows(rows, responses, model):
    result=[]
    for row in rows:
        kind=row['input']['questions']['decision']['type']
        response=responses[row['id']]
        if model=='Jev 1.13.0':
            answer=response['answers']['decision']
            if kind=='noul':probs={'false':1-answer['noul'],'true':answer['noul']}
            else:
                labels=list(row['input']['questions']['decision']['criteria']) if kind=='choice' else [str(i) for i in range(len(row['input']['questions']['decision']['criteria']))]
                total=sum(answer['probabilities'].values());probs={k:answer['probabilities'][k]/total for k in labels}
            a=assess(probs,row['reference']['target'],kind)
            prediction=answer['choice'] if kind=='choice' else answer['noul']>.5 if kind=='noul' else a['prediction']
        else:
            assert response['input_fingerprint']==fingerprint(row['input'])
            probs=response['result']['fields']['decision']['probabilities']
            a=assess(probs,row['reference']['target'],kind)
            prediction=response['result']['fields']['decision']['prediction']
            assert prediction==a['prediction']
        r={'id':row['id'],'domain':row['domain'],'family':row['family'],'subtopic':row['subtopic'],
           'kind':kind,'target':row['reference']['target'],'prediction':prediction,
           'correct':prediction==row['reference']['target'],'probabilities':probs,
           'nll':a['negative_log_likelihood'],'brier':a['multiclass_brier']}
        if kind=='score':r.update(expected_score=a['expected_score'],score_absolute_error=a['absolute_score_error'])
        result.append(r)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'evaluations/lint_gaming_eval_v1')
    parser.add_argument('--jev-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();rows=read_rows(args.data);out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.data.with_name('manifest.json').read_text());assert fingerprint(rows)==manifest['record_fingerprint']
    assert hashlib.sha256(args.data.read_bytes()).hexdigest()==manifest['dataset_sha256']
    settings={'dataset_fingerprint':fingerprint(rows),'dataset_sha256':manifest['dataset_sha256'],'jev_model':'jev-1.13.0','nimble_model':'Bespoke-Nimble-9B',
              'evaluation_only':True,'labels_frozen_before_evaluation':True,'request_fields':['model','state','questions']}
    settings_path=out/'settings.json'
    if settings_path.exists():assert evaluation_settings(json.loads(settings_path.read_text()))==settings
    else:settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    if args.report_only:
        saved=read_rows(out/'jev_responses.jsonl');assert {r['id'] for r in saved}=={r['id'] for r in rows}
        lookup={r['id']:r for r in rows}
        for r in saved:
            assert r['input_fingerprint']==fingerprint(lookup[r['id']]['input'])
            validate_teacher(lookup[r['id']],r['response'],'jev-1.13.0')
        responses={r['id']:r['response'] for r in saved}
    else:
        load_dotenv(PROJECT_ROOT/'.env');responses=jev_evaluate(rows,out,'jev-1.13.0')
    jev=assess_rows(rows,responses,'Jev 1.13.0')
    (out/'jev_results.json').write_text(json.dumps({'summary':summarize(jev),'rows':jev},indent=2)+'\n')
    if args.jev_only:return
    raw=read_rows(response_path(out));assert {r['id'] for r in raw}=={r['id'] for r in rows}
    nimble=assess_rows(rows,{r['id']:r for r in raw},'Bespoke-Nimble-9B')
    models={'Bespoke-Nimble-9B':nimble,'Jev 1.13.0':jev}
    baseline=None
    if (out/'base_responses.jsonl').exists():
        base_raw=read_rows(out/'base_responses.jsonl')
        assert {r['id'] for r in base_raw}=={r['id'] for r in rows}
        runtime=json.loads((out/'base/runtime.json').read_text())
        assert runtime['adapter_loaded'] is False
        assert runtime['base_model']=='Qwen/Qwen3.5-9B'
        assert runtime['base_revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
        assert runtime['input_records_fingerprint']==json.loads((out/'token_preflight.json').read_text())['input_records_fingerprint']
        baseline=assess_rows(rows,{r['id']:r for r in base_raw},'Original Qwen3.5-9B')
        models={'Original Qwen3.5-9B':baseline,**models}
    summaries={domain:{name:summarize([r for r in rs if domain=='all' or r['domain']==domain]) for name,rs in models.items()}
               for domain in ('all','semantic_code_linting','gaming')}
    paired={domain:dict(Counter('both_correct' if a['correct'] and b['correct'] else 'nimble_only' if a['correct'] else 'jev_only' if b['correct'] else 'neither_correct'
                               for a,b in zip(nimble,jev) if domain=='all' or a['domain']==domain)) for domain in summaries}
    result={**settings,'summary':summaries,'paired':paired,'rows':models,
            'by_subtopic':{topic:{name:summarize([r for r in rs if r['subtopic']==topic]) for name,rs in models.items()} for topic in sorted({r['subtopic'] for r in rows})},
            'jev_zero_gold_probabilities':sum(r['probabilities'][str(r['target']).lower() if isinstance(r['target'],bool) else str(r['target'])]==0 for r in jev),
            'nll_floor':1e-15,'jev_usage':{k:sum(r['usage'][k] for r in responses.values()) for k in ('input_tokens','output_tokens')}}
    if baseline is not None:
        result['base_provenance']=runtime
        result['base_vs_nimble']={domain:dict(Counter(
            'both_correct' if a['correct'] and b['correct'] else 'base_only' if a['correct']
            else 'nimble_only' if b['correct'] else 'neither_correct'
            for a,b in zip(baseline,nimble) if domain=='all' or a['domain']==domain)) for domain in summaries}
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    names=list(models)
    lines=['# New-domain evaluation: semantic code linting and gaming','',
           'Frozen 120-example evaluation; no training, prompt tuning, threshold fitting, or model-driven example selection.','',
           '| Domain | Examples | '+' | '.join(names)+' |','| --- | ---: | '+' | '.join(['---:']*len(names))+' |']
    for domain,s in summaries.items():
        values=[f"{v['all']['correct']}/{v['all']['count']} ({v['all']['accuracy']:.2%})" for v in s.values()]
        lines.append(f"| {domain} | {next(iter(s.values()))['all']['count']} | {' | '.join(values)} |")
    lines += ['','## Task breakdown','','| Domain / type | '+' | '.join(names)+' |','| --- | '+' | '.join(['---:']*len(names))+' |']
    for domain,s in summaries.items():
        for kind in ('choice','noul','score'):
            lines.append(f"| {domain} / {kind} | "+' | '.join(f"{v[kind]['correct']}/{v[kind]['count']}" for v in s.values())+' |')
    lines += ['','## Probability quality','','| Domain / model | NLL ↓ | Brier ↓ | Expected score MAE ↓ |','| --- | ---: | ---: | ---: |']
    for domain,s in summaries.items():
        for name,v in s.items():lines.append(f"| {domain} / {name} | {v['all']['nll']:.4f} | {v['all']['brier']:.4f} | {v['score']['score_absolute_error']:.4f} |")
    lines += ['','## Task families','','| Task family | '+' | '.join(names)+' |','| --- | '+' | '.join(['---:']*len(names))+' |']
    for topic,stats in result['by_subtopic'].items():
        lines.append('| '+topic+' | '+' | '.join(f"{v['all']['correct']}/{v['all']['count']}" for v in stats.values())+' |')
    if baseline is not None:
        lines += ['','## What fine-tuning changed','','| Domain | Both correct | Base only | Nimble only | Both wrong |','| --- | ---: | ---: | ---: | ---: |']
        for domain,counts in result['base_vs_nimble'].items():
            lines.append('| '+domain+' | '+' | '.join(str(counts.get(k,0)) for k in ('both_correct','base_only','nimble_only','neither_correct'))+' |')
        lines += ['', 'Original Qwen3.5-9B uses the exact pinned base revision and the same frozen tokenizer, prompt, candidate order, and CUDA BF16-autocast scoring implementation as Bespoke-Nimble-9B. The original base is loaded directly without any adapter. Nimble and Jev results are reused from their completed evaluations on these identical frozen inputs.', '']
    lines += ['','## Method and limits','',
              'Each domain contains 60 examples: 20 Choice, 20 Boolean/Noul, and 20 Score. Twelve task families have ten examples each. The 60 linting cases contain actual Python snippets that pass syntax parsing; their behavior is reviewed semantically, not validated by executing code.', '',
              'GPT-5.6 Sol generated the examples and reviewed them in separate calls with the proposed labels and rationales hidden. Only agreed, unambiguous, domain-relevant items with complete rubrics passed. This is blind model review, not human ground truth or an independent-model consensus. Selection can favor examples this curator finds easy.', '',
              f"Checks compared states with {manifest['previous_rows_checked']:,} earlier training/evaluation rows and found no exact or token-Jaccard ≥0.8 duplicates. These are new scenarios and task families; prior training already included general software topics, so this is not proof of wholly unseen domains or absence of pretraining contamination.", '',
              'All evaluated models receive the identical state, instructions and criteria. Reference labels, generator reasons, and review notes are excluded from their inputs. Nimble uses the frozen 2,676-example-trained adapter, pinned Qwen3.5-9B base, and its original CUDA BF16-autocast candidate scorer. Jev is pinned to 1.13.0. No adaptation to this dataset was performed.', '',
              'Choice uses the API choice; Noul uses P(true)>0.5 for Jev and candidate argmax for Nimble. Score accuracy uses the modal level, while expected-score MAE evaluates the weighted score. All NLLs use a 1e-15 floor; rounded Jev probabilities can make zero-probability mistakes dominate NLL. No claim about comparable latency is made.', '',
              'The small sample and shared task families limit generalization. Inspect disagreements and labels before drawing deployment conclusions. Code linting here is schema-based semantic judgment, not an exhaustive security audit or compiler analysis.', '',
              'Sources: [TypeSafe use-case map](https://docs.typesafe.ai/concepts/use-case-map#gaming), [Choice moderation cookbook](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook).', '',
              'Artifacts: `comparison.json`, `nimble_responses.jsonl`, `jev_responses.jsonl`, and `base_responses.jsonl` when the baseline has been evaluated; curation plans, request caches and frozen labels are in `data/lint_gaming_eval_v1/`.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
