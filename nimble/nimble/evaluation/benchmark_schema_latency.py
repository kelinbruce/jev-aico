"""Measure resident schema-adapter inference, including request preparation."""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import prepare_prompts
from nimble.training.model_loading import load_base
from nimble.training.schema_data import fingerprint, read_rows
from nimble.training.schema_train import CandidateCollator, candidate_logits, decision_result


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {'count':len(values), 'mean':float(values.mean()), 'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)), 'p99':float(np.percentile(values,99)),
            'min':float(values.min()), 'max':float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--allow-device-differences',action='store_true',
                        help='Record cross-GPU differences from saved logits instead of requiring bitwise equality')
    args=parser.parse_args()
    if args.repetitions<1 or args.warmup<1:
        raise ValueError('Positive repetitions and warmup required')
    contract=json.loads((args.adapter/'schema_config.json').read_text())
    raw=read_rows(args.data)
    assert fingerprint(raw)==contract['data_audit']['validation_fingerprint']
    prompt_path=Path(__file__).parents[1]/'scoring/parallel_schema.py'
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest()==contract['prompt_code_sha256']
    expected={r['id']:r for r in json.loads((args.adapter/'after.json').read_text())['rows']}
    torch.cuda.synchronize()
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(args.adapter)
    model=PeftModel.from_pretrained(load_base(contract['model'],contract['revision']),args.adapter).eval()
    collator=CandidateCollator(tokenizer.pad_token_id)
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-started

    def request(raw):
        torch.cuda.synchronize()
        start=time.perf_counter()
        context,schema=adapt_input(raw['input'])
        prepared=prepare_prompts(tokenizer,context,schema,contract['max_length'])
        assert len(prepared.names)==1
        row={'input_ids':prepared.full_ids[0], 'candidate_ids':prepared.candidate_ids[0],
             'choices':prepared.choices[0], 'kind':raw['input']['questions']['decision']['type']}
        inputs={k:v.to('cuda') for k,v in collator([row]).items()}
        torch.cuda.synchronize()
        forward_start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=candidate_logits(model,inputs)[0]
        torch.cuda.synchronize()
        forward_end=time.perf_counter()
        result=decision_result(row,logits.cpu())
        json.dumps(result,allow_nan=False)
        end=time.perf_counter()
        return {'id':raw['id'],'kind':row['kind'],'tokens':len(row['input_ids']),
                'server_ms':(end-start)*1000,'forward_ms':(forward_end-forward_start)*1000},result

    reference_comparisons={}
    device_results={}
    def verify(raw,result):
        saved=expected[raw['id']]
        assert set(result['logits'])==set(saved['logits'])
        delta=max(abs(result['logits'][k]-saved['logits'][k]) for k in saved['logits'])
        reference_comparisons[raw['id']]={'prediction_matches':result['prediction']==saved['prediction'],
                                        'max_logit_difference':delta,
                                        'prediction':result['prediction'],
                                        'correct':result['prediction']==raw['reference']['target']}
        if not args.allow_device_differences:
            assert result['prediction']==saved['prediction'] and delta==0,raw['id']
        if raw['id'] in device_results:
            assert result==device_results[raw['id']], 'Repeated result changed on the same GPU: '+raw['id']
        else:
            device_results[raw['id']]=result

    warmup_order=list(raw)
    random.Random(17).shuffle(warmup_order)
    first_request=None
    for i in range(args.warmup):
        raw_row=warmup_order[i%len(raw)]
        timed,result=request(raw_row)
        verify(raw_row,result)
        if i==0:first_request=timed
    print(json.dumps({'phase':'warmup_complete','load_seconds':load_seconds,'first_request':first_request}),flush=True)
    torch.cuda.reset_peak_memory_stats()
    measurements=[]
    for repeat in range(args.repetitions):
        order=list(raw)
        random.Random(29+repeat).shuffle(order)
        for raw_row in order:
            timed,result=request(raw_row)
            verify(raw_row,result)
            measurements.append({'repeat':repeat+1,**timed})
        print(json.dumps({'phase':'measured','repeat':repeat+1,'requests':len(measurements)}),flush=True)
    summary={'server_ms':distribution([r['server_ms'] for r in measurements]),
             'forward_ms':distribution([r['forward_ms'] for r in measurements]),
             'prompt_tokens':distribution([r['tokens'] for r in measurements]),
             'serial_requests_per_second':1000/np.mean([r['server_ms'] for r in measurements])}
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'model':contract['model'],
            'revision':contract['revision'],'adapter':str(args.adapter),'gpu':torch.cuda.get_device_name(),
            'precision':'BF16 with unmerged LoRA','attention':'sdpa','batch_size':1,'concurrency':1,
            'data_fingerprint':fingerprint(raw),'unique_examples':len(raw),'repetitions':args.repetitions,
            'warmup_requests':args.warmup,'model_load_seconds':load_seconds,'first_request':first_request,
            'summary':summary,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'predictions_and_logits_match_saved_evaluation':all(v['prediction_matches'] and v['max_logit_difference']==0 for v in reference_comparisons.values()),
            'allow_device_differences':args.allow_device_differences,
            'repeated_results_identical':True,
            'reference_comparison':{'unique_examples':len(reference_comparisons),
                                    'prediction_matches':sum(v['prediction_matches'] for v in reference_comparisons.values()),
                                    'correct':sum(v['correct'] for v in reference_comparisons.values()),
                                    'max_logit_difference':max(v['max_logit_difference'] for v in reference_comparisons.values()),
                                    'per_example':reference_comparisons},
            'versions':{p:version(p) for p in ['torch','transformers','peft']},
            'method':'Synchronized wall time. Server timing includes schema adaptation, prompt rendering/tokenization, tensor construction/transfer, one forward pass, probability conversion and JSON serialization. Resident model; parsed input already in memory. No network, HTTP stack, queueing, model loading, prefix-cache reuse, batching, or generated reasoning. This is the existing PyTorch scoring path, not an optimized serving benchmark.',
            'measurements':measurements}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'phase':'complete','summary':summary,'peak_gpu_allocated_gib':report['peak_gpu_allocated_gib']}),flush=True)


if __name__=='__main__':
    main()
