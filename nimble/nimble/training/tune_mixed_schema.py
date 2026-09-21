"""Bounded 9B search over learning rate, adapter rank and complete epochs."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from nimble.training.schema_data import read_rows, fingerprint, validate_separation
from nimble.training.tune_schema import choose_candidate


def completed_trial_history(directory, config, plan, trial_steps):
    """Reuse measured epochs only after their matching checkpoint was saved."""
    epochs=plan['epochs']
    history=json.loads((directory/'epoch_metrics.json').read_text())
    if [h['epoch'] for h in history]!=epochs:
        raise ValueError('Completed trial epochs differ from the reduced plan')
    checkpoint=directory/f"checkpoint-{trial_steps*epochs[-1]}"
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    contract=json.loads((checkpoint/'schema_config.json').read_text())
    expected={'model':plan['model'],'revision':plan['revision'],
              'learning_rate':config['learning_rate'],'lora_rank':config['lora_rank'],
              'max_steps':trial_steps*3,'warmup_steps':math.ceil(trial_steps*3*.1),
              'batch_size':plan['batch_size'],'gradient_accumulation':plan['gradient_accumulation']}
    if any(contract.get(k)!=v for k,v in expected.items()):
        raise ValueError('Saved trial optimization contract differs from the reduced plan')
    if contract['data_audit']['training_fingerprint']!=plan['inner_training_fingerprint']:
        raise ValueError('Saved trial training data differs from the reduced plan')
    if state['global_step']!=trial_steps*epochs[-1] or state['epoch']!=epochs[-1]:
        raise ValueError('Saved trial checkpoint is not at the requested complete epoch')
    if not (checkpoint/'adapter_model.safetensors').is_file():
        raise ValueError('Completed trial has no saved adapter')
    return history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--deadline',type=float,help='Shared job deadline, including preparation')
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if plan['epochs'] not in ([1],[1,2],[1,2,3]):
        raise ValueError('Tuning epochs must be consecutive, between one and three')
    root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def save(name,value):
        p=root/name;tmp=p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    def run(phase,command):
        elapsed=time.monotonic()-started
        save('status.json',{'phase':phase,'state':'running','elapsed_seconds':elapsed})
        print(json.dumps({'phase':phase,'state':'started'}),flush=True)
        remaining=plan['max_runtime_seconds']-elapsed
        if args.deadline is not None:remaining=min(remaining,args.deadline-time.time())
        if remaining<=0:raise TimeoutError('Training job runtime limit reached')
        with (root/(phase+'.log')).open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
    data=Path(plan['data_root']);inner=data/'inner';full=data/'full'
    tr,va,ft,ev=[read_rows(p) for p in [inner/'train.jsonl',inner/'eval.jsonl',full/'train.jsonl',full/'eval.jsonl']]
    assert {r['id'] for r in tr+va}=={r['id'] for r in ft}
    validate_separation(tr,va);validate_separation(ft,ev)
    for rows,key in [(tr,'inner_training_fingerprint'),(va,'inner_validation_fingerprint'),(ft,'full_training_fingerprint'),(ev,'outer_evaluation_fingerprint')]:
        assert fingerprint(rows)==plan[key]
    assert fingerprint(read_rows(data/'original_eval.jsonl'))==plan['original_outer_fingerprint']
    batch=plan['batch_size'];accum=plan['gradient_accumulation']
    trial_steps=math.ceil(math.ceil(len(tr)/batch)/accum)
    refit_steps=math.ceil(math.ceil(len(ft)/batch)/accum)
    common=[sys.executable,'-u','-m','nimble.training.schema_train','train',
            '--model',plan['model'],'--revision',plan['revision'],'--batch-size',str(batch),
            '--gradient-accumulation',str(accum),'--max-length','2048','--seed','17']
    save('plan.json',plan)
    candidates=[]
    try:
        for config in plan['configurations']:
            name=config['name'];directory=root/name
            reuse=config.get('completed_through_epoch')
            if reuse:
                if reuse!=plan['epochs'][-1]:raise ValueError('Reused trial epoch does not match search limit')
                history=completed_trial_history(directory,config,plan,trial_steps)
            elif not (directory/'run_report.json').exists():
                command=common+['--data-dir',str(inner),'--validation',str(inner/'eval.jsonl'),
                    '--output-dir',str(directory),'--learning-rate',str(config['learning_rate']),
                    '--lora-rank',str(config['lora_rank']),'--max-steps',str(trial_steps*3),
                    '--warmup-steps',str(math.ceil(trial_steps*3*.1)),'--save-steps',str(trial_steps),'--eval-each-epoch',
                    '--stop-after-epochs',str(plan['epochs'][-1])]
                if directory.exists():
                    checkpoints=sorted(directory.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                    if not checkpoints:raise ValueError('Incomplete trial has no checkpoint: '+name)
                    command+=['--resume-from-checkpoint',str(checkpoints[-1])]
                run(name,command)
            history=json.loads((directory/'epoch_metrics.json').read_text())
            assert [h['epoch'] for h in history]==plan['epochs']
            candidates.extend({**config,**h} for h in history)
            save('candidates.json',candidates)
        winner=choose_candidate(candidates)
        save('selection.json',{'criterion':'Inner validation mean NLL; ties Brier, fewer epochs, lower LR, then lower adapter rank',
             'selected':winner,'outer_holdout_used_for_selection':False})
        final=root/'refit'
        if not (final/'run_report.json').exists():
            command=common+['--data-dir',str(full),'--validation',str(full/'eval.jsonl'),'--output-dir',str(final),
                '--learning-rate',str(winner['learning_rate']),'--lora-rank',str(winner['lora_rank']),
                '--max-steps',str(refit_steps*3),'--warmup-steps',str(math.ceil(refit_steps*3*.1)),
                '--save-steps',str(refit_steps),'--stop-after-epochs',str(winner['epoch'])]
            if final.exists():
                checkpoints=sorted(final.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                if not checkpoints:raise ValueError('Incomplete refit has no checkpoint')
                command+=['--resume-from-checkpoint',str(checkpoints[-1])]
            run('refit',command)
        report=json.loads((final/'run_report.json').read_text())
        assert report['optimizer_steps']==refit_steps*winner['epoch'] and report['training']['epoch']==winner['epoch']
        run('verify',[sys.executable,'-u','-m','nimble.training.verify_schema_run','--run',str(final),
            '--data-dir',str(full),'--validation',str(full/'eval.jsonl')])
        run('previous_adapter',[sys.executable,'-u','-m','nimble.evaluation.evaluate_schema_adapter',
            '--adapter',plan['previous_adapter'],'--training-file',plan['previous_training_file'],
            '--data',str(full/'eval.jsonl'),'--output',str(root/'previous_adapter_extended.json')])
        files=['adapter_model.safetensors','adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','schema_config.json']
        (final/'artifact.sha256').write_text(''.join(hashlib.sha256((final/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in files))
        save('status.json',{'phase':'complete','state':'complete','elapsed_seconds':time.monotonic()-started,
            'selected':winner,'outer_evaluation':report['after']})
        print((root/'status.json').read_text(),flush=True)
    except Exception as exc:
        save('status.json',{'phase':'failed','state':'failed','error':str(exc),'elapsed_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':
    main()
