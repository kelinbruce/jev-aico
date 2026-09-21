"""Reload an adapter and verify identical trained/base evaluation outputs."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from nimble.training.model_loading import load_base
from nimble.training.schema_data import prepare_data
from nimble.training.schema_train import CandidateCollator, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.run / 'schema_config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.run)
    _, validation, audit = prepare_data(args.data_dir, args.validation, tokenizer, contract['max_length'],
                                        contract['model'], contract['revision'])
    assert audit == contract['data_audit'], 'Changed data audit'
    model = PeftModel.from_pretrained(load_base(contract['model'], contract['revision']), args.run).eval()
    collator = CandidateCollator(tokenizer.pad_token_id)
    trained = evaluate(model, validation, collator, args.run / 'reloaded.json')
    with model.disable_adapter():
        base = evaluate(model, validation, collator, args.run / 'base_reloaded.json')
    verification = {'passed': True, 'examples': len(validation)}
    for name, actual, filename in [('adapter', trained, 'after.json'), ('base', base, 'before.json')]:
        expected = json.loads((args.run / filename).read_text())
        assert len(actual['rows']) == len(expected['rows'])
        deltas = []
        for a, b in zip(actual['rows'], expected['rows']):
            assert a['id'] == b['id'] and a['prediction'] == b['prediction']
            deltas.extend(abs(a['logits'][k] - b['logits'][k]) for k in a['logits'])
        verification[f'{name}_max_logit_difference'] = max(deltas)
        assert max(deltas) == 0, (name, max(deltas))
    nonzero = sum(torch.count_nonzero(p).item() for n, p in model.named_parameters() if 'lora_B' in n)
    assert nonzero > 0
    verification['nonzero_lora_B_elements'] = nonzero
    (args.run / 'reload_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
