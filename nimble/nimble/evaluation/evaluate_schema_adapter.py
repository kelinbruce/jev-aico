"""Evaluate a saved adapter on another source-family-disjoint labeled set."""
import argparse
import hashlib
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer
from nimble.training.model_loading import load_base
from nimble.training.schema_data import read_rows, fingerprint, validate_separation, as_scoring, encode_scoring, runtime_record
from nimble.training.schema_train import CandidateCollator, evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--training-file',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    contract=json.loads((args.adapter/'schema_config.json').read_text())
    training,validation=read_rows(args.training_file),read_rows(args.data)
    assert fingerprint(training)==contract['data_audit']['training_fingerprint']
    validate_separation(training,validation)
    prompt=Path(__file__).parents[1]/'scoring/parallel_schema.py'
    assert hashlib.sha256(prompt.read_bytes()).hexdigest()==contract['prompt_code_sha256']
    tokenizer=AutoTokenizer.from_pretrained(args.adapter)
    data=[runtime_record(encode_scoring(as_scoring(r,False),tokenizer,contract['max_length']),r) for r in validation]
    model=PeftModel.from_pretrained(load_base(contract['model'],contract['revision']),args.adapter).eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evaluate(model,data,CandidateCollator(tokenizer.pad_token_id),args.output)


if __name__=='__main__':
    main()
