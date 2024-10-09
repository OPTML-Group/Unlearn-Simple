from metrics.verbmem import eval as eval_verbmem
from metrics.privleak import eval as eval_privleak
from metrics.knowmem import eval as eval_knowmem
from utils import load_model, load_tokenizer, write_csv, read_json, write_json
from constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN

import os
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Dict, Literal
from pandas import DataFrame


def eval_model(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer = LLAMA_DIR,
    metrics: List[str] = SUPPORTED_METRICS,
    corpus: Literal['news', 'books'] | None = None,
    privleak_auc_key: str = 'forget_holdout_Min-40%',
    verbmem_agg_key: str = 'mean_rougeL',
    verbmem_max_new_tokens: int = 128,
    knowmem_agg_key: str = 'mean_rougeL',
    knowmem_max_new_tokens: int = 32,
    verbmem_forget_file: str | None = None,
    privleak_forget_file: str | None = None,
    privleak_retain_file: str | None = None,
    privleak_holdout_file: str | None = None,
    knowmem_forget_qa_file: str | None = None,
    knowmem_forget_qa_icl_file: str | None = None,
    knowmem_retain_qa_file: str | None = None,
    knowmem_retain_qa_icl_file: str | None = None,
    temp_dir: str | None = None,
) -> Dict[str, float]:
    # Argument sanity check
    if not metrics:
        raise ValueError(f"Specify `metrics` to be a non-empty list.")
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Given metric {metric} is not supported.")
    if corpus is not None and corpus not in CORPORA:
        raise ValueError(f"Invalid corpus. `corpus` should be either 'news' or 'books'.")
    if corpus is not None:
        verbmem_forget_file = DEFAULT_DATA[corpus]['verbmem_forget_file'] if verbmem_forget_file is None else verbmem_forget_file
        privleak_forget_file = DEFAULT_DATA[corpus]['privleak_forget_file'] if privleak_forget_file is None else privleak_forget_file
        privleak_retain_file = DEFAULT_DATA[corpus]['privleak_retain_file'] if privleak_retain_file is None else privleak_retain_file
        privleak_holdout_file = DEFAULT_DATA[corpus]['privleak_holdout_file'] if privleak_holdout_file is None else privleak_holdout_file
        knowmem_forget_qa_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_file'] if knowmem_forget_qa_file is None else knowmem_forget_qa_file
        knowmem_forget_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_icl_file'] if knowmem_forget_qa_icl_file is None else knowmem_forget_qa_icl_file
        knowmem_retain_qa_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_file'] if knowmem_retain_qa_file is None else knowmem_retain_qa_file
        knowmem_retain_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_icl_file'] if knowmem_retain_qa_icl_file is None else knowmem_retain_qa_icl_file

    out = {}
    model = model.to('cuda')
     
    # 1. verbmem_f
    if 'verbmem_f' in metrics:
        data = read_json(verbmem_forget_file)
        agg, log = eval_verbmem(
            prompts=[d['prompt'] for d in data],
            gts=[d['gt'] for d in data],
            model=model, tokenizer=tokenizer,
            max_new_tokens=verbmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "verbmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "verbmem_f/log.json"))
        out['verbmem_f'] = agg[verbmem_agg_key] * 100

    # 2. privleak
    if 'privleak' in metrics:
        auc, log = eval_privleak(
            forget_data=read_json(privleak_forget_file),
            retain_data=read_json(privleak_retain_file),
            holdout_data=read_json(privleak_holdout_file),
            model=model, tokenizer=tokenizer
        )
        if temp_dir is not None:
            write_json(auc, os.path.join(temp_dir, "privleak/auc.json"))
            write_json(log, os.path.join(temp_dir, "privleak/log.json"))
        out['privleak'] = (auc[privleak_auc_key] - AUC_RETRAIN[corpus][privleak_auc_key]) / AUC_RETRAIN[corpus][privleak_auc_key] * 100

    # 3. knowmem_f
    if 'knowmem_f' in metrics:
        qa = read_json(knowmem_forget_qa_file)
        icl = read_json(knowmem_forget_qa_icl_file)
        agg, log = eval_knowmem(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "knowmem_f/log.json"))
        out['knowmem_f'] = agg[knowmem_agg_key] * 100

    # 4. knowmem_r
    if 'knowmem_r' in metrics:
        qa = read_json(knowmem_retain_qa_file)
        icl = read_json(knowmem_retain_qa_icl_file)
        agg, log = eval_knowmem(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_r/agg.json"))
            write_json(log, os.path.join(temp_dir, "knowmem_r/log.json"))
        out['knowmem_r'] = agg[knowmem_agg_key] * 100

    return out


def load_then_eval_models(
    model_dirs: List[str],
    names: List[str],
    corpus: Literal['news', 'books'],
    tokenizer_dir: str = LLAMA_DIR,
    out_file: str | None = None,
    metrics: List[str] = SUPPORTED_METRICS,
    temp_dir: str = "temp"
) -> DataFrame:
    print(out_file)
    # Argument sanity check
    if not model_dirs:
        raise ValueError(f"`model_dirs` should be non-empty.")
    if len(model_dirs) != len(names):
        raise ValueError(f"`model_dirs` and `names` should equal in length.")
    if out_file is not None and not out_file.endswith('.csv'):
        raise ValueError(f"The file extension of `out_file` should be '.csv'.")

    # Run evaluation
    out = []
    for model_dir, name in zip(model_dirs, names):
        model = load_model(model_dir)
        tokenizer = load_tokenizer(tokenizer_dir)
        res = eval_model(
            model, tokenizer, metrics, corpus,
            temp_dir=os.path.join(temp_dir, name)
        )
        out.append({'name': name} | res)
        if out_file is not None: write_csv(out, out_file)
        # DataFrame(out).to_csv(out_file, index=False)
    return DataFrame(out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--names', type=str, nargs='+', default=[])
    parser.add_argument('--tokenizer_dir', type=str, default=LLAMA_DIR)
    parser.add_argument('--corpus', type=str, required=True, choices=CORPORA)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--metrics', type=str, nargs='+', default=SUPPORTED_METRICS)
    args = parser.parse_args()

    load_then_eval_models(**vars(args))