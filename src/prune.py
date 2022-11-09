import os
from pathlib import Path

import argh
import matplotlib
import pandas as pd
import torch
from torch.nn.utils import prune
from transformers import GPT2Model, RobertaModel, T5ForConditionalGeneration

MODEL_TYPE_TO_HF_STRING = {
    "gpt2": "gpt2", # gpt, gpt2-xl
    "roberta": "roberta-base", # roberta-base, roberta-large
    "t5": "t5-small", # t5-small, t5-3b
}

MODEL_TYPE_TO_HF_CLASS = {
    "gpt2": GPT2Model,
    "roberta": RobertaModel,
    "t5": T5ForConditionalGeneration,
}

#PRUNE_AMOUNTS = [0.0, 0.1, 0.5, 0.9, 0.95, 0.99]
PRUNE_AMOUNTS = [0.0, 0.1]

MODELFILE = Path('/tmp/model.h5')
ZIPFILE = Path('/tmp/model.h5.gz')

@argh.arg('model_type', choices=list(MODEL_TYPE_TO_HF_CLASS.keys()))
def main(model_type):
    hf_class = MODEL_TYPE_TO_HF_CLASS[model_type]
    hf_string = MODEL_TYPE_TO_HF_STRING[model_type]
    model = hf_class.from_pretrained(hf_string)

    uncompressed_mbs = []
    compressed_mbs = []

    for amount in PRUNE_AMOUNTS:
        print(f"Pruning {amount}...")
        if amount > 0.0:
            # It's safe to call this on an already-pruned model as long as amount is greater than before.
            # since we're selecting the smallest weights, the already-pruned weights will always be selected.
            model = prune_global_l1(model, amount)
        print(f"Writing to file...")
        torch.save(model.state_dict(), MODELFILE)
        uncompressed_mbs.append(round(MODELFILE.stat().st_size / 2**20))
        print(f"Compressing...")
        os.system(f"gzip -qf {MODELFILE}")
        compressed_mbs.append(round(ZIPFILE.stat().st_size / 2**20))
        ZIPFILE.unlink()
    df = pd.DataFrame({
        'pruned_frac': PRUNE_AMOUNTS,
        'uncompressed': uncompressed_mbs,
        'compressed': compressed_mbs,
    })
    df.plot(
        x='pruned_frac', y=['uncompressed', 'compressed'],
        title=f"{model_type} model size on disk", ylabel='Size on disk (MiBs)',
        xticks=df['pruned_frac'],
    ).get_figure().savefig(f"plots/{model_type}_disk.png")


def prune_global_l1(model, amount):
    modules = [module for module in model.modules() if hasattr(module, 'weight')]
    prune.global_unstructured(
        parameters=[(m, 'weight') for m in modules],
        pruning_method=prune.L1Unstructured,
        amount=amount)
    for module in modules:
        prune.remove(module, 'weight')
    return model


argh.dispatch_command(main)
