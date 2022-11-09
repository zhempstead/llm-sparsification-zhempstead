import os
from pathlib import Path
import timeit

import argh
import matplotlib
import pandas as pd
import torch
from torch.nn.utils import prune
from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaForCausalLM, RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer

MODEL_TYPE_TO_HF_STRING = {
    "gpt2": "gpt2-xl", # gpt, gpt2-xl
    "roberta": "roberta-large", # roberta-base, roberta-large
    "t5": "t5-3b", # t5-small, t5-3b
}

MODEL_TYPE_TO_HF_CLASS = {
    "gpt2": GPT2LMHeadModel,
    "roberta": RobertaForCausalLM,
    "t5": T5ForConditionalGeneration,
}

MODEL_TYPE_TO_HF_TOKENIZER = {
    "gpt2": GPT2Tokenizer,
    "roberta": RobertaTokenizer,
    "t5": T5Tokenizer,
}

PRUNE_AMOUNTS = [0.0, 0.1, 0.5, 0.9, 0.95, 0.99]

ITERS = 10

MODELFILE = Path('/tmp/model.h5')
ZIPFILE = Path('/tmp/model.h5.gz')

@argh.arg('model_type', choices=list(MODEL_TYPE_TO_HF_CLASS.keys()))
def main(model_type):
    hf_class = MODEL_TYPE_TO_HF_CLASS[model_type]
    hf_string = MODEL_TYPE_TO_HF_STRING[model_type]
    hf_tokenizer = MODEL_TYPE_TO_HF_TOKENIZER[model_type]
    model = hf_class.from_pretrained(hf_string)
    tokenizer = hf_tokenizer.from_pretrained(hf_string)

    text = "Hello, world! Or should I say, hello, large language model?"
    tokens = tokenizer.encode(text, return_tensors='pt')
    #model.generate(tokenizer.encode(text

    times = []

    for amount in PRUNE_AMOUNTS:
        print(f"Pruning {amount}...")
        if amount > 0.0:
            # It's safe to call this on an already-pruned model as long as amount is greater than before.
            # since we're selecting the smallest weights, the already-pruned weights will always be selected.
            model = prune_global_l1(model, amount)
            
        these_times = []
        for i in range(ITERS):
            print(f"Iteration {i}...")
            start = timeit.default_timer()
            model.generate(tokens, max_new_tokens=1)
            end = timeit.default_timer()
            these_times.append(end - start)
        times.append(sum(these_times) / ITERS)

    df = pd.DataFrame({
        'pruned_frac': PRUNE_AMOUNTS,
        'times': times,
    })
    df.plot(
        x='pruned_frac', y='times',
        title=f"{model_type} runtimes", ylabel='Runtime (seconds)',
        xticks=df['pruned_frac'],
    ).get_figure().savefig(f"plots/{model_type}_runtimes.png")


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
