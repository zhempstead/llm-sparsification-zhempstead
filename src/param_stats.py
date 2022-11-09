import argh
import matplotlib
import pandas as pd
import torch
from transformers import GPT2Model, RobertaModel, T5ForConditionalGeneration

MODEL_TYPE_TO_HF_STRING = {
    "gpt2": "gpt2-xl", # gpt2-xl
    "roberta": "roberta-large", # roberta-large
    "t5": "t5-3b", # t5-3b
}

MODEL_TYPE_TO_HF_CLASS = {
    "gpt2": GPT2Model,
    "roberta": RobertaModel,
    "t5": T5ForConditionalGeneration,
}

@argh.arg('model_type', choices=list(MODEL_TYPE_TO_HF_CLASS.keys()))
def main(model_type):
    hf_class = MODEL_TYPE_TO_HF_CLASS[model_type]
    hf_string = MODEL_TYPE_TO_HF_STRING[model_type]
    model = hf_class.from_pretrained(hf_string)
    df = explore_weights(model, model_type)

    # Not weighted by # of weights
    df = df.groupby(['encoder', 'bias', 'mlp', 'block']).mean().reset_index()
    for metric in ['mean', 'std', 'skew', 'excess_kurtosis', 'gt01', 'gt1']:
        df.pivot(
            index='block', columns=['encoder', 'bias', 'mlp'], values=metric
        ).plot(
            title=f"{model_type}: {metric} over blocks by types of weight",
            logy=(metric.startswith('gt')),
        ).get_figure().savefig(
            f"plots/by_layer/{model_type}_block_{metric}.png")


def explore_weights(model, model_type):
    weights_total = 0
    weights_gt01 = 0
    weights_gt1 = 0

    layer_df = pd.DataFrame({col: [] for col in [
        'block', 'encoder', 'bias', 'mlp', 'mean', 'std', 'skew', 'excess_kurtosis', 'gt1', 'gt01',
    ]})
    for layer, tensor in model.named_parameters():
        nw = tensor.numel()

        mean = torch.mean(tensor)
        diffs = tensor - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skew = torch.mean(torch.pow(zscores, 3.0))
        excess_kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0

        gt1_raw = torch.sum(torch.abs(tensor) > 1)
        gt01_raw = torch.sum(torch.abs(tensor) > 0.1)
        gt1 = gt1_raw / nw
        gt01 = gt01_raw / nw

        weights_total += nw
        weights_gt01 += gt01_raw.item()
        weights_gt1 += gt1_raw.item()

        cats = categories(layer, model_type)
        if cats is None:
            continue
        cats.update({'mean': mean.item(), 'std': std.item(), 'skew': skew.item(), 'excess_kurtosis': excess_kurtosis.item(), 'gt1': gt1.item(), 'gt01': gt01.item()})
        layer_df = layer_df.append(cats, ignore_index=True)

    print(f"Fraction of weights w/abs value > 0.1: {weights_gt01/weights_total}")
    print(f"Fraction of weights w/abs value > 1: {weights_gt1/weights_total}")
    layer_df['block'] = layer_df['block'].astype(int)
    return layer_df

def categories(layer, model_type):
    split = layer.split('.')
    out = {}
    if model_type == "gpt2":
        if split[0] != 'h':
            return None
        out['block'] = split[1]
        out['encoder'] = False
        out['bias'] = "bias" in layer
        out['mlp'] = "mlp" in layer
    elif model_type == "t5":
        if split[0] not in ["encoder", "decoder"] or "final" in layer:
            return None
        out['block'] = split[2]
        out['encoder'] = split[0] == "encoder"
        out['bias'] = "bias" in layer
        out['mlp'] = "Relu" in layer
    elif model_type == "roberta":
        if split[0] != "encoder":
            return None
        out['encoder'] = True
        out['block'] = split[2]
        out['bias'] = "bias" in layer
        out['mlp'] = "dense" in layer
    return out

argh.dispatch_command(main)
