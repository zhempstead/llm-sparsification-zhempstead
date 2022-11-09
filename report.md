# Report

- [outline](https://oaklight.github.io/dls2022/assignments/lab%204%20sketch.pdf)

## 1

Sparsification is the process of converting a densely-connected model to a (relatively) sparse model. The goal is generally model compression - to produce a smaller model that performs inference more quickly.

Pruning is a particular type of sparsification where certain weights are removed or masked to 0 based on some heuristic, either during or after training.

Quantization is another way of performing model compression, by reducing the number of bits used to store each weight. This sacrifices precision.

Distillation is another model compression technique where a smaller model is trained to mimic the output layer of the larger model. Crucially, the smaller model is trying to learn the distribution of probabilities predicted by the teacher model - in a classification context, if there is an image of a husky we want the small model to not only learn that it should classify the image as "Husky" but also learn that the large model thinks that "Malamute" is also a fairly likely class for the image.

MoEfication (MoE stands for Mixture-of-Experts) is a technique for faster inference specific to the feedforward network portions of transformers. Rather than having one feedforward network, there are multiple smaller feedforward networks ("experts") and a router (with learned parameters) which selects suitable experts for each input. This doesn't make the model smaller but it allows the forward computation to be faster as we only have to compute the activations for the chosen experts.

## 2

Encoder-only: RoBERTa (this is only 355M parameters but it was the largest encoder-only model supported by Huggingface)
Decoder-only: GPT2 (the largest version of the model is 1.5B parameters)
Encoder-Decoder: T5-3B (3B parameters, as the name suggests)

## 3

Fraction of parameters overall whose absolute values are greater than 0.1 and 1 respectively:
- GPT2: 1.842e-2, 2.400e-6
- RoBERTa: 9.441e-2, 6.754e-8
- T5: 5.346e-1, 2.357e-2

I am surprised that T5 has such large weights. I also produced some plots that measure these fractions by layer. Here's an example from GPT2:
![gpt2_gt01](./plots/by_layer/gpt2_block_gt01.png)

In this plot, the y-axis is the fraction of weights with absolute values > 0.1, and the x axis is the layer (or block). Each line represents one type of weight. 'encoder' is 0 for weights from decoder layers and 1 for weights from encoder layers. 'bias' is 1 for weights from bias terms and 0 for all other weights. 'mlp' is 1 for weights that are part of the feedforward networks, and 0 for weights that are part of the attention blocks (including the linear projections).

For instance, the blue line represents weights from the non-bias (since bias is 0) attention portion (since mlp is 0) of decoder layers (since encoder=0.0; in gpt2, of course, all layers are decoder layers!). The red line, on the other hand, is weights from the bias terms in the feedforward portion of decoder layers.

In addition to plots of 'gt01' and 'gt1' (the fraction of weights whose absolute values are >= 1), I also produced plots for basic summary statistics of the distributions of weights: mean, standard deviation, skewness, and excess kurtosis, also by layer. Here's the standard deviation plot for gpt2:

![gpt2_std](./plots/by_layer/gpt2_block_std.png)

3. Devise approaches to assess sparsity structure in your choice of models and answer these questiosn:
   - what fraction of parameters >> 0? overall? by layer?
   - how does this vary by layer?

4. Produce sparsified versions of your models at 10%, 50%, 90%, 95%, 99%, by either coding your methods or using existing tools provided below
   Explain the nature of your methods, regardless of whether you code it yourselves.

5. Find 2 common benchmarks used by your models, by reviewing their publications. \
   Set them up and obtain baseline results of original models. \
   Compare performance of your sparsified versions with the baselines.
   Include plots and explanations.

6. Compare size of models and runtime for sparsified models. Include plots and explanations.

7. Explain the challenges of sparsification on LLMs.

## submission:
1. Due: Nov 9th, 12 PM CST
2. Fork your public Github repository, change the repo name to `llm-sparsification-<cnetid>`
3. we will look out for the following files:
   - `report.md`
   - `src/*`
   - `requirements.txt` for `pip` or `environment.yml` for `conda`
   - any jupyter notebooks

## resources to choose from
- [Pytorch Prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [TensorFlow Pruning](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide)
- [OpenBMB - BMCook](https://github.com/OpenBMB/BMCook)
- [airaria - TextPruner](https://github.com/airaria/TextPruner)
- [airaria - TextBrewer](https://github.com/airaria/TextBrewer)
- ...
