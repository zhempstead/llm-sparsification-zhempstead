# llm-sparsification-release

- [outline](https://oaklight.github.io/dls2022/assignments/lab%204%20sketch.pdf)

## task

1. Understand and distinguish these concepts: 
   - Sparsification
   - Pruning
   - Quantization
   - Distillation
   - MoEfication

2. Choose your models. *Pick 3 models, 1 from each category.* Each pick should be of more than 1B parameters before pruning.
   - Encoder-only
   - Decoder-only
   - Encoder-Decoder

   You can find info about model size at https://openbmb.github.io/BMList/list/. You may use huggingface or other modelhub that you see fit.

3. Devise approaches to assess sparsity structure in your choice of models and answer these questiosn:
   - what fraction of parameters >> 0? overall? by layer?
   - how does this vary by layer?

4. Produce sparsified version of your models at 10%, 50%, 90%, 95%, 99%, by either coding your methods or using existing tools provided below
   Explain the nature of your methods, regardless of whether you code it yourselves.

5. Find 2 common benchmarks used by your models, by reviewing their publications. \
   Set them up and obtain baseline results of unpruned models. \
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
- [TextPruner](https://github.com/airaria/TextPruner)
- [TextBrewer](https://github.com/airaria/TextBrewer)
- [OpenBMB - BMCook](https://github.com/OpenBMB/BMCook)
- ...