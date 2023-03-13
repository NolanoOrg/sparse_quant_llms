# Sparse + Quant LLMs [WIP]

## Do not use, unless you really know what you are doing. Broken in multiple places.



Based on Sparse GPT, GPTQ, Optimal Bert Surgeon and others.

run opt.py to get the optimal sparsity for each layer.

`python3 opt.py facebook/opt-X [--seed 0] [--nsamples 128] [--wbits 16] [--groupsize -1] [--save PATH_TO_SAVE] [--compression_type {quantizeonly, prunemaskonly, prunemaskreconstruction, prunemagnitudemask, quantizeprune, none}] [--amount_prune 0.5]`

## Requirements:

torch == 1.13.1
transformers == 4.21.2
sentencepiece == 0.1.97

