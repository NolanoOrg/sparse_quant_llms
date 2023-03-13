"""Compress LLaMa models."""
import os
import random
import time
import json
import copy

import torch
import torch.nn as nn
from datasets import load_dataset
import llama_model
import llama_tokenizer

import smart_compressors
import quant
from pathlib import Path

DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0') # pylint: disable=no-member

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''): # pylint: disable=dangerous-default-value
    """Find linear and conv layers in a model."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_wikitext2(nsamples, seed, seqlen, tokenizer_path):
    """For now we take nsamples datapoints from wikitext2 and tokenize them."""
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = llama_tokenizer.Tokenizer(tokenizer_path)
    trainenc = tokenizer.encode("\n\n".join(traindata['text']), bos=False, eos=False)
    testenc = tokenizer.encode("\n\n".join(testdata['text']), bos=False, eos=False)
    bos_id = tokenizer.sp_model.bos_id()

    random.seed(seed)
    trainloader = []
    print('\n\n\n', len(trainenc), '\n\n', len(trainenc) - seqlen - 1)
    for _ in range(nsamples):
        # print('\n\n\n', '-----', '\n\n\n')
        i = random.randint(0, len(trainenc) - seqlen - 1)
        j = i + seqlen - 1
        inp = [bos_id] + trainenc[i:j]
        tar = copy.deepcopy(inp)
        tar = [-100] * (len(tar) - 1) + tar[-1:]
        trainloader.append((torch.LongTensor([inp]), torch.LongTensor([tar])))
    _testloader = []
    print('\n\n\n', len(testenc), '\n\n', len(testenc) - seqlen - 1)
    for _ in range(nsamples):
        # print('\n\n\n', '-----', '\n\n\n')
        i = random.randint(0, len(testenc) - seqlen - 1)
        j = i + seqlen - 1
        inp = [bos_id] + testenc[i:j]
        tar = copy.deepcopy(inp)
        tar = [-100] * (len(tar) - 1) + tar[-1:]
        _testloader.append((torch.LongTensor([inp]), torch.LongTensor([tar])))
    return trainloader, _testloader

def benchmark(model_to_be_benched, _dataloader):
    """Benchmark a model."""
    current_device = next(model.parameters()).device
    model_to_be_benched = model_to_be_benched.to(DEVICE)
    data_iterator = iter(_dataloader)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss = 0.0
        for i in range(100):
            inputs = next(data_iterator)
            # print('adsas:', inputs, 'iter:', _dataloader)
            inputs = inputs[0].to(DEVICE)
            outputs = model_to_be_benched(inputs[:, :-1])
            loss += loss_fn(outputs.permute(0, 2, 1), inputs[:, 1:]).item()
            if i % 10 == 5:
                print(i)
    model_to_be_benched = model_to_be_benched.to(current_device)
    return loss

def get_model(model_dir):
    """Get llama model."""
    def skip(*args, **kwargs): # pylint: disable=unused-argument, redefined-outer-name
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    with open(Path(model_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    checkpoint = torch.load(model_dir + "/consolidated.00.pth", map_location="cpu")
    if "consolidated.01.pth" in os.listdir(model_dir):
        checkpoint_2 = torch.load(model_dir + "consolidated.01.pth", map_location="cpu")
        # for every key concatenate the tensors in the two checkpoints
        i = 0
        for key in checkpoint.keys():
            if 'norm.weight' in key:
                continue
            dim = 1 if any(x in key for x in [
                '.feed_forward.w2.weight', '.attention.wo.weight', 'tok_embeddings.weight']) else 0
            checkpoint[key] = torch.cat((checkpoint[key], checkpoint_2[key]), dim=dim)
            if i <= 5:
                print(key, checkpoint[key].shape, checkpoint_2[key].shape)
                i += 1

    model_args = llama_model.ModelArgs(max_seq_len=2048, **params)
    model_loaded = llama_model.Transformer(model_args)
    model_loaded.seqlen = 2048 # We need this for the dataloader trimming.
    print(model_loaded.state_dict().keys())
    print(checkpoint.keys())
    model_loaded.load_state_dict({k: v for k, v in checkpoint.items() if 'rope.freqs' not in k})
    model_loaded = model_loaded.cuda()

    # If device is CPU then we convert from fp16 to fp32
    if DEVICE.type == 'cpu':
        model_loaded = model_loaded.half().to(torch.float32)
    print("existing")
    return model_loaded

@torch.no_grad()
def model_sequential(model, dataloader, device, compressor_class): # pylint: disable=redefined-outer-name
    """Optimize model sequentially."""
    print('Starting ...')
    device = "cuda"
    layers = model.layers

    # Transfer to device
    model = model.to(device)

    # Initialize inputs, cache
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.params.dim), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}
    print('\n\n\nseq\n\n\n')

    # Get input and attention mask after layer 0
    class Catcher(nn.Module): # pylint: disable=missing-class-docstring
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, mask, **kwargs):
            """Forward pass."""
#             print('Catcher input:', inp)
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = mask
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Transfer back to CPU
    model = model.cpu()
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps) # Store outputs after each layer # pylint: disable=no-member
    attention_mask = cache['attention_mask']
    print('Ready.')

    all_compressors = {} # pylint: disable=redefined-outer-name
    for i in range(len(layers)): # pylint: disable=consider-using-enumerate
        layer = layers[i].to(device)

        # Find linear layers and initialize quantizer for it
        subset = find_layers(layer)
        # print(subset)
        single_layer_compressor = {}
        for name in subset: # pylint: disable=consider-using-dict-items
            single_layer_compressor[name] = compressor_class(subset[name], args.amount_prune)
            single_layer_compressor[name].quantizer = quant.Quantizer()
            single_layer_compressor[name].quantizer.configure(
                args.wbits, perchannel=True, sym=False, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                single_layer_compressor[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset: # pylint: disable=consider-using-dict-items
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            # print('inps:', inps[j], inps[j].shape)
            outs[j] = layer(inps[j].unsqueeze(0), mask=attention_mask)[0]
        for hhh in handles:
            hhh.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            single_layer_compressor[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize)
            # all_compressors[
            #     'model.decoder.layers.%d.%s' % (i, name)] = single_layer_compressor[name] # pylint: disable=consider-using-f-string
            single_layer_compressor[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del single_layer_compressor
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    return all_compressors


if __name__ == '__main__':
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='LLaMa model path to load;')
    parser.add_argument('vocab', type=str, help='LLaMa vocab path to load;')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 16],
                        help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--groupsize', type=int, default=-1,
                        help='Groupsize to use for quantization/pruning; default uses full row.')
    parser.add_argument('--do_save', action='store_true', default=False,
                        help='Whether to save or not')
    parser.add_argument('--savepath', type=str, default='',
                        help='Save quantized/pruned checkpoint under this name.')
    parser.add_argument('--load', type=str, default='',
                        help='Load quantized/pruned checkpoint under this name.')
    parser.add_argument('--compression_type', type=str, required=True,
                        choices=['quantizeonly', 'prunemaskonly', 'prunemaskreconstruction',
                                 'prunemagnitudemask', 'quantizeprune', 'none',# 'pruneonly'
                                 ],
                        help='Type of compression to perform.')
    parser.add_argument('--amount_prune', type=float, default=0.5,
                        help='Amount of pruning to perform.')
    args = parser.parse_args()
    if args.savepath != '':
        args.do_save = True
    if args.compression_type == 'none':
        args.do_save = None

    # If prune is to be done then args.amount_prune must be between 0 and 1
    if args.compression_type in ['pruneonly', 'quantizeprune', 'prunemaskonly',
                                 'prunemaskreconstruction']:
        assert 0 <= args.amount_prune <= 1, 'Amount of pruning must be between 0 and 1'

    # Load model
    model = get_model(args.model)
    model.eval()
    if args.load:
        model.load_state_dict(torch.load(args.load))
    if args.compression_type != 'quantizeonly':
        args.wbits = 16

    # Load data
    dataloader, testloader = get_wikitext2(
        nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer_path=args.vocab)

    # Perform compression
    if args.compression_type != None:
        compression_class = None # pylint: disable=invalid-name
        if args.compression_type == 'quantizeonly':
            assert args.wbits < 16, 'Quantize only works with 4-bit quantization'
            compression_class = smart_compressors.QuantizeOnly
        elif args.compression_type == 'prunemaskonly':
            compression_class = smart_compressors.PruneMaskOnly
        elif args.compression_type == 'prunemaskreconstruction':
            compression_class = smart_compressors.PruneMaskReconstruction
        elif args.compression_type == 'prunemagnitudemask':
            compression_class = smart_compressors.PruneMagnitudeMask
        elif args.compression_type == 'none':
            pass
        elif args.compression_type == 'quantizeprune':
            raise NotImplementedError
        else:
            raise ValueError('Unknown compression type: %s' % args.compression_type)

        if compression_class is not None:
            tick = time.time()
            computed_compressors = model_sequential(model, dataloader, DEVICE, compression_class)
            print("Total time taken: %.2f seconds" % (time.time() - tick)) # pylint: disable=consider-using-f-string

    savefolder = None
    if args.do_save:
        if args.savepath == '':
            raise ValueError('Must specify savepath if do_save is True')
        savefolder = os.path.join(
            os.path.dirname(args.savepath),
            f'Model-{args.model.replace("/", "_")}_Compression-{args.compression_type}_Prune-{args.amount_prune}_Bits-{args.wbits}_Group-{args.groupsize}.pth')

    # Save
    if args.do_save:
        torch.save(model.state_dict(), savefolder)

    # Do benchmark
    if args.compression_type in ["quantizeonly", "prunemaskonly", "prunemaskreconstruction",
                                 "none"]:
        model = model.to(DEVICE)
        score = benchmark(model, testloader)
        print(score, savefolder)
        if args.do_save:
            open(savefolder + ".score", 'w+').write(str(score) + '\n')
            print(score, savefolder)

    print("Done")
    print("\n" * 5)
