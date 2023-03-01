import math
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import transformers

import quant

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')

class QuantizeOnly:
    """Quantize only, no pruning."""
    def __init__(self, layer, amount_prune=0.0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quant.quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        if DEVICE.type != 'cpu':
            torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

class PruneMaskOnly(QuantizeOnly):
    """Prunes `amount_prune` of the weights based on second order info without reconstruction."""
    def __init__(self, layer, amount_prune):
        super().__init__(layer)
        self.amount_prune = amount_prune
        self.new_weight_with_mask = None

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float() # Shape: Out X In

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        M = torch.zeros_like(W) + 1
        E = (torch.zeros_like(W))[:, :blocksize]

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        print("Starting pruning: ", end="")

        for i in range(0, self.columns, blocksize):
            i2 = min(i + blocksize, self.columns)
            count = i2 - i
            assert count == blocksize, (
                count, blocksize, i, i2, self.layer.weight.data.shape)

            for ji in range(count):
                j = ji + i

                if ji == 0:
                    # Determine the weights for pruning mask selection for the next column
                    copy_linear = torch.nn.Linear(blocksize, W.shape[0]).to(W.device)
                    copy_linear.weight.data = W[:, j:j+blocksize] ** 2
                    copy_linear.weight.data /= torch.diag(Hinv).unsqueeze(0)[:, j:j+blocksize]

                    prune.l1_unstructured(copy_linear, name='weight', amount=self.amount_prune)
                    print(f"{j}:{j+blocksize}", end = ", ")
                    M[:, j:j+blocksize] = copy_linear.weight_mask

                E[:, j-i] = (1 - M[:, j]) * (W[:, j] / Hinv[j, j])
                W[:, j:i+blocksize] -= E[:, j-i].unsqueeze(1) * Hinv[j, j:i+blocksize].unsqueeze(0)

            W[:, i+blocksize:] -= E.matmul(Hinv[i:i+blocksize, i+blocksize:]) # Keep as is

        if DEVICE.type != "cpu":
            torch.cuda.synchronize()
        print('\nPrune time %.2f' % (time.time() - tick))
        self.new_weight_with_mask = (W, M)
        self.layer.weight.data = (self.layer.weight.data * M).to(self.layer.weight.data.dtype)

class PruneMaskReconstruction(PruneMaskOnly):
    """Prunes `amount_prune` of the weights based on second order info with reconstruction."""
    def __init__(self, layer, amount_prune):
        super().__init__(layer, amount_prune)

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1):
        super().fasterquant(blocksize, percdamp, groupsize)
        (W, M) = self.new_weight_with_mask
        self.layer.weight.data = (W * M.reshape(self.layer.weight.shape)).to(
            self.layer.weight.data.dtype)

class PruneMagnitudeMask(PruneMaskOnly):
    """Prunes `amount_prune` of the weights with lower magnitude in given layer."""
    def __init__(self, layer, amount_prune):
        super().__init__(layer, amount_prune)

    def add_batch(self, inp, out):
        """We don't need hessian for this."""
        return

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1):
        W = self.layer.weight.data.clone()
        copy_linear = torch.nn.Linear(W.shape[1], W.shape[0]).to(W.device)
        copy_linear.weight.data = W.clone()
        prune.l1_unstructured(copy_linear, name='weight', amount=self.amount_prune)

        self.new_weight_with_mask = (copy_linear.weight.data, copy_linear.weight_mask)
        self.layer.weight.data = (copy_linear.weight.data * copy_linear.weight_mask).to(
            self.layer.weight.data.dtype)