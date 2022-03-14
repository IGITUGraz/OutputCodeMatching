from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import operator
from models.quantization import quan_Conv2d, quan_Linear
cudnn.benchmark = True

## This script is adapted from the following public repository:
## https://github.com/adnansirajrakin/T-BFA


def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    Note that, the conversion is different depends on number of bit used.
    '''
    output = input.clone()
    if num_bits == 1:
        output = output/2 + .5
    elif num_bits > 1:
        output[input.lt(0)] = 2 ** num_bits + output[input.lt(0)]

    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    if num_bits == 1:
        output = input * 2 - 1
    elif num_bits > 1:
        mask = 2 ** (num_bits - 1) - 1
        output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            m.__reset_stepsize__()
            m.__reset_weight__()

    # convert weights
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).char()
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return


class SneakyBFA(object):
    def __init__(self, criterion, C, k_top=10):
        self.criterion = criterion
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.C = C
        self.random_flip_flag = False

    def flip_bit(self, m, offs):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        self.k_top = m.weight.grad.detach().abs().view(-1).size()[0]
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(self.k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() + 1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits, 1) & m.b_w.abs().repeat(1, self.k_top).short()) \
                     // m.b_w.abs().repeat(1, self.k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        grad_mask = (torch.ones(grad_mask.size()).short().cuda() - grad_mask).short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            self.random_flip_flag = True
            print('Random flip flag turned on - will not continue trying further')
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) ^ w_bin_topk
        weight_changed = w_bin_topk - w_bin_topk_flipped
        idx = (weight_changed != 0).nonzero()  ## index of the weight changed

        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin, m.N_bits).view(m.weight.data.size()).float()
        offse = (w_idx_topk[idx])
        return param_flipped, offse

    def progressive_bit_search(self, model, data, target, data1, target1, args):
        if args.ocm:
            target = torch.Tensor(torch.vstack([self.C[int(i.item()), :] for i in target[:]]).to('cpu')).cuda()
            target1 = torch.Tensor(torch.vstack([self.C[int(i.item()), :] for i in target1[:]]).to('cpu')).cuda()
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data.cuda())
        self.loss = self.criterion(output, target.cuda())
        output1 = model(data1.cuda())
        self.loss += self.criterion(output1, target1.cuda()).item()

        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        for j in range(1):
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n, offs = 0, 0
            for name, module in model.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(module, quan_Linear):
                    n = n + 1
                    if n < 220:
                        clean_weight = module.weight.data.detach()
                        attack_weight, _ = self.flip_bit(module, offs)
                        # change the weight to attacked weight and get loss
                        module.weight.data = attack_weight
                        output = model(data.cuda())
                        self.loss_dict[name] = self.criterion(output, target.cuda()).item()
                        output1 = model(data1.cuda())
                        xx = self.criterion(output1, target1.cuda()).item()
                        # print(xx, self.loss_dict[name])
                        self.loss_dict[name] += xx
                        # change the weight back to the clean weight
                        module.weight.data = clean_weight
                    w = module.weight.size()
                    offs += w[0] * w[1] * w[2] * w[3] if len(w) == 4 else w[0] * w[1]
            max_loss_module = min(self.loss_dict.items(), key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        n, offs = 0, 0
        for name, module in model.named_modules():
            if isinstance(module, quan_Conv2d) or isinstance(module, quan_Linear):
                n = n + 1
                if name == max_loss_module:
                    attack_weight, offset = self.flip_bit(module, offs)
                    module.weight.data = attack_weight
                    print(n, offset)
                w = module.weight.size()
                offs += w[0] * w[1] * w[2] * w[3] if len(w) == 4 else w[0] * w[1]

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return n, offset
