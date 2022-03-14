import argparse
import copy
from bitstring import Bits
import datasets
import models
from utils import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss

## This script is adapted from the following public repository:
## https://github.com/jiawangbai/TA-LBF

parser = argparse.ArgumentParser(description='Stealthy TA-LBF on DNNs')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--ocm', action='store_true', help='output layer coding with bit strings')
parser.add_argument('--randcode', action="store_true", help='enable random output code matching')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (either linear and tanh is supported)')
parser.add_argument('--code_length', '-cl', default=16, type=int, help='length of codewords')
parser.add_argument('--outdir', type=str, default='results/cifar10/resnet20_quan8_OCM64/', help='folder where the model is saved')
parser.add_argument('--batch', '-b', default=128, type=int, metavar='N', help='batchsize (default: 128)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--attack_info', type=str, default='cifar10_talbf.txt', help='attack info list')
parser.add_argument('--init-k', '-init_k', default=5, type=float)
parser.add_argument('--init-lam', '-init_lam', default=100, type=float)
parser.add_argument('--max-search-k', '-max_search_k', default=6, type=int)
parser.add_argument('--max-search-lam', '-max_search_lam', default=8, type=int)
parser.add_argument('--n_aux', type=int, default=64, help='number of auxiliary samples')
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=50, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=50, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=5, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr', '-inn_lr', default=0.01, type=float)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=2000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
gpu_list = [int(i) for i in args.gpu.strip().split(",")] if args.gpu is not "0" else [0]
if args.gpu == "-1":
    device = torch.device('cpu')
    print('Using cpu')
else:
    device = torch.device('cuda')
    print('Using gpu: ' + args.gpu)


class AugLag(nn.Module):
    def __init__(self, n_bits, w, b, step_size, args, C):
        super(AugLag, self).__init__()
        self.n_bits = n_bits
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=True)
        self.w_twos = nn.Parameter(torch.zeros([w.shape[0], w.shape[1], self.n_bits]), requires_grad=True)
        self.step_size = step_size
        self.w = w
        self.args = args
        self.C = C
        self.output_act = nn.Tanh() if args.output_act == 'tanh' else None

        self.reset_w_twos()
        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

    def forward(self, x):
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size
        x = F.linear(x, w, self.b)
        if self.args.ocm:
            x = nn.Sigmoid()(2 * x)         # scale to [0, 1]
        else:
            x = self.output_act(x) if self.output_act is not None else x
        return x

    def predict(self, x):
        x = self.forward(x)
        x = 2 * x - 1 if self.args.ocm else x       # rescale OCM output back to [-1, 1] for our natural way of prediction
        x = F.softmax(torch.log(F.relu(torch.matmul(x, self.C.T)) + 1e-6)) if self.args.ocm else F.softmax(x)
        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] += torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])


def project_box(x):
    xp = x
    xp[x > 1] = 1
    xp[x < 0] = 0
    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec
    return xp


def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp


def loss_func(output, labels, s, t, lam, w_twos, b_ori, k_bits, y1, y2, y3, z1, z2, z3, rho1, rho2, rho3, C):
    if args.ocm:
        # applying the tanh to sigmoid trick to be able to compute BCELoss and eventually attack OCM models via TA-LBF
        C_shift = (C + 1) / 2
        output = torch.nan_to_num(output)
        l1 = torch.nn.BCELoss()(output[-1], C_shift[t])
        l2 = torch.nn.BCELoss()(output[:-1], C_shift[labels[:-1]])
    else:
        l1 = - torch.log(torch.nn.Softmax()(output[-1]))[t]
        l2 = CrossEntropyLoss()(output[:-1], labels[:-1])
    b_ori = torch.tensor(b_ori).float().cuda()
    b = w_twos.view(-1)

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(), \
                             torch.tensor(y3).float().cuda(), torch.tensor(z1).float().cuda(), \
                             torch.tensor(z2).float().cuda(), torch.tensor(z3).float().cuda()

    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k_bits + y3)

    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 + \
         (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return l1 + lam * l2 + l3 + l4


def attack(auglag_ori, all_data, labels, labels_cuda, target_idx, target_class, source_class, aux_idx, lam, k_bits, args):
    n_aux = args.n_aux
    lam = lam
    ext_max_iters = args.ext_max_iters
    inn_max_iters = args.inn_max_iters
    initial_rho1 = args.initial_rho1
    initial_rho2 = args.initial_rho2
    initial_rho3 = args.initial_rho3
    max_rho1 = args.max_rho1
    max_rho2 = args.max_rho2
    max_rho3 = args.max_rho3
    rho_fact = args.rho_fact
    inn_lr = args.inn_lr

    all_idx = np.append(aux_idx, target_idx)
    auglag = copy.deepcopy(auglag_ori)

    b_ori = auglag.w_twos.data.view(-1).detach().cpu().numpy()
    b_new = b_ori

    y1, y2, y3 = b_ori, b_ori, 0
    z1, z2, z3 = np.zeros_like(y1), np.zeros_like(y1), 0
    rho1, rho2, rho3 = initial_rho1, initial_rho2, initial_rho3

    stop_flag = False
    for ext_iter in range(ext_max_iters):
        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, p=2)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):
            input_var = torch.autograd.Variable(all_data[all_idx], volatile=True)
            target_var = torch.autograd.Variable(labels_cuda[all_idx].long(), volatile=True)

            output = auglag(input_var)
            loss = loss_func(output, target_var, source_class, target_class, lam, auglag.w_twos,
                             b_ori, k_bits, y1, y2, y3, z1, z2, z3, rho1, rho2, rho3, auglag.C)

            loss.backward(retain_graph=True)
            auglag.w_twos.data = auglag.w_twos.data - inn_lr * auglag.w_twos.grad.data
            auglag.w_twos.grad.zero_()

        b_new = auglag.w_twos.data.view(-1).detach().cpu().numpy()

        if True in np.isnan(b_new):
            return -1

        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)

        if max(temp1, temp2) <= 1e-4 and ext_iter > 100:
            print('END iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss.item()))
            stop_flag = True
            break

    auglag.w_twos.data[auglag.w_twos.data > 0.5] = 1.0
    auglag.w_twos.data[auglag.w_twos.data < 0.5] = 0.0

    output = auglag.predict(all_data)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)

    expose_list = [i for i in range(len(output)) if labels[i].to('cpu') == pred[i].to('cpu') and i != target_idx and i not in aux_idx]
    pa_acc = len(expose_list) / (len(labels) - 1 - n_aux)
    n_bit = torch.norm(auglag_ori.w_twos.data.view(-1) - auglag.w_twos.data.view(-1), p=0).item()
    ret = {"pa_acc": pa_acc, "stop": stop_flag, "suc": target_class == pred[target_idx].item(), "n_bit": n_bit}

    return ret


def load_data(model, test_loader, args, C):
    mid_out, labels = np.zeros([len(test_loader.dataset), model.mid_dim]), np.zeros([len(test_loader.dataset)])
    start = 0
    model.eval()
    for i, (input, target) in enumerate(test_loader):
        if C is not None:
            target = torch.tensor([torch.where(torch.all(C.to('cpu') == target[i], dim=1))[0][0] for i in range(target.shape[0])])
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        output = model(input_var)
        mid_out[start: start + args.batch] = output.detach().cpu().numpy()
        labels[start: start + args.batch] = target.numpy()
        start += args.batch
    mid_out = torch.tensor(mid_out).float().cuda()
    labels = torch.tensor(labels).float()
    return mid_out, labels


def load_model(args, DATASET):
    n_output = args.code_length if args.ocm else args.num_classes
    C = torch.tensor(DATASET.C).to(device) if args.ocm else None

    # Evaluate clean accuracy
    model = models.__dict__[args.arch + '_mid'](n_output, args.bits)
    model = nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict'])
    weight_conversion(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    weight = model.linear.weight.data.detach().cpu().numpy()
    bias = model.linear.bias.data.detach().cpu().numpy()
    step_size = np.float32(model.linear.step_size.detach().cpu().numpy())

    return weight, bias, step_size, model, C


def main():
    # Load dataset
    DATASET = datasets.__dict__[args.dataset](args)
    _, test_loader = DATASET.loaders()

    weight, bias, step_size, model, C = load_model(args, DATASET)
    mid_out, labels = load_data(model, test_loader, args, C)
    labels_cuda = labels.cuda()
    
    auglag = AugLag(args.bits, weight, bias, step_size, args, C).cuda()
    clean_output = auglag.predict(mid_out)
    _, pred = clean_output.cpu().topk(1, 1, True, True)
    corrects = [i for i in range(len(pred.squeeze(1))) if labels[i] == pred.squeeze(1)[i]]
    acc_ori = len([i for i in range(len(pred.squeeze(1))) if labels[i] == pred.squeeze(1)[i]]) / len(labels)
    print('Original ACC: ', acc_ori)

    print("Attack Start")
    attack_info = np.loadtxt(args.attack_info).astype(int)
    asr, pa_acc, n_bit, n_stop, param_lam, param_k_bits = [], [], [], [], [], []
    for i, (target_class, attack_idx) in enumerate(attack_info):
        print('Target class: ', target_class)
        print('Attack idx: ', attack_idx)
        source_class = int(labels[attack_idx])
        aux_idx = np.random.choice([i for i in range(len(labels)) if i != attack_idx], args.n_aux, replace=False)

        suc = False
        cur_k = args.init_k
        for search_k in range(args.max_search_k):
            cur_lam = args.init_lam
            for search_lam in range(args.max_search_lam):
                print('k: ', str(cur_k), 'lambda: ', str(cur_lam))
                res = attack(auglag, mid_out, labels, labels_cuda, attack_idx,
                             target_class, source_class, aux_idx, cur_lam, cur_k, args)

                if res == -1:
                    print("Error[{0}]: Lambda:{1} K_bits:{2}".format(i, cur_lam, cur_k))
                    cur_lam = cur_lam / 2.0
                    continue
                elif res["suc"]:
                    n_stop.append(int(res["stop"]))
                    asr.append(int(res["suc"]))
                    pa_acc.append(res["pa_acc"])
                    n_bit.append(res["n_bit"])
                    param_lam.append(cur_lam)
                    param_k_bits.append(cur_k)
                    suc = True
                    break
                cur_lam = cur_lam / 2.0
            if suc:
                break
            cur_k = cur_k * 2.0

        if not suc:
            asr.append(0)
            n_stop.append(0)
            print("[{0}] Fail!".format(i))
        else:
            print("[{0}] PA-ACC:{1:.4f} Success:{2} N_flip:{3} Stop:{4} Lambda:{5} K:{6}".format(
                i, pa_acc[-1]*100, bool(asr[-1]), n_bit[-1], bool(n_stop[-1]), param_lam[-1], param_k_bits[-1]))

        if (i+1) % 10 == 0:
            print("END[0] PA-ACC:{1:.4f} ASR:{2} N_flip:{3:.4f}".format(
                i, np.mean(pa_acc)*100, np.mean(asr)*100, np.mean(n_bit)))

    print("END Original_ACC:{0:.4f} PA_ACC:{1:.4f} ASR:{2:.2f} N_flip:{3:.4f}".format(
            acc_ori*100, np.mean(pa_acc)*100, np.mean(asr)*100, np.mean(n_bit)))


if __name__ == '__main__':
    main()
