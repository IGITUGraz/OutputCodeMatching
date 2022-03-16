import argparse
import datasets
import models
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss
from torchvision import transforms
import torchvision
from xlwt import Workbook

## This script is adapted from the following public repository:
## https://github.com/adnansirajrakin/T-BFA

parser = argparse.ArgumentParser(description='Stealthy Targeted-BFA on DNNs')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--ocm', action='store_true', help='output layer coding with bit strings')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (only linear and tanh is supported)')
parser.add_argument('--code_length', '-cl', default=16, type=int, help='length of codewords')
parser.add_argument('--outdir', type=str, default='results/', help='folder where the model is saved')
parser.add_argument('--batch', '-b', default=128, type=int, metavar='N', help='Mini-batch size (default: 128)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--iters', type=int, default=5000, help='max attack iterations (def: 5000)')
parser.add_argument('--source_start', type=int, default=0, help='source_start')
parser.add_argument('--source_end', type=int, default=50, help='source_end')
parser.add_argument('--avgs', type=int, default=5, help='average of how many rounds')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not os.path.exists("tbfa_results/"):
    os.makedirs("tbfa_results/")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
gpu_list = [int(i) for i in args.gpu.strip().split(",")] if args.gpu is not "0" else [0]
if args.gpu == "-1":
    device = torch.device('cpu')
    print('Using cpu')
else:
    device = torch.device('cuda')
    print('Using gpu: ' + args.gpu)


def data_gen():
    if args.dataset == 'CIFAR10':
        tr_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'), train=False, transform=tr_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
        num_source, num_others = 1000, 9000
        im_size = 32
    elif args.dataset == 'CIFAR100':
        tr_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.267, 0.256, 0.276])])
        testset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'CIFAR100'), train=False, transform=tr_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
        num_source, num_others = 100, 9900
        im_size = 32
    elif args.dataset == 'ImageNet':
        tr_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        testset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'imagenet/validation'), tr_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        num_source, num_others = 50, 49950
        im_size = 224
    else:
        print('Dataset not implemented, will crash soon...')
        pass

    # dataT and targetT will contain only one class images whose image will be missclassified
    data, dataT = torch.zeros([num_others, 3, im_size, im_size]).cuda(), torch.zeros([num_source, 3, im_size, im_size]).cuda()
    target, targetT = torch.zeros([num_others]).long().cuda(), torch.zeros([num_source]).long().cuda()
    xs, xn = 0, 0
    for t, (x, y) in enumerate(test_loader):
        if t < num_others + num_source:
            if y != args.source:
                data[xs, :, :, :] = x[0, :, :, :]
                target[xs] = y.long()
                xs += 1
            if y == args.source:
                dataT[xn, :, :, :] = x[0, :, :, :]
                targetT[xn] = y.long()
                xn += 1

    data1, target1 = data[0:args.auxiliary, :, :, :], target[0:args.auxiliary]          # only separating validation samples
    data2, target2 = data[args.auxiliary:, :, :, :], target[args.auxiliary:]            # separating the rest
    dataT1, targetT1 = dataT[0:args.attacksamp, :, :, :], targetT[0:args.attacksamp]    # separating "to be attacked" test samples
    dataT2, targetT2 = dataT[num_source - args.attacksamp:num_source, :, :, :], targetT[num_source - args.attacksamp:num_source]  # separating the rest from source class

    return data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2


def validate(model, loader, C):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if args.ocm:
                probs = F.softmax(torch.log(F.relu(torch.matmul(model(data), C.T)) + 1e-6))
                labels = torch.tensor([torch.where(torch.all(C == target[i], dim=1))[0][0] for i in range(target.shape[0])])
                pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(labels.to(device).view_as(pred)).sum().item()
            else:
                output = nn.Softmax()(model(data))
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, loader.sampler.__len__(), 100. * correct / loader.sampler.__len__()))
    return 100. * correct / loader.sampler.__len__()


def validate_batchwise(model, input, label, C):
    """ this function computes the accuracy of a given input and label batchwise """
    model.eval()
    correct, n, m = 0, 0, 100
    with torch.no_grad():
        for i in range((input.shape[0]) // 100):
            data, target = input[n:m, :, :, :].cuda(), label[n:m].cuda()
            m += 100
            n += 100
            if args.ocm:
                probs = F.softmax(torch.log(F.relu(torch.matmul(model(data), C.T)) + 1e-6))
                pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                output = nn.Softmax()(model(data))
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nSub Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, input.shape[0], 100. * correct / input.shape[0]))
    return 100. * correct / input.shape[0]


def validate_batchwise_asr(model, data, target, xn, C):
    """ this function computes the accuracy for a given data and target on model """
    model.eval()
    correct = 0
    with torch.no_grad():
        data, target = data.cuda(), target.cuda()
        if args.ocm:
            probs = F.softmax(torch.log(F.relu(torch.matmul(model(data), C.T)) + 1e-6))
            pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            output = nn.Softmax()(model(data))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nSubTest set: Attack Success Rate: {}/{} ({:.4f}%)\n'.format(correct, xn, 100. * correct / xn))
    return 100. * correct / xn


def main():
    # Load dataset
    DATASET = datasets.__dict__[args.dataset](args)
    train_loader, test_loader = DATASET.loaders()

    if args.dataset == 'CIFAR10':
        args.attacksamp, args.auxiliary = 500, 500
        source_list = list(range(10))[args.source_start:args.source_end]
        target_list = []
        for s in source_list:
            if s == 0:
                target_list.append([9, 1])
            elif s == 9:
                target_list.append([8, 0])
            else:
                target_list.append([s - 1, s + 1])
    elif args.dataset == 'CIFAR100' or args.dataset == 'ImageNet':
        args.attacksamp = 50 if args.dataset == 'CIFAR100' else 25
        args.auxiliary = 50 if args.dataset == 'CIFAR100' else 25
        source_list = list(range(50))[args.source_start:args.source_end]
        target_list = []
        for s in source_list:
            if s == 0:
                target_list.append([49, 1])
            elif s == 49:
                target_list.append([48, 0])
            else:
                target_list.append([s - 1, s + 1])

    # Load model architecture
    if args.ocm:
        n_output = args.code_length
        criterion = L1Loss()
        C = torch.tensor(DATASET.C).to(device)
    else:
        assert args.output_act == 'linear'
        n_output = args.num_classes
        criterion = CrossEntropyLoss()
        C = torch.tensor(np.eye(args.num_classes)).to(device)

    # Evaluate clean accuracy
    model = models.__dict__[args.arch](n_output, args.bits, args.output_act)
    model = nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict'])
    weight_conversion(model)
    benign_test_acc = validate(model, test_loader, C)
    print(benign_test_acc)

    for (source, targets) in zip(source_list, target_list):
        for target in targets:
            print('source class: ', str(source), 'target class: ', str(target))
            args.source, args.target = source, target
            rounds = args.iters  # attack iterations
            acc = torch.Tensor(args.avgs, rounds + 1).fill_(0)  # accuracy tracker
            acc1 = torch.Tensor(args.avgs, rounds + 1).fill_(0)  # accuracy without attacked class and test samples used for attack
            temp = torch.Tensor(args.avgs, rounds + 1).fill_(0)  # ASR on attack samples
            temp1 = torch.Tensor(args.avgs, rounds + 1).fill_(0)  # ASR on rest of the samples
            layer = torch.Tensor(args.avgs, rounds + 1).fill_(0)
            offsets = torch.Tensor(args.avgs, rounds + 1).fill_(0)
            bfas = torch.Tensor(args.avgs).fill_(0)  # recording number of bit-flips
            for j in range(args.avgs):
                print("\nStealthy T-BFA Attack Repetition {}".format(j))
                attacker = SneakyBFA(criterion, C)
                model.load_state_dict(torch.load(args.outdir + 'model_best.pth.tar', map_location=device)['state_dict'])
                weight_conversion(model)
                data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2 = data_gen()
                targetT1[:], targetT2[:] = args.target, args.target
                acc[j, 0] = validate(model, test_loader, C)
                acc1[j, 0] = validate_batchwise(model, data2, target2, C)
                temp[j, 0] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                temp1[j, 0] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)
                for r in range(rounds):
                    layer[j, r + 1], offsets[j, r + 1] = attacker.progressive_bit_search(model, dataT1, targetT1.long(), data1, target1.long(), args)
                    acc[j, r + 1] = validate(model, test_loader, C)
                    acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                    temp[j, r + 1] = validate_batchwise_asr(model, dataT1, targetT1.long(), args.attacksamp, C)
                    temp1[j, r + 1] = validate_batchwise_asr(model, dataT2, targetT2.long(), args.attacksamp, C)
                    if float(temp1[j, r + 1]) > 99.9 or float(temp[j, r + 1]) > 99.9:
                        acc[j, r + 1] = validate(model, test_loader, C)
                        acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                        break
                    if attacker.random_flip_flag:
                        acc[j, r + 1] = validate(model, test_loader, C)
                        acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                        break
                    if r > 1 and offsets[j, r - 1] == offsets[j, r + 1] and offsets[j, r - 1] == offsets[j, r]:
                        if layer[j, r - 1] == layer[j, r + 1] and layer[j, r - 1] == layer[j, r]:
                            acc[j, r + 1] = validate(model, test_loader, C)
                            acc1[j, r + 1] = validate_batchwise(model, data2, target2, C)
                            acc[j, r] = acc[j, r + 1]
                            acc[j, r - 1] = acc[j, r + 1]
                            acc1[j, r] = acc1[j, r + 1]
                            acc1[j, r - 1] = acc1[j, r + 1]
                            break
                bfas[j] = int(r + 1)
                del data1, target1, data2, target2, dataT1, targetT1, dataT2, targetT2

            test_acc = torch.Tensor(args.avgs).fill_(0)     # overall test accuracy
            ASR_as = torch.Tensor(args.avgs).fill_(0)       # ASR
            ASR_val = torch.Tensor(args.avgs).fill_(0)      # validation ASR (for generalization)
            rem_acc = torch.Tensor(args.avgs).fill_(0)      # accuracy on remaining data (source class and val set not present)

            for i in range(args.avgs):
                test_acc[i] = acc[i, int(bfas[i])]
                ASR_as[i] = temp[i, int(bfas[i])]
                ASR_val[i] = temp1[i, int(bfas[i])]
                rem_acc[i] = acc1[i, int(bfas[i])]

            print(test_acc)
            print(ASR_as)
            print(ASR_val)
            print(rem_acc)
            print(bfas.mean())
            print(bfas.std())

            wb = Workbook()
            sheet1 = wb.add_sheet('Sheet1')
            sheet1.write(0, 0, ("Test_ACC"))
            sheet1.write(0, 1, ("ASR_AS "))
            sheet1.write(0, 2, ("ASR_rest "))
            sheet1.write(0, 3, ("layer number"))
            sheet1.write(0, 4, ("offset"))
            sheet1.write(0, 5, ("PA_ACC"))
            count = 0
            for j in range(args.avgs):
                for p in range(int(bfas[j]) + 1):
                    sheet1.write(p + 1 + count, 0, float(acc[j, p]))
                    sheet1.write(p + 1 + count, 1, float(temp[j, p]))
                    sheet1.write(p + 1 + count, 2, float(temp1[j, p]))
                    sheet1.write(p + 1 + count, 3, float(layer[j, p]))
                    sheet1.write(p + 1 + count, 4, float(offsets[j, p]))
                    sheet1.write(p + 1 + count, 5, float(acc1[j, p]))
                count += int(bfas[j]) + 2

            sheet2 = wb.add_sheet('Sheet2')
            sheet2.write(0, 0, ("Test_ACC"))
            sheet2.write(0, 1, ("PA_ACC"))
            sheet2.write(0, 2, ("ASR_AS "))
            sheet2.write(0, 3, ("ASR_rest "))
            sheet2.write(0, 4, ("Bitflips "))
            sheet2.write(1, 0, float(test_acc.mean()))
            sheet2.write(2, 0, float(test_acc.std()))
            sheet2.write(1, 1, float(rem_acc.mean()))
            sheet2.write(2, 1, float(rem_acc.std()))
            sheet2.write(1, 2, float(ASR_as.mean()))
            sheet2.write(2, 2, float(ASR_as.std()))
            sheet2.write(1, 3, float(ASR_val.mean()))
            sheet2.write(2, 3, float(ASR_val.std()))
            sheet2.write(1, 4, float(bfas.mean()))
            sheet2.write(2, 4, float(bfas.std()))
            file_string = "tbfa_results/" + args.outdir[8:-1] + "_from_" + str(args.source) + "_to_" + str(args.target)
            wb.save(file_string + ".xls")


if __name__ == "__main__":
    main()
