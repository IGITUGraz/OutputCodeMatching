import argparse
import time
import datetime
import datasets
import models
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import SGD, Adam


parser = argparse.ArgumentParser(description='Improving stealthy BFA robustness via output code matching')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--ocm', action='store_true', help='output layer coding with bit strings')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (only linear and tanh is supported)')
parser.add_argument('--code_length', '-cl', default=16, type=int, help='length of codewords')
parser.add_argument('--outdir', type=str, default='results/', help='folder to save model and training log')
parser.add_argument('--epochs', '-e', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch', '-b', default=128, type=int, metavar='N', help='Mini-batch size (default: 128)')
parser.add_argument('--opt', type=str, default='sgd', help='sgd or adam optimizer')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--schedule', type=str, default='step', help='learning rate schedule')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='weight decay (default: 1e-4 for OCM)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=250, type=int, help='print frequency (default: 250)')
parser.add_argument('--clustering', '-pc', action='store_true', help='add piecewise clustering regularization')
parser.add_argument('--lambda_coeff', '-lam', type=float, default=1e-3, help='piecewise clustering strength')
parser.add_argument('--resume', action="store_true", help='resume training from path')
parser.add_argument('--finetune', action="store_true", help='for finetuning pre-trained imagenet models')
parser.add_argument('--ft_path', type=str, default='results/imagenet/resnet50_quan8/', help='finetune model path')
parser.add_argument('--eval', action="store_true", help='load saved model weights from path to evaluate only')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

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


def train(loader, model, criterion, optimizer, epoch, C):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, data in enumerate(loader):
        data_time.update(time.time() - end)

        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        if args.ocm:
            output_probs = lambda z: F.softmax(torch.log(F.relu(torch.matmul(z, C.T)) + 1e-6))
            probs = output_probs(outputs)
            labels = torch.tensor([torch.where(torch.all(C == targets[i], dim=1))[0][0] for i in range(targets.shape[0])])
            acc1, acc5 = accuracy(probs, labels.to(device), topk=(1, 5))
        else:
            acc1, acc5 = accuracy(nn.Softmax()(outputs), targets, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def test(loader, model, criterion, C):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader), [batch_time, losses, top1, top5], prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if args.clustering:
                loss += clustering_loss(model, args.lambda_coeff)

            if args.ocm:
                output_probs = lambda z: F.softmax(torch.log(F.relu(torch.matmul(z, C.T)) + 1e-6))
                probs = output_probs(outputs)
                labels = torch.tensor([torch.where(torch.all(C == targets[i], dim=1))[0][0] for i in range(targets.shape[0])])
                acc1, acc5 = accuracy(probs, labels.to(device), topk=(1, 5))
            else:
                acc1, acc5 = accuracy(nn.Softmax()(outputs), targets, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg


def main():
    # Load dataset and model architecture
    DATASET = datasets.__dict__[args.dataset](args)
    train_loader, test_loader = DATASET.loaders()

    if args.ocm:
        n_output = args.code_length
        criterion = L1Loss()
        C = torch.tensor(DATASET.C).to(device)
    else:
        assert args.output_act == 'linear'
        n_output = args.num_classes
        criterion = CrossEntropyLoss()
        C = torch.tensor(np.eye(args.num_classes)).to(device)
    model = models.__dict__[args.arch](n_output, args.bits, args.output_act)
    model = nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else nn.DataParallel(model).to(device)
    
    if args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    log_filename = os.path.join(args.outdir, 'log.txt')
    
    if not args.eval:
        if args.resume:
            resume_best = torch.load(args.outdir + 'model_best.pth.tar', map_location=device)
            model.load_state_dict(resume_best['state_dict'])
            _, best_acc1 = test(test_loader, model, criterion, C)
            resume = torch.load(args.outdir + 'checkpoint.pth.tar', map_location=device)
            model.load_state_dict(resume['state_dict'])
            optimizer.load_state_dict(resume['optimizer'])
            start_epoch = resume['epoch']
        else:
            if args.finetune:
                pre_dict = torch.load(args.ft_path + 'model_best.pth.tar', map_location=device)['state_dict']
                pre_dict = {k: v for k, v in pre_dict.items() if 'module.linear' not in k}
                model.load_state_dict(pre_dict, strict=False)
                init_logfile(log_filename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
                start_epoch, best_acc1 = 0, 0
            else:
                init_logfile(log_filename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
                start_epoch, best_acc1 = 0, 0

        for epoch in range(start_epoch, args.epochs):
            lr = lr_scheduler(optimizer, epoch, args)

            before = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, C)
            test_loss, test_acc = test(test_loader, model, criterion, C)
            after = time.time()

            is_best = test_acc > best_acc1
            best_acc1 = max(test_acc, best_acc1)

            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                             'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, is_best, args.outdir)

            log(log_filename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))), lr, train_loss, train_acc, test_loss, test_acc))
    else:
        test(test_loader, model, criterion, C)


if __name__ == "__main__":
    main()
