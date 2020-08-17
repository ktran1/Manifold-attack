from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import resnet as RN
from utils import Logger, AverageMeter, mkdir_p, ColorJitter, Lighting




def seed_(p):
    """  for reproductive """
    torch.manual_seed(p)
    np.random.seed(p)
    random.seed(p)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(p)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
            
    return 0



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Optimization options

parser.add_argument('--method', default='ERM', type=str,
                    help='ERM, MixUp, MixUpAttack, CutMix (default: ERM)')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

#parser.add_argument('--resume', default='C:/Hung/Python/Torch/RAE_ImageNet/result/MixUp/checkpoint.pth.tar', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options

parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

#Method options

parser.add_argument('--out', default='./result',
                        help='Directory to output the result')

parser.add_argument('--beta', default=1., type=float,
                    help='hyperparameter in the beta distribution')

parser.add_argument('--xi', default=0.1, type=float)

parser.add_argument('--xi-end', default=0.01, type=float)

parser.add_argument('--n_iters', default=1, type=int)

parser.add_argument('--lock', default=False, type= bool,
                    help='in attack stage, keep only greater loss')


args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}



# Use CUDA
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

use_cuda = torch.cuda.is_available()
if use_cuda :
    print('Using gpu ' + str(gpu_devices))
else:
    print('Using cpu, maybe very slow ')    


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

seed_(args.manualSeed)

best_err1 = 100
best_err5 = 100  # best test accuracy

def main():
    global args, best_err1, best_err5

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Config save
    dir_output = os.path.join(args.out, 'config.txt')
    f = open(dir_output,"w")
    for k, v in state.items():
        f.write(str(k) + ' = '+ str(v) + '\n')
    f.close()
    
    """ declare path to train and valid data folder """  
    traindir = 
    valdir = 
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])
    
    print("=> loading data ... ")

    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    numberofclass = 948
    
   
        

    print("=> creating model ResNet50 ")
    model = RN.ResNet('imagenet', 50, numberofclass, False)  # for ResNet50
    
    
    
    if len(args.gpu_devices) > 1 :
        model = torch.nn.DataParallel(model).cuda()  
    else:
        model = model.cuda()
        print('using only one gpu')
        

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    start_epoch = 0

    # Resume
    title = 'imagenet_resnet_50'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_err1 = checkpoint['best_err1']
        best_err5 = checkpoint['best_err5']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
        seed_(random.randint(1, 1000000000))
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Valid Loss', 'Valid Err1','Valid Err5'])

    Beta_dis = torch.distributions.beta.Beta(args.beta, args.beta)

    # Train and val
    for epoch in range(start_epoch+1, args.epochs+1):
        
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, get_learning_rate(optimizer)[0]))

        
        # train for one epoch
        if args.method =='ERM':
            train_loss = train_ERM(train_loader, model, criterion, optimizer, epoch)
        elif args.method =='MixUp':
            train_loss = train_MixUp(train_loader, model, criterion, optimizer, epoch, Beta_dis)
        elif args.method =='CutMix':
            train_loss = train_CutMix(train_loader, model, criterion, optimizer, epoch, Beta_dis)
        elif args.method =='MixUpAttack':
            xi_epoch = args.xi - float(epoch-1.)*(args.xi - args.xi_end) / float(args.epochs-1) 
            train_loss = train_MixUpAttack(train_loader, model, criterion, optimizer, epoch, Beta_dis, xi_epoch)    
        
        else:
            print('not valid training method')
            exit()
        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)


       
        # append logger file
        logger.append([train_loss, val_loss, err1 ,err5])

        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        
    logger.close()

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train_ERM(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if i % args.print_freq == 0 :
            print('Epoch: [{0}/{1}][{2}/{3}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f}) | '
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg

def train_MixUp(train_loader, model, criterion, optimizer, epoch, Beta_dis):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

     
        # generate mixed sample
        lam = Beta_dis.sample()
        rand_index = torch.randperm(input.size()[0])
        # compute output
        output = model(lam*input + (1.-lam)*input[rand_index])
        loss = criterion(output, target) * lam + criterion(output, target[rand_index]) * (1. - lam)
       

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 :
            print('Epoch: [{0}/{1}][{2}/{3}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f}) | '
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
    epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def train_CutMix(train_loader, model, criterion, optimizer, epoch, Beta_dis):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

     
        # generate mixed sample
        lam = Beta_dis.sample()
        rand_index = torch.randperm(input.size()[0])
        target = target
        target[rand_index] = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        output = model(input)
        loss = criterion(output, target) * lam + criterion(output, target[rand_index]) * (1. - lam)
       

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 :
            print('Epoch: [{0}/{1}][{2}/{3}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f}) | '
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
    epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    

def train_MixUpAttack(train_loader, model, criterion, optimizer, epoch, Beta_dis, xi_epoch) :
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

     
        # generate mixed sample
        lam = Beta_dis.sample()
        
        rand_index = torch.randperm(input.size()[0])
        
        
        #Attack
        if args.lock == True :

            loss_best = torch.tensor(-float("Inf")).cuda()
            l_best = lam
            l = lam.cuda()       
            
    #        tab_loss= torch.zeros(args.n_iters+1)
    #        tab_l = torch.zeros(args.n_iters+1)  
            
            for j in range(args.n_iters+1):
                l.requires_grad = True
            
                output = model(l*input + (1.-l)*input[rand_index])
                
                loss_a = criterion(output, target) * l + criterion(output, target[rand_index]) * (1. - l)
           
    #            tab_loss[j] = loss_a.item()
    #            tab_l[j] = l.item()
                
                if (loss_a.item() > loss_best):
                    loss_best = loss_a.item()
                    l_best = l.item()
                
                if j < (args.n_iters) :
                    grad_l = torch.autograd.grad(loss_a, l)[0].detach()
                    
                    
                    l = l.detach()
                    l = torch.clamp(l + grad_l*xi_epoch,0.,1.)
                    del grad_l
                else:
                    l = l.detach()
                
                del loss_a
                
            output = model(l_best*input + (1.-l_best)*input[rand_index])   
            loss = criterion(output, target) * l_best + criterion(output, target[rand_index]) * (1. - l_best)   
            
    #        print(tab_loss)
    #        print(tab_l)
            
        else : 
            l = lam.cuda()       
            
    #        tab_loss= torch.zeros(args.n_iters+1)
    #        tab_l = torch.zeros(args.n_iters+1)  
            
            for j in range(args.n_iters+1):
                l.requires_grad = True
            
                output = model(l*input + (1.-l)*input[rand_index])
                
                loss = criterion(output, target) * l + criterion(output, target[rand_index]) * (1. - l)
           
    #            tab_loss[j] = loss.item()
    #            tab_l[j] = l.item()
                

                if j < (args.n_iters) :
                    grad_l = torch.autograd.grad(loss, l)[0].detach()
               
                    
                    l = l.detach()
                    l = torch.clamp(l + grad_l*xi_epoch,0.,1.)
                  
                    
                else:
                    l = l.detach()
                
    #        print(tab_loss)
    #        print(tab_l)

        
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 :
            print('Epoch: [{0}/{1}][{2}/{3}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f}) | '
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
    epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg

    
    
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
    
            output = model(input)
            loss = criterion(output, target)
    
            # measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))
    
            losses.update(loss.item(), input.size(0))
    
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0 :
                print('Test (on val set): [{0}/{1}][{2}/{3}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'Top 1-err {top1.val:.4f} ({top1.avg:.4f}) | '
                      'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                    epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            
        
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg
    
   
def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    

    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs or every 300 if epochs = 300"""
    if args.epochs == 300:
        lr = args.lr * (0.1 ** (epoch // 75))
    else:
        lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
    
    
if __name__ == '__main__':
    main()
