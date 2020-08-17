# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 08:59:37 2020

@author: admin-local
"""
import torch
import resnet as RN
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
#import torch.nn as nn

from utils import AverageMeter
import torch.nn as nn
import time


gpu_devices = ','.join([str(id) for id in '0123'])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

print("gpu devices " + gpu_devices )

""" to load model from Data Parallel """
class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)


class Normalize(nn.Module):
    def __init__(self, mean, std, i_gpu):
        super(Normalize, self).__init__()
        self.mean = (torch.tensor(mean)[None, :, None, None]).to('cuda:'+str(i_gpu))
        self.std = (torch.tensor(std)[None, :, None, None]).to('cuda:'+str(i_gpu))

    def forward(self, input):
        x = input
        x = x - self.mean
        x = x / self.std
        return x


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

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test( model_ERM, model_MixUp, model_CutMix ,model_MixUpAttack, test_loader, epsilon ):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_MU = AverageMeter()
    top5_MU = AverageMeter()
    
    top1_CM = AverageMeter()
    top5_CM = AverageMeter()
    
    top1_MUA = AverageMeter()
    top5_MUA = AverageMeter()
    
    
    # Accuracy counter
    model_MixUp.eval()
    model_CutMix.eval()
    model_MixUpAttack.eval()
    model_ERM.eval()
    # Loop over all examples in test set
    end = time.time()
    for i,(data, target) in enumerate(test_loader):
        data_time.update(time.time() - end)

        # Send the data and label to the device
        data, target = data.to('cuda:0'), target.to('cuda:0')
#        print('data shape')
#        print(data.shape)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model_ERM(data)
      

        # Calculate the loss
        #criterion = nn.CrossEntropyLoss().cuda()
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model_ERM.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model_MixUp(perturbed_data.to('cuda:1'))

        # Check for success
        err1, err5 = accuracy(output.data, target.to('cuda:1'), topk=(1, 5))
        
        
        top1_MU.update(err1.item(), data.size(0))
        top5_MU.update(err5.item(), data.size(0))

        output = model_CutMix(perturbed_data.to('cuda:2'))

        # Check for success
        err1, err5 = accuracy(output.data, target.to('cuda:2'), topk=(1, 5))
        
        
        top1_CM.update(err1.item(), data.size(0))
        top5_CM.update(err5.item(), data.size(0))

        output = model_MixUpAttack(perturbed_data.to('cuda:3'))
        
        err1, err5 = accuracy(output.data, target.to('cuda:3'), topk=(1, 5))
        
        
        top1_MUA.update(err1.item(), data.size(0))
        top5_MUA.update(err5.item(), data.size(0))


        batch_time.update(time.time() - end)
        end = time.time()

       
        print('Batch [{0}/{1}] | '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '.format(
            i, len(test_loader), batch_time=batch_time,
            data_time=data_time))


    
    # Return the accuracy and an adversarial example
    return top1_MU, top5_MU, top1_CM, top5_CM, top1_MUA,top5_MUA

""" declare number of class  by user """
numberofclass = 
    
   
""" create and load model ERM """
print("=> creating model ResNet50 ")
model_ERM = RN.ResNet('imagenet', 50, numberofclass, False)  # for ResNet50

print('Total params model ERM: %.2fM' % (sum(p.numel() for p in model_ERM.parameters())/1000000.0))

""" declare path to saved model ERM, type pth.tar """
resume_ERM = 
assert os.path.isfile(resume_ERM), 'Error: no checkpoint directory found!'

model_ERM = WrappedModel(model_ERM)
model_ERM.load_state_dict(torch.load(resume_ERM)['state_dict'])


model_ERM = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],0),model_ERM).to('cuda:0')



""" create and load model MixUp """
print("=> creating model ResNet50 ")
model_MixUp = RN.ResNet('imagenet', 50, numberofclass, False)  # for ResNet50

print('Total params model MixUp: %.2fM' % (sum(p.numel() for p in model_MixUp.parameters())/1000000.0))

""" declare path to saved model Mix-Up, type pth.tar """
resume_MixUp = 
assert os.path.isfile(resume_MixUp), 'Error: no checkpoint directory found!'

model_MixUp = WrappedModel(model_MixUp)
model_MixUp.load_state_dict(torch.load(resume_MixUp)['state_dict'])

model_MixUp = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],1),model_MixUp).to('cuda:1')
    


""" create and load model CutMix """
print("=> creating model ResNet50 ")
model_CutMix = RN.ResNet('imagenet', 50, numberofclass, False)  # for ResNet50

print('Total params model CutMix: %.2fM' % (sum(p.numel() for p in model_CutMix.parameters())/1000000.0))

""" declare path to saved model Cut-Mix, type pth.tar """
resume_CutMix = 
assert os.path.isfile(resume_CutMix), 'Error: no checkpoint directory found!'

model_CutMix = WrappedModel(model_CutMix)
model_CutMix.load_state_dict(torch.load(resume_CutMix)['state_dict'])

model_CutMix = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],2),model_CutMix).to('cuda:2')



""" create and load model MixUpAttack """
print("=> creating model ResNet50 ")
model_MixUpAttack = RN.ResNet('imagenet', 50, numberofclass, False)  # for ResNet50

print('Total params model CutMix: %.2fM' % (sum(p.numel() for p in model_MixUpAttack.parameters())/1000000.0))

""" declare path to saved model MixUpAttack, type pth.tar """
resume_MixUpAttack = 
assert os.path.isfile(resume_MixUpAttack), 'Error: no checkpoint directory found!'

model_MixUpAttack = WrappedModel(model_MixUpAttack)
model_MixUpAttack.load_state_dict(torch.load(resume_MixUpAttack)['state_dict'])

model_MixUpAttack = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],3),model_MixUpAttack).to('cuda:3')



valdir = os.path.join('/home/kt254686/Data/ImageNet/val')



val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=100, shuffle=False,
        num_workers=4, pin_memory=True)


""" epsilon in FGSM """
epsilon = 

""" evaluation """
top1_MU, top5_MU ,top1_CM, top5_CM, top1_MUA, top5_MUA = test( model_ERM, model_MixUp,  model_CutMix,model_MixUpAttack,  val_loader, epsilon )

""" declare path output folder """
out_dir = 

if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

dir_output = os.path.join(out_dir, 'result.txt')

f = open(dir_output,"w")
f.write( 'top1_MixUp = ' + str(top1_MU.avg) + '\n')
f.write( 'top5_MixUp = ' + str(top5_MU.avg) + '\n')


f.write( 'top1_CutMix = ' + str(top1_CM.avg) + '\n')
f.write( 'top5_CutMix = ' + str(top5_CM.avg) + '\n')


f.write( 'top1_MixUpAttack = ' + str(top1_MUA.avg) + '\n')
f.write( 'top5_MixUpAttack = ' + str(top5_MUA.avg) + '\n')

f.close()


print('MixUp Top 1-err ({top1.avg:.4f}) | '
             'Top 5-err ({top5.avg:.4f})'.format(
           top1=top1_MU, top5=top5_MU))
        


print('CutMix Top 1-err  ({top1.avg:.4f}) | '
             'Top 5-err  ({top5.avg:.4f})'.format(
           top1=top1_CM, top5=top5_CM))


print('MixUpAttack Top 1-err  ({top1.avg:.4f}) | '
             'Top 5-err  ({top5.avg:.4f})'.format(
           top1=top1_MUA, top5=top5_MUA))
