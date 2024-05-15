import torch
import numpy as np
from data_utils.data_utils_training import  AttackParser,train_convs, defend_jpeg, load_model_rob, load_model_nat, Identity
from data_utils.data_utils import AdversarialDataset
import argparse
import foolbox as fb
import copy
import torch.nn as nn
import pynvml
import time 
parser = AttackParser(argparse.ArgumentParser())
parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/valid.csv')
parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/beagle_examples/'
# parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/mscoco/unlabeled2017'
measures_path = parser.save_path+'/curves/'

from torch.nn.utils import prune
def prune_model_l1_unstructured(model, prop):
    module = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and module.weight is not None:
                prune.l1_unstructured(module, 'weight', prop)
                prune.remove(module, 'weight')
            if hasattr(module, "bias")and module.bias is not None:
                prune.l1_unstructured(module, 'bias', prop)
                prune.remove(module, 'bias')
            print('pruned last')
    return model

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return (mem_info.free//1024//1024)

print('NO SHUFFLE NO AUGMENT')
limit= parser.limit
orig_set = AdversarialDataset(parser, img_size=224)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=parser.batch_size, shuffle=False)
parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/nips17/'
parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/nips17.csv')
accuracy_set = AdversarialDataset(parser, img_size=224, limit_dataset=100)
accuracy_loader = torch.utils.data.DataLoader(accuracy_set, batch_size=parser.batch_size, shuffle=False)

model_softmax, conv_nat = load_model_nat(parser, 0)
preprocess,effnet,softmax = model_softmax

if 'efficient' in parser.models[0]:
    classifier = effnet.classifier
elif 'resnet' in parser.models[0]:
    classifier = effnet.fc

fmodel = fb.PyTorchModel(model_softmax, bounds=[0,255])

conv_nat, _= train_convs(conv_nat,0)

train_loss_adv = np.zeros(512)
eval_loss_adv = np.zeros(512)
cpt=0
# print("avant", get_memory_free_MiB(0))
def transform_image(attack_model, method, image_batch, eps_value, labels):
    batch_size = image_batch.shape[0]
    prediction = attack_model(image_batch)

    prediction_sorted = prediction.argsort(1)
    pred_labels = prediction_sorted[:,-1]
    adv_class = prediction_sorted[:,-2]
    adv_class = (adv_class!=labels)*adv_class + (adv_class==labels)*pred_labels

    labels_onehot = torch.zeros(labels.size(0), 1000, device=image_batch.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1).long(),1)

    adv_labels_onehot = torch.zeros(adv_class.size(0), 1000, device=image_batch.device)
    adv_labels_onehot.scatter_(1, adv_class.unsqueeze(1).long(),1)

    classification_loss = prediction*labels_onehot-prediction*adv_labels_onehot
    classification_loss = classification_loss.sum(axis=1)
    classification_loss.mean().backward()

    gradients = image_batch.grad.data
    gradients_normalized = gradients/(gradients.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1))
    
    adversarial_image = image_batch - 388*gradients_normalized*eps_value

    prediction = attack_model(adversarial_image)

    prediction_sorted = prediction.argsort(1)
    pred_labels = prediction_sorted[:,-1]

    return(adversarial_image)

model_softmax = prune_model_l1_unstructured(model_softmax,0.4)
total_correct=0
for j, data_batch in enumerate(accuracy_loader, 0):
        orig_batch, initial_labels, _ = data_batch
        with torch.no_grad():
            pred = model_softmax(orig_batch)
        pred_labels = pred.argmax(1)
        correct_preds = (pred_labels==initial_labels).sum()
        total_correct += correct_preds
print("accuracy: ", total_correct/10)

for i, data_batch in enumerate(orig_loader, 0):

    orig_batch, labels, _ = data_batch
    orig_batch= torch.autograd.Variable(orig_batch, requires_grad=True)
    adversarial_image = transform_image(model_softmax, parser.transforms, orig_batch, 1, labels) 
    batch_size = orig_batch.shape[0]

    with torch.no_grad():
        nat_conv_nat = conv_nat(orig_batch)
        adv_conv_nat = conv_nat(adversarial_image)

    nat_vals = nat_conv_nat.mean(0).detach().cpu().numpy()
    adv_vals = adv_conv_nat.mean(0).detach().cpu().numpy()
    nat_stds = nat_conv_nat.std(0).detach().cpu().numpy()
    adv_stds = adv_conv_nat.std(0).detach().cpu().numpy()
        
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


fig, axs = plt.subplots(figsize=(24,12))
nat_args = nat_vals.argsort()
new_vals = np.zeros(512)
new_stds = np.zeros(512)
for i in range(512):
    new_vals[i] = nat_vals[nat_args[i]]
    new_stds[i] = nat_stds[nat_args[i]]

data_id  = np.arange(nat_vals.shape[0])

# curve = axs.plot(data_id, nat_vals, 'b', label='f_mean', linewidth=1)
# curve = axs.errorbar(data_id, new_vals, new_stds, color='b', ecolor='r', linestyle='None', marker='^')
curve = axs.errorbar(data_id, new_vals, color='b', linestyle='None', marker='^')
curve = axs.errorbar(data_id, new_stds, color='r', linestyle='None', marker='x')
axs.set_ylim(0, 10)
# axs.set_xlim(0, 1024)
# axs.set_xticks(np.arange(0, 100+0.1, 50))


plt.xlabel("Features", {'fontsize': 14})
plt.ylabel("Value", {'fontsize': 14})
plt.grid( linestyle="dashed")
# axs.set_title(fig_title, fontweight="bold")
plt.savefig("{}/bgl/nat_beagles.jpeg".format(measures_path),bbox_inches='tight')


fig, axs = plt.subplots(figsize=(24,12))
nat_args = adv_vals.argsort()
new_vals = np.zeros(512)
new_stds = np.zeros(512)

for i in range(512):
    new_vals[i] = adv_vals[nat_args[i]]
    new_stds[i] = adv_stds[nat_args[i]]

data_id  = np.arange(nat_vals.shape[0])


curve = axs.errorbar(data_id, new_vals, color='b', linestyle='None', marker='^')
curve = axs.errorbar(data_id, new_stds, color='r', linestyle='None', marker='x')
axs.set_ylim(0, 10)
# axs.set_xticks(np.arange(0, 100+0.1, 50))


plt.xlabel("Features", {'fontsize': 14})
plt.ylabel("Value", {'fontsize': 14})
plt.grid( linestyle="dashed")
# axs.set_title(fig_title, fontweight="bold")
plt.savefig("{}/bgl/adv_beagles.jpeg".format(measures_path),bbox_inches='tight')