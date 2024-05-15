import torch
import numpy as np
from data_utils.data_utils_training import  AdversarialDataset,AttackParser,train_convs, defend_jpeg, load_model_rob, load_model_nat, Identity
import argparse
import foolbox as fb
import copy
import torch.nn as nn
import pynvml
import time 
parser = AttackParser(argparse.ArgumentParser())
parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/valid.csv')
parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/imagenet_val/'
# parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/mscoco/unlabeled2017'
# parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/imagenet_val_small_centered/'
# parser.labels_path = ('/gpfswork/rech/gbk/uzz94dr/data/labels/processed.csv')
# parser.input_path = '/gpfswork/rech/gbk/uzz94dr/data/images/test/'

measures_path = parser.save_path+'/curves/'
ckpts_path = parser.save_path+'/ckpts/'

from torch.nn.utils import prune



def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return (mem_info.free//1024//1024)

print('NO SHUFFLE NO AUGMENT')
limit= parser.limit
orig_set = AdversarialDataset(parser, img_size=224, limit_dataset=limit, augment=True)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=parser.batch_size, shuffle=True)
valid_set = AdversarialDataset(parser, img_size=224, limit_dataset=limit, valid=True, augment=False)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=parser.batch_size, shuffle=False)
parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/nips17/'
parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/nips17.csv')
accuracy_set = AdversarialDataset(parser, img_size=224, limit_dataset=100, augment=False)
accuracy_loader = torch.utils.data.DataLoader(accuracy_set, batch_size=parser.batch_size, shuffle=False)
print(len(orig_loader), "train batches")
print(len(valid_loader), "valid batches")
cpt3=0
epochs = parser.epochs
display = [0]#,400,800,1200,1600,2000]
old_ckpt = ckpts_path+"/baseline.pth"
# old_ckpt = ckpts_path+"/prout.pth"
print(old_ckpt)

model_softmax, conv_nat = load_model_nat(parser, 0)
preprocess,effnet,softmax = model_softmax

if 'efficient' in parser.models[0]:
    classifier = effnet.classifier
elif 'resnet' in parser.models[0]:
    classifier = effnet.fc
    

conv_rob = load_model_rob(parser, old_ckpt)

weights = classifier.weight.detach()
weights_obj = weights/weights.min().abs() + 1

# weights_obj = weights/weights.view(1000,-1).norm(dim=1).view(1000,1)


_,effnet_rob = conv_rob
effnet_rob.fc = classifier
new_model = nn.Sequential(preprocess, copy.deepcopy(effnet_rob), softmax)
new_model.eval();
effnet_rob.fc = Identity()
fmodel = fb.PyTorchModel(model_softmax, bounds=[0,255])

conv_nat, _= train_convs(conv_nat,0)
conv_rob, params= train_convs(conv_rob,parser.convs)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, conv_rob.parameters()), lr=1*10**(-parser.lr))
epsilon = parser.epsilon
print("training on {} epochs {} images, disto {}. {} convs are trained".format(epochs, limit, epsilon, len([a for a in filter(lambda p: p.requires_grad, conv_rob.parameters())])))
print("learning rate: ", 10**(-parser.lr))

train_loss_adv = np.zeros(5*epochs)
eval_loss_adv = np.zeros(5*epochs)
train_loss_nat = np.zeros(5*epochs)
eval_loss_nat = np.zeros(5*epochs)
accuracy_rob = np.zeros(5*epochs)
eval_loss_oth = np.zeros((5*epochs,6))
train_loss_oth = np.zeros((5*epochs,6))
cpt=0

def transform_signal(weight_signal, features_signal, weight_labels):
    
    labeled_weights = weight_labels@weight_signal
    batch_size, n_size  = features_signal.shape
    labeled_weights = labeled_weights.view(batch_size,-1)
    features_signal = features_signal.view(batch_size,-1)
    #0-mean on weights signal
    labeled_weights = labeled_weights - labeled_weights.mean(1).unsqueeze(-1)
    normalization_vector = torch.sqrt((features_signal.norm(dim=1)**2-n_size*features_signal.mean(dim=1)**2)/labeled_weights.norm(dim=1)**2)
    normalized_weight_signal = labeled_weights*normalization_vector.unsqueeze(-1) + features_signal.mean(dim=1).unsqueeze(-1)

    return(normalized_weight_signal.view(features_signal.shape))

#conv_rob = normalize_convs(conv_rob)
for epoch in range(epochs):
    for i, data_batch in enumerate(orig_loader, 0):

        optimizer.zero_grad();
        orig_batch, initial_labels, _ = data_batch
        orig_batch= torch.autograd.Variable(orig_batch, requires_grad=True)
        batch_size = orig_batch.shape[0]


        labels_onehot = torch.zeros(initial_labels.size(0), 1000, device=orig_batch.device)
        labels_onehot.scatter_(1, initial_labels.unsqueeze(1).long(),1)

        with torch.no_grad():
            nat_conv_nat = conv_nat(orig_batch)
        nat_conv_rob = conv_rob(orig_batch)
        # nat_conv_rob_norm = nat_conv_rob.detach().view(batch_size,-1).norm(dim=1).view(batch_size,1)
        nat_conv_nat_norm = nat_conv_nat.detach().view(batch_size,-1).norm(dim=1).view(batch_size,1)
    
        labeled_weights = labels_onehot@weights_obj
        target_weights =  labeled_weights*nat_conv_nat.mean() #transform_signal(weights_obj, nat_conv_nat, labels_onehot)


        """MSE"""
        # sqerror_nat =((target_weights - nat_conv_rob)**2).view(batch_size,-1).sum(dim=1)  
        # sqerror =  sqerror_nat.mean()
        # print((target_weights*nat_conv_rob_norm).shape)
        # print((target_weights*nat_conv_rob_norm)[0,:50])
        # print((nat_conv_rob)[0,:50])
        """MSLE"""
        sqerror_nat_msle  =((torch.log(target_weights+1) - torch.log(nat_conv_rob+1))**2).view(batch_size,-1).sum(dim=1)
        sqerror =  sqerror_nat_msle.mean()
        if i%50==0:
            print(target_weights[0].norm(), nat_conv_rob[0].norm(), nat_conv_nat[0].norm(), sqerror.mean())
            print("max", target_weights[0].max(), nat_conv_rob[0].max(), "mean",  target_weights[0].mean(), nat_conv_nat[0].mean())
        # exit()

        sqerror.backward()
        optimizer.step()
         
        """MISE A JOUR DU MODELE D ATTAQUE"""
        # if epoch%20==0:
        _,effnet_rob = conv_rob
        effnet_rob.fc = classifier
        new_model = nn.Sequential(preprocess, copy.deepcopy(effnet_rob), softmax)
        new_model.eval();
        effnet_rob.fc = Identity()
    
        if i%((len(orig_loader)+10)//4)==0:
            train_loss_nat[cpt] = sqerror            
            np.save(measures_path+'train_nat.npy',train_loss_nat)


            for j, data_batch in enumerate(valid_loader, 0):
                optimizer.zero_grad();
                orig_batch, initial_labels, _ = data_batch
                orig_batch = torch.autograd.Variable(orig_batch, requires_grad=True)
                batch_size = orig_batch.shape[0]

                labels_onehot = torch.zeros(initial_labels.size(0), 1000, device=orig_batch.device)
                labels_onehot.scatter_(1, initial_labels.unsqueeze(1).long(),1)

                with torch.no_grad():
                    nat_conv_nat = conv_nat(orig_batch)
                    nat_conv_rob = conv_rob(orig_batch)

                nat_conv_rob_norm = nat_conv_rob.detach().view(batch_size,-1).norm(dim=1).view(batch_size,1)
                nat_conv_rob = nat_conv_rob/nat_conv_rob_norm
                target_weights = transform_signal(weights_obj,nat_conv_nat,labels_onehot)
                    
                nat_conv_nat_norm = nat_conv_nat.detach().view(batch_size,-1).norm(dim=1).view(batch_size,1)
                nat_conv_nat = nat_conv_rob/nat_conv_nat_norm

                sqerror_nat =((target_weights - nat_conv_rob)**2).view(batch_size,-1).sum(dim=1)  
                sqerror_init=((target_weights - nat_conv_nat)**2).view(batch_size,-1).sum(dim=1)  


                eval_loss_nat[cpt] = sqerror_nat.mean()

                np.save(measures_path+'eval_nat.npy',eval_loss_nat)
            
   
            total_correct = 0
            for j, data_batch in enumerate(accuracy_loader, 0):
                orig_batch, initial_labels, _ = data_batch
                with torch.no_grad():
                    pred = new_model(orig_batch)
                pred_labels = pred.argmax(1)
                correct_preds = (pred_labels==initial_labels).sum()
                total_correct += correct_preds
            accuracy_rob[cpt] = total_correct/10
            print("accuracy: ", total_correct/10, "train: ",train_loss_nat[cpt], "eval: ",eval_loss_nat[cpt], "eval nat: ", sqerror_init.mean(), "norm: ", nat_conv_rob_norm[0])
            
            
            np.save(measures_path+'accuracy.npy',accuracy_rob)
            _,core_new,_ = new_model
            torch.save(core_new.state_dict(), ckpts_path+"last.pth")
            cpt+=1