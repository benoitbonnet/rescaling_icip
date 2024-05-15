import torch
import numpy as np
from data_utils.data_utils_training import  AdversarialDataset,AttackParser,train_convs, defend_jpeg, load_model_rob, load_model_nat, Identity
import argparse
import foolbox as fb
import copy
import torch.nn as nn
import pynvml
import time 
from torchvision import transforms, datasets
parser = AttackParser(argparse.ArgumentParser())


limit= parser.limit
epsilon = parser.epsilon
pgd_steps=parser.pgd_steps
epochs = parser.epochs
update_frac = 100

parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/valid.csv')
parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/imagenet_val'

valid_set = AdversarialDataset(parser, img_size=224, limit_dataset=limit, valid=True, augment=False)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=parser.batch_size, shuffle=False)

measures_path = parser.save_path+'/curves/'
ckpts_path = parser.save_path+'/ckpts/'


parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/imagenet_train'
orig_dataset = datasets.ImageFolder( parser.input_path , transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()]))
orig_loader = torch.utils.data.DataLoader(orig_dataset, batch_size=parser.batch_size, shuffle=True, num_workers=14)

print(len(orig_loader))


parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/nips17/'
parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/nips17.csv')
accuracy_set = AdversarialDataset(parser, img_size=224, limit_dataset=100, augment=False)
accuracy_loader = torch.utils.data.DataLoader(accuracy_set, batch_size=parser.batch_size, shuffle=False)
print(len(orig_loader), "train batches")
print(len(valid_loader), "valid batches")

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return (mem_info.free//1024//1024)

def update_model(robust_convs, fully_connected, preprocessing_layer, softmax_activation):
    _,effnet_rob = robust_convs
    effnet_rob.fc = fully_connected
    new_model = nn.Sequential(preprocessing_layer, copy.deepcopy(effnet_rob), softmax_activation)
    new_model.eval();
    effnet_rob.fc = Identity()
    return(new_model)

def transform_image(attack_model, method, image_batch, eps_value, nat_lab=None, adv_lab=None):
    batch_size = image_batch.shape[0]
    prediction = attack_model(image_batch)

    labels = prediction.argsort(1)
    if nat_lab==None and adv_lab==None:
        pred_labels = labels[:,-1]
        adv_class = labels[:,-2]
    elif adv_lab==None:
        pred_labels = nat_lab
        adv_class = labels[:,-2]
        adv_class =  (adv_class==nat_lab)*labels[:,-1]+(adv_class!=nat_lab)*adv_class
    else:
        pred_labels = nat_lab
        adv_class = adv_lab

    labels_onehot = torch.zeros(pred_labels.size(0), 1000, device=image_batch.device)
    labels_onehot.scatter_(1, pred_labels.unsqueeze(1).long(),1)

    adv_labels_onehot = torch.zeros(adv_class.size(0), 1000, device=image_batch.device)
    adv_labels_onehot.scatter_(1, adv_class.unsqueeze(1).long(),1)

    classification_loss = prediction*labels_onehot #-prediction*adv_labels_onehot
    classification_loss = classification_loss.sum(axis=1)
    classification_loss.mean().backward()

    gradients = image_batch.grad.data
    #print(classification_loss.mean())
    gradients_normalized = gradients/(gradients.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)+1e-9)
    #rand_norm = (torch.rand((batch_size, 1,1,1), device=image_batch.device)*4+2).round()/4

    #adversarial_image = (image_batch - gradients_normalized*387.9794*rand_norm)
    # adversarial_image = (image_batch - gradients_normalized*387.9794*classification_loss.detach().view(batch_size,1,1,1))
    adversarial_image = (image_batch - gradients_normalized*387.9794*eps_value)

    return(adversarial_image, pred_labels, adv_class)

def get_sigmas(natural_object, adversarial_object, batch_size):
    batch_size = natural_object.shape[0]
    object_dimension = np.prod(natural_object.shape[1:])
    kappa_x = natural_object.view(batch_size,-1).norm(dim=1)
    kappa_a = (natural_object-adversarial_object).view(batch_size,-1).norm(dim=1)

    sigma_x_object = kappa_x/np.sqrt(object_dimension)
    sigma_a_object = kappa_a/np.sqrt(object_dimension)
    
    return(sigma_a_object,sigma_x_object.detach(), object_dimension)

def get_losses(natural_conv_nat, natural_conv_rob, adversarial_conv_rob, image_batch, adversarial_batch):
    batch_size = natural_conv_nat.shape[0]

    natnat_norm = natural_conv_nat.view(batch_size,-1).norm(dim=1)
    natrob_norm = natural_conv_rob.view(batch_size,-1).norm(dim=1)
    advrob_norm = adversarial_conv_rob.view(batch_size,-1).norm(dim=1)

    # angle_nat = (natural_conv_nat.detach()*natural_conv_rob).view(batch_size,-1).sum(dim=1)/(natnat_norm.detach()*natrob_norm)
    # angle_adv = (natural_conv_rob.detach()*adversarial_conv_rob).view(batch_size,-1).sum(dim=1)/(natrob_norm.detach()*advrob_norm) 
    # angle_adv_2 =  (natural_conv_nat.detach()*adversarial_conv_rob).view(batch_size,-1).sum(dim=1)/(natnat_norm.detach()*advrob_norm)

    angle_nat = (natural_conv_nat*natural_conv_rob).view(batch_size,-1).sum(dim=1)/(natnat_norm*natrob_norm)
    angle_adv = (natural_conv_rob*adversarial_conv_rob).view(batch_size,-1).sum(dim=1)/(natrob_norm*advrob_norm) 
    #angle_adv_2 =  (natural_conv_nat*adversarial_conv_rob).view(batch_size,-1).sum(dim=1)/(natnat_norm*advrob_norm)

    sqerror_nat = (1-angle_nat**2)
    sqerror_adv = (1-angle_adv**2) #+ (1-angle_adv_2**2)

    
    return sqerror_nat, sqerror_adv

def measure_accuracy(acc_loader, current_model):
    total_correct = 0
    for j, data_batch in enumerate(acc_loader, 0):
        orig_batch, initial_labels, _ = data_batch
        with torch.no_grad():
            pred = current_model(orig_batch)
        pred_labels = pred.argmax(1)
        correct_preds = (pred_labels==initial_labels).sum()
        total_correct += correct_preds
    return(total_correct)

model_softmax, conv_nat = load_model_nat(parser, 0)
preprocess,effnet,softmax = model_softmax

if 'efficient' in parser.models[0]:
    classifier = effnet.classifier
elif 'resnet' in parser.models[0]:
    classifier = effnet.fc

conv_rob = load_model_rob(parser, '')
new_model = update_model(conv_rob, classifier, preprocess, softmax)
conv_nat, _= train_convs(conv_nat,0)
conv_rob, params= train_convs(conv_rob,parser.convs)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, conv_rob.parameters()), lr=1*10**(-parser.lr))
optimizer_class = torch.optim.Adam(filter(lambda p: p.requires_grad, conv_rob.parameters()), lr=1*10**(-5))

print("training on {} epochs {} images, disto {}. {} convs are trained".format(epochs, limit, epsilon, len([a for a in filter(lambda p: p.requires_grad, conv_rob.parameters())])))
print("learning rate: ", 10**(-parser.lr))

train_loss_adv = np.zeros(update_frac*epochs)
eval_loss_adv = np.zeros(update_frac*epochs)
train_loss_nat = np.zeros(update_frac*epochs)
eval_loss_nat = np.zeros(update_frac*epochs)
train_loss_rob = np.zeros(update_frac*epochs)
eval_loss_rob = np.zeros(update_frac*epochs)
accuracy_rob = np.zeros(update_frac*epochs)

cpt=0
epsilon_base = 0.5

for epoch in range(epochs):
    #if epoch==0:
    #    optimizer.param_groups[0]["lr"]=1e-5
    #    print("changed to:" , optimizer.param_groups[0]["lr"])
    print('epoch: ', epoch)
    _,core_new,_ = new_model
    torch.save(core_new.state_dict(), ckpts_path+"last.pth")
    for i, data_batch in enumerate(orig_loader, 0):
        optimizer.zero_grad();
        orig_batch,_ = data_batch
        batch_size = orig_batch.shape[0]
        
        orig_batch = orig_batch.transpose(1,3).transpose(2,1).contiguous()*255
        orig_batch = orig_batch.to(parser.device)
        
        norm_ima = (orig_batch).view(batch_size,-1).norm(dim=1)
        #labouls=labouls[norm_ima!=0]
        orig_batch=orig_batch[norm_ima!=0]
        true_orig_batch = orig_batch
        batch_size = orig_batch.shape[0]

        #labouls = None
        for k in range(pgd_steps):
            orig_batch = torch.autograd.Variable(orig_batch, requires_grad=True)

            if k==0:
                adversarial_image, nat_label, adv_label = transform_image(new_model, parser.transforms, orig_batch, epsilon, nat_lab=None)
            else:
                adversarial_image, _, _ = transform_image(new_model, parser.transforms, orig_batch, epsilon, nat_lab=nat_label, adv_lab=adv_label)
            optimizer.zero_grad();
            norm_adv = (adversarial_image).view(batch_size,-1).norm(dim=1)
            adversarial_image = adversarial_image.view(batch_size,-1)[~norm_adv.isnan()].view(-1,orig_batch.shape[1], orig_batch.shape[2], orig_batch.shape[3])
            true_orig_batch = true_orig_batch.view(batch_size,-1)[~norm_adv.isnan()].view(-1,orig_batch.shape[1], orig_batch.shape[2], orig_batch.shape[3])
            batch_size = true_orig_batch.shape[0]

            norm_adv = (adversarial_image).view(batch_size,-1).norm(dim=1)
            #print((adversarial_image-true_orig_batch).view(batch_size,-1).norm(dim=1).mean(), "norm en train")

            with torch.no_grad():
                nat_conv_nat = conv_nat(true_orig_batch)
            nat_conv_rob = conv_rob(true_orig_batch)
            adv_conv_rob = conv_rob(adversarial_image)

            sqerror_nat_mse, sqerror_adv_mse =get_losses(nat_conv_nat, nat_conv_rob, adv_conv_rob, true_orig_batch, adversarial_image)
            
            sqerror =  sqerror_adv_mse.mean() +sqerror_nat_mse.mean()

            if sqerror.isnan():
                print('"exiting because nan')
                exit()                   
            sqerror.backward()
            optimizer.step()

            """MISE A JOUR DU MODELE D ATTAQUE"""
            new_model = update_model(conv_rob, classifier, preprocess, softmax)

            orig_batch=adversarial_image.detach()
            optimizer.zero_grad();
            
        if i%((len(orig_loader)+4)//update_frac)==0:
            """VALIDATION"""
            for j, data_batch in enumerate(valid_loader, 0):
                optimizer.zero_grad();
                orig_batch, _, _ = data_batch

                batch_size = orig_batch.shape[0]
                norm_ima = (orig_batch).view(batch_size,-1).norm(dim=1)
                orig_batch=orig_batch[norm_ima!=0]
                true_orig_batch = orig_batch
                batch_size = orig_batch.shape[0]

                for h in range(pgd_steps):
                    orig_batch = torch.autograd.Variable(orig_batch, requires_grad=True)
                    if h==0:
                        adversarial_image, nat_label, adv_label = transform_image(new_model, parser.transforms, orig_batch, epsilon, nat_lab=None)
                    else:
                        adversarial_image, _, _ = transform_image(new_model, parser.transforms, orig_batch, epsilon, nat_lab=nat_label, adv_lab=adv_label)
                    
                    optimizer.zero_grad();
                    norm_adv = (adversarial_image).view(batch_size,-1).norm(dim=1)
                    adversarial_image = adversarial_image.view(batch_size,-1)[~norm_adv.isnan()].view(-1,orig_batch.shape[1], orig_batch.shape[2], orig_batch.shape[3])
                    true_orig_batch = true_orig_batch.view(batch_size,-1)[~norm_adv.isnan()].view(-1,orig_batch.shape[1], orig_batch.shape[2], orig_batch.shape[3])
                    #print((adversarial_image-true_orig_batch).view(batch_size,-1).norm(dim=1).mean(), "norm en eval")
                    batch_size = true_orig_batch.shape[0]
                    orig_batch = adversarial_image.detach()

                with torch.no_grad():
                    nat_conv_nat = conv_nat(true_orig_batch)
                    nat_conv_rob = conv_rob(true_orig_batch)
                    adv_conv_rob = conv_rob(adversarial_image)

                    sqerror_nat_mse_eval, sqerror_adv_mse_eval =get_losses(nat_conv_nat, nat_conv_rob, adv_conv_rob, true_orig_batch, adversarial_image)

            """ACCURACY"""
            accuracy_step = measure_accuracy(accuracy_loader, new_model)

            """SAVING LOSSES"""
            train_loss_adv[cpt] = sqerror_adv_mse.mean()
            train_loss_nat[cpt] = sqerror_nat_mse.mean()
            eval_loss_adv[cpt] = sqerror_adv_mse_eval.mean()
            eval_loss_nat[cpt] = sqerror_nat_mse_eval.mean()
            accuracy_rob[cpt] = accuracy_step

            print("train adv, nat: ", sqerror_adv_mse.mean().detach(), sqerror_nat_mse.mean().detach())
            print("eval adv, nat, rob: ", sqerror_adv_mse_eval.mean(), sqerror_nat_mse_eval.mean())
            print("accuracy: ", accuracy_step)

            np.save(measures_path+'eval_adv.npy',eval_loss_adv)
            np.save(measures_path+'eval_nat.npy',eval_loss_nat)
            np.save(measures_path+'accuracy.npy',accuracy_rob)
            np.save(measures_path+'train_adv.npy',train_loss_adv)
            np.save(measures_path+'train_nat.npy',train_loss_nat)
            cpt+=1


        """SAVING MODEL"""
        if i%500==0:
            _,core_new,_ = new_model
            torch.save(core_new.state_dict(), ckpts_path+"last.pth")