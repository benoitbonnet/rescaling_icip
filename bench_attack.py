import torch
import numpy as np
import sys
import os
from attacks.attack_gen import attack_generator
from quantizers.quantizer import Quantizer
from data_utils.data_utils import  AdversarialDataset,AttackParser,Preprocessing_Layer
import argparse
import foolbox as fb
import glob
from PIL import Image
import timm
import torch.nn as nn
torch.cuda.empty_cache()

parser = AttackParser(argparse.ArgumentParser())

parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/nips17/'
parser.labels_path = '/nfs/nas4/bbonnet/bbonnet/datasets/labels/nips17.csv'

# parser.labels_path = ('/nfs/nas4/bbonnet/bbonnet/datasets/labels/imagenet_2012_val/processed.csv')
# parser.input_path = '/nfs/nas4/bbonnet/bbonnet/datasets/images/test/'


def load_model(arg_parser, ckpt_path):
    device = arg_parser.device
    model_name = arg_parser.models[0]
    model_init = timm.create_model(model_name, pretrained=True)
    preprocess_layer = Preprocessing_Layer(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225], im_size=224)

    softmax = nn.Softmax(dim=1)
    if model_name=='efficientnet_b0':
        """efficientnet"""
        if ckpt_path!='none':
            state_dict = torch.load(ckpt_path)
            checkpoint = state_dict#['model']
            new_checkpoint = {}
            for check_key in checkpoint.keys():
                new_checkpoint[check_key[2:]]=checkpoint[check_key]
            model_init.load_state_dict(new_checkpoint)

    elif model_name=='resnet18':
        """resnet18"""
        if ckpt_path!='none':
            try:
                state_dict = torch.load(ckpt_path)
                checkpoint = state_dict#['model']

                # new_checkpoint = checkpoint#{}
                
                new_checkpoint = {}
                for check_key in checkpoint.keys():
                    new_checkpoint[check_key[2:]]=checkpoint[check_key]
                model_init.load_state_dict(new_checkpoint)
            except:
                state_dict = torch.load(ckpt_path)
                checkpoint = state_dict#['model']

                new_checkpoint = checkpoint#{}
                
                model_init.load_state_dict(new_checkpoint)
        
    elif model_name=='resnet50':
        if ckpt_path!='none':
            try:
                state_dict = torch.load(ckpt_path)
                checkpoint = state_dict['model']
                prefix = 'module.attacker.model.'
                new_checkpoint = {}
                for check_key in checkpoint.keys():
                    if prefix in check_key:
                        new_checkpoint[check_key[len(prefix):]]=checkpoint[check_key]

                model_init.load_state_dict(new_checkpoint)
            except:
                state_dict = torch.load(ckpt_path)
                checkpoint = state_dict#['model']

                new_checkpoint = checkpoint#{}
                
                model_init.load_state_dict(new_checkpoint)

        # if ckpt_path!='none':
        #     state_dict = torch.load(ckpt_path)
        #     checkpoint = state_dict['model']
        #     new_checkpoint = {}
        #     # print(model_init.state_dict.keys())
        #     # print(checkpoint['model'].keys())
        #     # exit()
        #     for check_key in checkpoint.keys():
        #         if 'module.attacker.model.' in check_key: 
        #             new_checkpoint[check_key[len('module.attacker.model.'):]]=checkpoint[check_key]
        #     model_init.load_state_dict(new_checkpoint)

    model_softmax = nn.Sequential(preprocess_layer, model_init, softmax)
    model_softmax.eval();
    model_softmax.to(device)
    return(model_softmax,device)

measures_path = parser.save_path+'/measures/'

orig_set = AdversarialDataset(parser, img_size=224)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=parser.batch_size, shuffle=False)

results_cpt=0
results_array = np.zeros(len(orig_set))
results_array_unquant = np.zeros(len(orig_set))
results_array_quant = np.zeros(len(orig_set))
print(len(orig_loader), " batches")


def correlate(a,b):
    return((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)) )


cpt=0
folders = glob.glob(parser.save_path+'/*sin*')
#folders = glob.glob(parser.save_path)
print('attacking models: ', folders)
for folder in folders:
    measures_name = folder[len(parser.save_path):]

    ckpt_path = folder+'/ckpts/'
    ckpts = glob.glob(ckpt_path+'/*')
    # if cpt==0:
    #     ckpts.append('none')
    #     for a in timm.list_models(pretrained=True):
    #         print(a)
    cpt+=1
    # print(ckpts,ckpt_path, 'saving as {}{}.npy'.format(measures_path, measures_name))
    for ckpt in ckpts:
        print('loading', ckpt)
        print(folder[len(parser.save_path):])
        print('saving measures in: ','{}{}.npy'.format(measures_path,folder[len(parser.save_path):]))
        model, device = load_model(parser, ckpt)
        quantizer = Quantizer(model, parser, number_classes=1000, binary_search_steps=20)
        fmodel = fb.PyTorchModel(model, bounds=[0,255])

        quant_success = 0
        adv_success = 0
        misclass = 0
        for attack_cpt, attack_name in enumerate(parser.attacks):

            results_cpt=0
            results_array = np.zeros(len(orig_set))
            results_array_unquant = np.zeros(len(orig_set))

            if parser.attack_type=='foolbox':
                attack = getattr(fb.attacks, attack_name, fb.attacks.FGSM)(steps=100, stepsize=1e-3)
            else:
                attack = attack_generator(parser,attack_cpt)

            for i, data_batch in enumerate(orig_loader, 0):
                orig_batch, initial_label, image_nbs = data_batch
                criterion = fb.criteria.Misclassification(initial_label)
                batch_size = orig_batch.shape[0]

                #Keep track of iterations
                # if (i%10)==0:
                #     print(i*batch_size)

                adversarial_image,_,_ =  attack(fmodel, orig_batch, criterion ,epsilons = parser.epsilon)

                pred_orig= model(orig_batch)
                orig_label = torch.argmax(pred_orig,axis=-1)

                pred_adv = model(adversarial_image)
                adv_label = torch.argmax(pred_adv,axis=-1)
                is_adv_unquant = (adv_label!=initial_label).cpu()
                was_adv = (orig_label!=initial_label).cpu()

                adv_success += is_adv_unquant.sum()
                misclass += was_adv.sum()

                disto_unquantized = (adversarial_image-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794

                results_array_unquant[results_cpt:results_cpt+batch_size] = is_adv_unquant*disto_unquantized + (~is_adv_unquant)*1e6

                results_cpt+=batch_size

            print('attack success: ',adv_success , 'quant success: ', quant_success,'misclass: ', misclass)          
            np.save('{}{}.npy'.format(measures_path,folder[len(parser.save_path):]), results_array_unquant)
