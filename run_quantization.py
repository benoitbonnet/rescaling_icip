import torch
import numpy as np
import sys
import os
from attacks.attack_gen import attack_generator
from quantizers.quantizer import Quantizer
from data_utils.data_utils import  AdversarialDataset,AttackParser,Preprocessing_Layer, load_model
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


measures_path = parser.save_path+'/measures/'

orig_set = AdversarialDataset(parser, img_size=224)
orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=parser.batch_size, shuffle=False)

results_cpt=0
results_array = np.zeros(len(orig_set))
results_array_unquant = np.zeros(len(orig_set))
results_array_quant = np.zeros(len(orig_set))
print(len(orig_loader), " batches")


cpt=0
folders = glob.glob(parser.save_path+'/*imagenet*')
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
        images_path = folder+'/images/'+ckpt[len(ckpt_path):]
        if not os.path.isdir(images_path):
            os.mkdir(images_path)
        print('saving images in: ',images_path)
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
                quantized_adv = quantizer.quantize_samples(adversarial_image, orig_batch, initial_label)

                with torch.no_grad():
                    pred_orig= model(orig_batch)
                    pred_adv = model(adversarial_image)
                    pred_quant = model(quantized_adv)

                orig_label = torch.argmax(pred_orig,axis=-1)
                adv_label = torch.argmax(pred_adv,axis=-1)
                quant_label = torch.argmax(pred_quant,axis=-1)

                is_adv_unquant = (adv_label!=initial_label).cpu()
                is_adv_quant = (quant_label!=initial_label).cpu()
                was_adv = (orig_label!=initial_label).cpu()

                quant_success += is_adv_quant.sum()
                adv_success += is_adv_unquant.sum()
                misclass += was_adv.sum()

                disto_unquantized = (adversarial_image-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794
                disto_quantized = (quantized_adv-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794

                results_array_unquant[results_cpt:results_cpt+batch_size] = is_adv_unquant*disto_unquantized + (~is_adv_unquant)*1e6
                results_array_quant[results_cpt:results_cpt+batch_size] = is_adv_quant*disto_quantized + (~is_adv_quant)*1e6

                print('distos : ', disto_quantized.mean(), disto_unquantized.mean())
                print('advs : ', is_adv_quant.float().mean(), is_adv_unquant.float().mean())
                results_cpt+=batch_size
                for batch_image in range(quantized_adv.shape[0]):
                #Uncomment following line to save only adversarial images
                    if initial_label[batch_image].item()!=quant_label[batch_image].item():
                        #Uncomment following line to ignore already misclassified images
                        if initial_label[batch_image].item()==orig_label[batch_image].item():
                            #print(initial_label[batch_image], quant_label[batch_image], orig_label[batch_image])
                            im = Image.fromarray(quantized_adv[batch_image].cpu().numpy().astype(np.uint8))
                            im.save(images_path+"/{}.png".format(image_nbs[0][batch_image]))
            print('attack success: ',adv_success , 'quant success: ', quant_success,'misclass: ', misclass)  
                    
            if ckpt!='none':
                np.save('{}{}.npy'.format(measures_path,ckpt[len(ckpt_path):]), results_array_unquant)
                np.save('{}{}_q.npy'.format(measures_path,ckpt[len(ckpt_path):]), results_array_quant)
            else:
                np.save('{}/{}.npy'.format(measures_path,ckpt), results_array_unquant)
                np.save('{}/{}_q.npy'.format(measures_path, ckpt), results_array_quant)
                
                
        #     with torch.no_grad():
        #         pred_orig= model(orig_batch)
        #         pred_adv = model(adversarial_image)
        #         pred_quant = model(quantized_adv)

        #     orig_label = torch.argmax(pred_orig,axis=-1)
        #     adv_label = torch.argmax(pred_adv,axis=-1)
        #     quant_label = torch.argmax(pred_quant,axis=-1)

        #     is_adv_unquant = (adv_label!=initial_label).cpu()
        #     is_adv_quant = (quant_label!=initial_label).cpu()

        #     disto_unquantized = (adversarial_image-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794
        #     disto_quantized = (quantized_adv-orig_batch).view(batch_size,-1).norm(dim=1).cpu().numpy()/387.9794
        #     # print(is_adv_quant, is_adv_unquant)
        #     print(disto_unquantized.mean(), disto_quantized.mean())
        #     # exit()
        #     #Saves distortion measures. Unsuccesful attacks are saved with a distortion of 1e6 by default
        #     results_array[results_cpt:results_cpt+batch_size] = is_adv_quant*disto_quantized + (~is_adv_quant)*1e6
        #     results_array_unquant[results_cpt:results_cpt+batch_size] = is_adv_unquant*disto_unquantized + (~is_adv_unquant)*1e6

        #     for batch_image in range(quantized_adv.shape[0]):
        #         #Uncomment following line to save only adversarial images
        #         #if initial_label[batch_image].item()!=quant_label[batch_image].item():
        #             #Uncomment following line to ignore already misclassified images
        #             #if initial_label[batch_image].item()==orig_label[batch_image].item():
        #                  if parser.jpeg_quality==0:
        #                      """ Save spatial images """
        #                      im = Image.fromarray(quantized_adv[batch_image].cpu().numpy().astype(np.uint8))
        #                      im.save(images_path+"/{}/{}/{}.png".format(attack_name, model_name,image_nbs[0][batch_image]))
        #                  else:
        #                      """ JPEG images cannot be saved directly. The original image is saved as JPEG and coefficients in a separate .npy file
        #                      The script build_jpeg.py swaps coefficients of the original image with the computed ones"""
        #                      quantized_images = orig_batch[batch_image]#[:, :, [2,1,0]]
        #                      im = Image.fromarray(quantized_images.cpu().numpy().astype(np.uint8))
        #                      im.save(images_path+"{}/{}/{}.jpg".format(attack_name, model_name,image_nbs[0][batch_image]), quality=parser.jpeg_quality, subsampling=0)
        #                      np.save(images_path+"{}/{}/{}.npy".format(attack_name, model_name,image_nbs[0][batch_image]),np.float32(quantized_adv[batch_image].cpu()) )

        #     results_cpt+=batch_size

        # np.save('{}{}_{}_quant.npy'.format(measures_path, model_name, attack_name), results_array)
        # np.save('{}{}_{}_unquant.npy'.format(measures_path, model_name, attack_name), results_array_unquant)
