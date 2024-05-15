import torch
import numpy as np
from scipy.fftpack import dct, idct
from .color_coding import rgb_to_ycbcr,ycbcr_to_rgb,rgb_to_ycbcr_grad,ycbcr_to_rgb_grad
from .quant_tables import quantization_tables


def dct2(a):
    """ Performs 2-dimensional dct """
    return dct(dct(a, axis=0, norm='ortho' ), axis=1, norm='ortho')

def idct2(a):
    """ Performs 2-dimensional idct """
    return idct(idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def dct2_8_8(image, quant_tables):
    """ Performs 2-dimensional dct over a whole array"""
    imsize = image.shape
    dct_array = np.zeros(imsize)
    for batch_nb in range(image.shape[0]):
        for channel in range(imsize[3]):
            for i in np.r_[:imsize[1]:8]:
                for j in np.r_[:imsize[2]:8]:
                    dct_array[batch_nb, i:(i+8),j:(j+8), channel] = dct2(image[batch_nb, i:(i+8),j:(j+8), channel])
                    dct_array[batch_nb, i:(i+8),j:(j+8), channel] = dct_array[batch_nb, i:(i+8),j:(j+8), channel]/quant_tables[:,:,channel]
    return dct_array

def idct2_8_8(dct_array, quant_tables):
    """ Performs 2-dimensional idct over a whole array"""
    idct_array = np.zeros(dct_array.shape)
    for batch_im in range(dct_array.shape[0]):
        for channel in range(dct_array.shape[3]):
            for i in np.r_[:dct_array.shape[1]:8]:
                for j in np.r_[:dct_array.shape[2]:8]:
                    idct_array[batch_im, i:(i+8),j:(j+8), channel] = dct_array[batch_im, i:(i+8),j:(j+8), channel]*quant_tables[:,:,channel]
                    idct_array[batch_im, i:(i+8),j:(j+8), channel] = idct2(idct_array[batch_im, i:(i+8),j:(j+8), channel] )
    return idct_array

def spatial_to_jpeg_grad(spatial_tensor, quant_tables):
    """ Converts a tensor of spatial differential quantities to JPEG domain """
    jfif_tensor = rgb_to_ycbcr_grad(spatial_tensor)
    jfif_tensor = jfif_tensor
    jpeg_array = dct2_8_8(jfif_tensor.numpy(), quant_tables)
    jpeg_tensor = torch.tensor(jpeg_array, device=spatial_tensor.device)
    return(jpeg_tensor.float())

def jpeg_to_spatial_grad(jpeg_tensor, quant_tables):
    """ Converts a tensor of JPEG differential quantities to spatial domain """
    jpeg_array = jpeg_tensor.detach().cpu().numpy()
    jfif_tensor = idct2_8_8(jpeg_array, quant_tables)
    jfif_tensor = jfif_tensor
    spatial_tensor = ycbcr_to_rgb_grad(torch.tensor(jfif_tensor)).to(jpeg_tensor.device)
    return(spatial_tensor.float())

def jpeg_to_spatial(jpeg_tensor, quant_tables, quantization=True):
    """ Converts a tensor of JPEG images to spatial domain """
    jpeg_array = jpeg_tensor.detach().cpu().numpy()
    jfif_tensor = idct2_8_8(jpeg_array, quant_tables)
    jfif_tensor = jfif_tensor+128
    spatial_tensor = ycbcr_to_rgb(torch.tensor(jfif_tensor))

    if quantization:
        spatial_tensor = spatial_tensor.round()
        spatial_tensor = torch.clamp(spatial_tensor,0,255)
    spatial_tensor = spatial_tensor.to(jpeg_tensor.device)
    return(spatial_tensor.float())

def spatial_to_jpeg(spatial_tensor, quant_tables, quantization=True):
    """ Converts a tensor of spatial images to JPEG domain """
    jfif_tensor = rgb_to_ycbcr(spatial_tensor)
    jfif_tensor = jfif_tensor.clamp(0,255)
    jfif_tensor = jfif_tensor-128
    jpeg_array = dct2_8_8(jfif_tensor.numpy(), quant_tables)

    if quantization:
        jpeg_array = np.around(jpeg_array)
    jpeg_tensor = torch.tensor(jpeg_array, device=spatial_tensor.device)
    return(jpeg_tensor.float())

def switch_to_dct(adversarial_images_png, orig_images_png, quantization_tables):
    """ This function converts the needed quantities to JPEG domain """
    origins_dct = spatial_to_jpeg(orig_images_png, quantization_tables, quantization=False).to(orig_images_png.device)
    adversarial_dct = spatial_to_jpeg(adversarial_images_png, quantization_tables, quantization=False).to(orig_images_png.device)
    perturbations_dct = adversarial_dct-origins_dct
    return(origins_dct, perturbations_dct, adversarial_dct)

def best_quantization_dct(quantizer, adversarial_images, images_orig, init_labels):
    """
    This is the main function for the steganographic quantization with the Hill method.
    adversarial_images = Tensor of the already attacked images
    images_orig = Tensor of the original images before attack
    init_labels = Ground Truth labels for images
    """
    batch_size = adversarial_images.shape[0]

    quant_tables = quantization_tables(quantizer.jpeg_quant)
    images_orig_dct, perturbations_adv_dct, adversarial_images_dct = switch_to_dct(adversarial_images, images_orig,quant_tables)
    perturbation_norm = perturbations_adv_dct.view(batch_size,-1).norm(dim=1)
    grads, adv_labels = quantizer.find_grads(init_labels, adversarial_images)

    quantized,_ = quantize(quantizer, torch.zeros(batch_size, device=quantizer.pytorch_device), perturbations_adv_dct, grads, quant_tables, adversarial_images_dct)

    adversarial_quantized =  images_orig_dct + quantized
    adversarial_quantized_spatial = jpeg_to_spatial(adversarial_quantized, quant_tables).to(quantizer.pytorch_device)
    adversarial_quantized_spatial = torch.clamp(adversarial_quantized_spatial,0,255)

    loss_th_base =  quantizer.classif_loss(init_labels, adv_labels, adversarial_quantized_spatial)

    #This little trick creates a little perturbation if JPEG compression reformed an originally misclassified image
    perturbations_adv_dct = perturbations_adv_dct*(perturbation_norm.view(batch_size,1,1,1)!=0)-100*spatial_to_jpeg_grad(grads,quant_tables)*(perturbation_norm.view(batch_size,1,1,1)==0)
    adversarial_images_dct = images_orig_dct+perturbations_adv_dct

    #This term is <u,g> withdrawn to the loss to find lambda critic
    perturbation_term = (grads*(adversarial_images-images_orig)).view(batch_size,-1).sum(axis=1)

    lambdas_critic = find_lambdas_critic(grads, loss_th_base-perturbation_term)

    best_adversarials = lambda_binary_search_stega(quantizer, lambdas_critic, perturbations_adv_dct, grads, images_orig_dct, quant_tables, init_labels, adv_labels, adversarial_images_dct)
    return(best_adversarials)


def find_lambdas_critic(gradients, loss_th_0):
    """
    This function calculates all the possibles values of Lambda for which the quantization of a pixel
    will swap from minimum distortion to maximum distortion
    """
    batch_size = gradients.shape[0]
    grad_norm_squared = (gradients.view(batch_size,-1).norm(dim=1))**2

    lambda_critic = 2*(loss_th_0)/grad_norm_squared
    lambda_critic = torch.max(lambda_critic, torch.ones(lambda_critic.shape, device=lambda_critic.device))

    return(lambda_critic)

def quantize(quantizer, lambada, unquantized_perturbations, gradients, quant_mat, adv_dct):
    """
    This function creates the quantized perturbation by quantizing with respect to the given value of Lambda
    (called "lambada" for Python's sake ...)
    """
    maximum_distortion = quantizer.max_dist
    initial_shape = unquantized_perturbations.shape
    batch_size = initial_shape[0]
    quantization = torch.tensor(quant_mat).repeat(224//8,224//8,1).unsqueeze(0).repeat(batch_size,1,1,1).to(quantizer.pytorch_device).float()
    solutions_spatial =  - lambada.view(batch_size,1,1,1)*gradients/2 - jpeg_to_spatial_grad(unquantized_perturbations, quant_mat)
    solutions_jpeg = spatial_to_jpeg_grad(solutions_spatial.detach().cpu(), quant_mat).to(quantizer.pytorch_device).float()

    if quantizer.max_dist%2==0:
        max_pert = (unquantized_perturbations.round() + quantizer.max_dist/2)
        min_pert = (unquantized_perturbations.round() - quantizer.max_dist/2)
    else:
        max_pert = (unquantized_perturbations.ceil() + (quantizer.max_dist-1)/2)
        min_pert = (unquantized_perturbations.floor() - (quantizer.max_dist-1)/2)

    if quantizer.max_dist%2==0:
        max_pert = (adv_dct.round()-adv_dct + quantizer.max_dist/2)
        min_pert = (adv_dct.round()-adv_dct - quantizer.max_dist/2)
    else:
        max_pert = (adv_dct.ceil()-adv_dct + (quantizer.max_dist-1)/2)
        min_pert = (adv_dct.floor()-adv_dct - (quantizer.max_dist-1)/2)

    sol_min = torch.min(max_pert.float(), solutions_jpeg)
    sol_max = torch.max(min_pert.float(), sol_min)
    sol_max = (sol_max+adv_dct).float().round()-adv_dct

    return(sol_max, solutions_jpeg)

def lambda_binary_search_stega(quantizer, lambdas, perturbation_dct, grads_dct, images_orig_dct, quantization, init_label, adv_label, adv_dct):
    maximum_distortion = quantizer.max_dist
    batch_size = perturbation_dct.shape[0]

    perturbation_quantized,_ = quantize(quantizer, 10*lambdas, perturbation_dct, grads_dct, quantization, adv_dct)
    #
    adversarial_quantized =  images_orig_dct.clone().round() + perturbation_quantized*0
    adversarial_quantized_spatial = jpeg_to_spatial(adversarial_quantized, quantization).to(quantizer.pytorch_device)
    adversarial_quantized_spatial = torch.clamp(adversarial_quantized_spatial,0,255)

    new_loss = quantizer.classif_loss(init_label, adv_label, adversarial_quantized_spatial, real=True)
    best_adversarial = adversarial_quantized

    #Our simulated JPEG decompression (and compression) is not 100% equal to the ones performed by PILLOW
    #We add a little confidence margin slightly negative to enforce adversariality
    confidence_margin = -0.05
    adversarial = (new_loss<=confidence_margin).float().view(batch_size,1,1,1)
    already_adversarial = adversarial

    for lambda_search_step in range(quantizer.binary_search_steps+1):

            adversarial = (new_loss<=confidence_margin).float().view(batch_size,1,1,1)
            best_adversarial = adversarial_quantized*adversarial+best_adversarial*(1-adversarial)

            #Sampling value of lambda
            lambda_values = lambdas*10**(2* (quantizer.binary_search_steps-2*lambda_search_step)/ (quantizer.binary_search_steps))

            perturbation_quantized,_ = quantize(quantizer, lambda_values, perturbation_dct, grads_dct, quantization, adv_dct)
            adversarial_quantized = perturbation_quantized + adv_dct

            adversarial_quantized_spatial = jpeg_to_spatial(adversarial_quantized, quantization).to(quantizer.pytorch_device)
            adversarial_quantized_spatial = torch.clamp(adversarial_quantized_spatial,0,255)
            new_loss = quantizer.classif_loss(init_label, adv_label, adversarial_quantized_spatial, real=True)

    best_adversarial = already_adversarial*(images_orig_dct.round())+(1-already_adversarial)*best_adversarial
    return best_adversarial
