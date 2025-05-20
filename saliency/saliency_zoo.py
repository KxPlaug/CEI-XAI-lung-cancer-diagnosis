from saliency.core import GuidedIG,pgd_step, Ma2Ba,FGSMGrad
from captum.attr import Saliency
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import random
from torch.autograd import Variable as V
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def guided_ig(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        # m = torch.nn.Softmax(dim=1)
        # output = m(output)
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    result =  method.GetMask(
        im, call_model_function, call_model_args, x_steps=15, x_baseline=baseline)
    return np.expand_dims(result, axis=0)

def agi(model, data, target, epsilon=0.05, max_iter=20, topk=1):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 2)), topk)
    output = model(data)
    # get the index of the max log-probability
    # init_pred = output.max(1, keepdim=True)[1]
    init_pred = output.argmax(-1)

    top_ids = selected_ids  # only for predefined ids
    # initialize the step_grad towards all target false classes
    step_grad = 0
    # num_class = 1000 # number of total classes
    for l in top_ids:
        # targeted = torch.tensor([l]).to(device)
        targeted = torch.tensor([l] * data.shape[0]).to(device)
        # if targeted.item() == init_pred.item():
        #     if l < 999:
        #         # replace it with l + 1
        #         targeted = torch.tensor([l+1]).to(device)
        #     else:
        #         # replace it with l + 1
        #         targeted = torch.tensor([l-1]).to(device)
        #     # continue # we don't want to attack to the predicted class.
        if l < 2:
            targeted[targeted == init_pred] = l + 1
        else: 
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().unsqueeze(0).detach().cpu().numpy()  # / topk
    return adv_ex


def mfaba_smooth(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=False):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = Ma2Ba(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax,early_stop=True)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def saliencymap(model,data,target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = Saliency(model)
    return saliencymap.attribute(data, target).cpu().detach().numpy()


