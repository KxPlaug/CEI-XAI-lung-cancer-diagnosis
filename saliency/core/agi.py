import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image

import json


def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    grad_lab_norm = torch.norm(data_grad_lab, p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta
    # return perturbed_image, delta


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        
        # select the false class label
        # output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

        if len(leave_index) == 0:
            break
        
    return c_delta, perturbed_image

