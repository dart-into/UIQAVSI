import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from network import ThreeStreamIQA
import torch
import os

from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import cv2 as cv

import torchvision
unloader = torchvision.transforms.ToPILImage()

def GetMSCN(imdist):
    imdist=np.float64(imdist)
    mu = cv.GaussianBlur(
        imdist, (7, 7), 7 / 6, borderType=cv.BORDER_CONSTANT)
    mu_sq = mu * mu
    sigma = cv.GaussianBlur(
        imdist * imdist, (7, 7), 7 / 6, borderType=cv.BORDER_CONSTANT)
    sigma = np.sqrt(abs((sigma - mu_sq)))
    structdis = (imdist - mu) / (sigma + 1)
    #structdis=np.float64(structdis)
    structdis = Image.fromarray(structdis)
    return structdis

def CropPatches(image, patch_size=32, stride=32):
    w, h = image.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(image.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)
    return patches


def make_gradeint(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    gradxy = torch.from_numpy(gradxy)
    gradxy = gradxy.permute(2, 0, 1)
    gradxy = gradxy.cpu().clone()
    gradxy = gradxy.squeeze(0)
    gradxy = unloader(gradxy)
    return gradxy

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch UIQAVSI test demo')
    parser.add_argument("--im_path", type=str, default='/home/y222212065/TestThreeStreamIQA/data/2_Water-net.png')
                        
    parser.add_argument("--model_file", type=str, default='data/UIQAVSI_model_SAUD.pth')

    args = parser.parse_args()

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = ThreeStreamIQA().to(device)
    model.load_state_dict(torch.load(args.model_file))
  
    #model.load_state_dict(torch.load(args.model_file,map_location=torch.device('cuda:7')))
   
    
    im = Image.open(args.im_path).convert('L')

    im = cv.imread(args.im_path)
    im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
    im = unloader(im)
    # im gra
    im_gra = cv.imread(args.im_path)
    im_gra = cv.cvtColor(im_gra, cv.COLOR_BGR2RGB)
    im_gra = make_gradeint(im_gra)

    # im MSCN
    im_MSCN = cv.imread(os.path.join(args.im_path))
    im_MSCN = cv.cvtColor(im_MSCN, cv.COLOR_BGR2GRAY)
    im_MSCN = GetMSCN(im_MSCN)


    patches_lab = CropPatches(im, 32, 32)
    patches_gra = CropPatches(im_gra, 32, 32)
    patches_MSCN = CropPatches(im_MSCN, 32, 32)

    model.eval()
    with torch.no_grad():
        patch_scores = model((torch.stack(patches_lab).to(device), torch.stack(patches_gra).to(device), torch.stack(patches_MSCN).to(device)))
        print(patch_scores.mean().item())
