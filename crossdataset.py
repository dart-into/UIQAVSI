
# Implemented by Tianhai Chen
# Email: tianhai_chen@njnu.edu.cn
# Date: 2023/12/5


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import os

from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import cv2 as cv
from network import TwoStreamIQA
import torchvision
unloader = torchvision.transforms.ToPILImage()

######这是跨数据库测试代码 
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
    parser = ArgumentParser(description='PyTorch UIQAVIS test demo')

    parser.add_argument("--dataset_dir", type=str, default='/home/y222212065/Database/CSIQ/All_Image/',
                        help="dataset dir.")
    parser.add_argument("--model_file", type=str, default='/home/y222212065/TestThreeStreamIQA/data/UIQAVSI_model_SAUD.pth')
    parser.add_argument("--dataset", type=str, default="CSIQ")
    parser.add_argument("--test_save_path", type=str, default='/home/y222212065/TestThreeStreamIQA/crossdata/CSIQ_labels',
                        help="save path (default: score)")
    parser.add_argument("--predict_save_path", type=str, default='/home/y222212065/TestThreeStreamIQA/crossdata/CSIQ_predict_scores',
                        help="save path (default: score)")
    print("SAUD->CSIQ")
    args = parser.parse_args()
    #加载模型
    device = torch.device("cuda:4" if torch.cuda.is_available() else "CPU")
    model = TwoStreamIQA().to(device)
    model.load_state_dict(torch.load(args.model_file))
    #加载数据
    ref_idsPath="./data/"+args.dataset+"_ref_ids.txt"
    mosPath="./data/"+args.dataset+"_mos.txt"
    im_namesPath="./data/"+args.dataset+"_im_names.txt"
     
    ref_ids=[]
    mos=[]
    im_names=[]

    #for line0 in tqdm(open(ref_idsPath, "r")):
    #        line0 = int(line0.strip())
    #        ref_ids.append(line0)
    #ref_ids = np.array(ref_ids)
     
    for line5 in tqdm(open(mosPath, "r")):
            line5 = float(line5.strip())
            mos.append(line5)
    mos = np.array(mos)

    for line1 in tqdm(open(im_namesPath, "r")):
            line1 = line1.strip()
            im_names.append(line1)
    im_names = np.array(im_names)
    
  
 #测试阶段
    model.eval()
    with torch.no_grad():
        scores=[]
        for idx in tqdm(range(len(im_names))):
            #imlab
                im = cv.imread(os.path.join(args.dataset_dir, im_names[idx]))
                im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
                im = unloader(im)

            # im gra
                im_gra = cv.imread(os.path.join(args.dataset_dir, im_names[idx]))
                im_gra = cv.cvtColor(im_gra, cv.COLOR_BGR2RGB)
                im_gra = make_gradeint(im_gra)

            # im MSCN
                im_MSCN = cv.imread(os.path.join(args.dataset_dir, im_names[idx]))
                im_MSCN = cv.cvtColor(im_MSCN, cv.COLOR_BGR2GRAY)
                im_MSCN = GetMSCN(im_MSCN)


                patches_lab = CropPatches(im, 32, 32)
                patches_gra = CropPatches(im_gra, 32, 32)
                patches_MSCN = CropPatches(im_MSCN, 32, 32)

                patch_scores = model((torch.stack(patches_lab).to(device), torch.stack(patches_gra).to(device), torch.stack(patches_MSCN).to(device)))
                score=patch_scores.mean().item()
                scores.append(score)
    scores = np.array(scores)
#计算阶段
    y_pred=scores
    y_test=mos
    np.save(args.test_save_path, y_test)
    np.save(args.predict_save_path, y_pred)
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
    print("Results: SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(
                                                                                                       SROCC,
                                                                                                       PLCC,
                                                                                                       KROCC,
                                                                                                       RMSE))
    with open('UIQAMDNCross.txt', 'a') as f:  # 设置文件对象data.txt
        print(args.model_file,file=f)    
        print("UIQAMDN: SAUD->CSIQ Results: SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f} ".format(SROCC, PLCC, KROCC, RMSE),file=f)   
        
