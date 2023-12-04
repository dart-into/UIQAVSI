import torch
import os
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import cv2 as cv
import torchvision
from tqdm import tqdm
unloader = torchvision.transforms.ToPILImage()
from scipy.signal import convolve2d



def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln

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
            #patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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







class ThreeIQADataset(Dataset):
    def __init__(self, dataset, config, index, status):
 
        im_dir = config[dataset]['im_dir']
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []

        ref_ids = []
        self.mos = []
        im_names = []

        ref_idsPath="./data/"+dataset+"_ref_ids.txt"
        mosPath="./data/"+dataset+"_mos.txt"
        im_namesPath="./data/"+dataset+"_im_names.txt"
           
        for line0 in open(ref_idsPath, "r"):
            line0 = int(line0.strip())
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)
        
        for line5 in open(mosPath, "r"):
            line5 = float(line5.strip())
            self.mos.append(line5)
        self.mos = np.array(self.mos)

        for line1 in open(im_namesPath, "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)

        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        

        self.patches = ()
        self.patches_gradient = ()
        self.patches_MSCN = ()
        self.label = []

        self.im_names = [im_names[i] for i in self.index]

        self.mos = [self.mos[i] for i in self.index]

        for idx in tqdm(range(len(self.index))):

            #print("Preprocessing Image: {}".format(self.im_names[idx]))
            #im Lab
           
            im = cv.imread(os.path.join(im_dir, self.im_names[idx]))
            im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
            im = unloader(im)

            #im gra
            im_gra = cv.imread(os.path.join(im_dir, self.im_names[idx]))
            im_gra = cv.cvtColor(im_gra, cv.COLOR_BGR2RGB)
            im_gra = make_gradeint(im_gra)
            
            
            # im MSCN
            im_MSCN=cv.imread(os.path.join(im_dir, self.im_names[idx]))
            im_MSCN = cv.cvtColor(im_MSCN, cv.COLOR_BGR2GRAY)
            im_MSCN = GetMSCN(im_MSCN)

            patches = CropPatches(im, self.patch_size, self.stride)
            patches_gradient = CropPatches(im_gra, self.patch_size, self.stride)
            patches_MSCN = CropPatches(im_MSCN, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + patches
                self.patches_gradient = self.patches_gradient + patches_gradient
                self.patches_MSCN=self.patches_MSCN+patches_MSCN
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
            else:
                self.patches = self.patches + (torch.stack(patches),)
                self.patches_gradient = self.patches_gradient + (torch.stack(patches_gradient),)
                self.patches_MSCN = self.patches_MSCN + (torch.stack(patches_MSCN),)
                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (self.patches[idx], self.patches_gradient[idx],self.patches_MSCN[idx]), (torch.Tensor([self.label[idx]]))




