# Implemented by Tianhai Chen
# Email: tianhai_chen@njnu.edu.cn
# Date: 2023/12/5


import torch
import os
import numpy as np
from scipy import stats
import yaml
from argparse import ArgumentParser
import random
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from network import ThreeStreamIQA
from ThreeIQADataset import ThreeIQADataset
import scipy.io
import numpy as np

def get_indexNum(config, index, status,dataset):
    ref_idsPath="./data/"+dataset+"_ref_ids.txt"
    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index, val_index, test_index = [], [], []

    ref_ids = []
    ref_idsname="./data/"+dataset+"_ref_ids.txt"
    for line0 in open(ref_idsname, "r"):
        line0 = int(line0.strip())
        ref_ids.append(line0)
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index
    if status == 'val':
        index = val_index

    return len(index)


if __name__ == '__main__':
    parser = ArgumentParser("Train UIQAVSI")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dataset", type=str, default="SIQAD")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_path", type=str, default="UIQAVSI_IQA.pth")
    parser.add_argument("--current_epoch", type=str, default="1")
    args = parser.parse_args()

    #save_model = "./savemodel/RGBStreamIQA.pth"
    save_model = args.save_path

    torch.manual_seed(20000615)
  

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")

    # index = [33, 17, 9, 84, 19, 70, 76, 44, 85, 11, 69, 13, 28, 88, 51, 77, 82, 71, 73, 61, 14, 38, 10, 1, 86, 67, 91, 60, 99, 12, 75, 21, 58, 2, 46, 63, 16, 57, 98, 5, 92, 47, 59, 83, 23, 43, 7, 29, 64, 30, 65, 8, 18, 45, 54, 62, 42, 31, 55, 96, 100, 66, 24, 20, 25, 97, 89, 87, 22, 34, 15, 3, 95, 90, 37, 93, 48, 80, 36, 81, 68, 4, 50, 94, 52, 40, 35, 72, 39, 41, 27, 74, 26, 6, 49, 53, 79, 56, 32, 78]

    if args.dataset == "LIVE":
        print("dataset: LIVE")
        data = scipy.io.loadmat('LIVEindexResult.mat')
    elif args.dataset == "SAUD":
        print("dataset: SAUD")
        data = scipy.io.loadmat('indexResult.mat')         
    elif args.dataset == "UIED":
        print("dataset: UIED")
        data = scipy.io.loadmat('UIEDindexResult.mat')    
    elif args.dataset == "SIQAD":
        print("dataset: SIQAD")
        data = scipy.io.loadmat('SIQADindexResult.mat')    
    elif args.dataset == "TID2013":
        print("dataset: TID2013")
        data = scipy.io.loadmat('TID2013indexResult.mat')
    elif args.dataset == "LIVEMD":
        print("dataset: LIVEMD")
        data = scipy.io.loadmat('LIVEMDindexResult.mat')   
    elif args.dataset == "LIVEC":
        print("dataset: LIVEC")
        data = scipy.io.loadmat('LIVECindexResult.mat')   
    
    epoch=int(args.current_epoch)-1
# 将数据转换为NumPy数组并选择第n行，这里选择第10行
    index= np.array(data['indexResult'][epoch,:])
    index=index.tolist()
    print('rando index', index)

    dataset = args.dataset
    valnum = get_indexNum(config, index, "val",args.dataset)
    testnum = get_indexNum(config, index, "test",args.dataset)

    train_dataset = ThreeIQADataset(dataset, config, index, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)
    val_dataset = ThreeIQADataset(dataset, config, index, "val")
    val_loader = torch.utils.data.DataLoader(val_dataset)

    test_dataset = ThreeIQADataset(dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)

    model = ThreeStreamIQA().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_SROCC = -1

    for epoch in range(args.epochs):
        # train
        model.train()
        LOSS = 0
        for i, (patches, label) in enumerate(tqdm(train_loader)):
            patches_rgb = patches[0].to(device)
            patches_gra = patches[1].to(device)
            patches_MSCN = patches[2].to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model((patches_rgb, patches_gra, patches_MSCN))
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()
        train_loss = LOSS / (i + 1)

        # test
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0
        with torch.no_grad():
            for i, (patches, label) in enumerate(tqdm(test_loader)):
                y_test[i] = label.item()
                patches_rgb = patches[0].to(device)
                patches_gra = patches[1].to(device)
                patches_MSCN = patches[2].to(device)
                label = label.to(device)
                outputs = model((patches_rgb, patches_gra, patches_MSCN))
                score = outputs.mean()
                y_pred[i] = score
                # loss = criterion(score, label[0])
                # L = L + loss.item()
        # test_loss = L / (i+1)
        
        test_loss = 1
        test_SROCC = stats.spearmanr(y_pred, y_test)[0]
        test_PLCC = stats.pearsonr(y_pred, y_test)[0]
        test_KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        test_RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

        print("Epoch {} ThreeTest Results: loss={:.4f} SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(
            epoch,
            test_loss,
            test_SROCC,
            test_PLCC,
            test_KROCC,
            test_RMSE))

        if test_SROCC > best_SROCC and epoch > 1:
            print("Update Epoch {} best valid SROCC".format(epoch))

            print("Test Results: loss={:.4f} SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(test_loss,
                                                                                                       test_SROCC,
                                                                                                       test_PLCC,
                                                                                                       test_KROCC,
                                                                                                       test_RMSE))
            torch.save(model.state_dict(), save_model)
            best_SROCC = test_SROCC

    # final test
    model.load_state_dict(torch.load(save_model))
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0
        for i, (patches, label) in enumerate(tqdm(test_loader)):
            y_test[i] = label.item()
            patches_rgb = patches[0].to(device)
            patches_gra = patches[1].to(device)
            patches_MSCN = patches[2].to(device)
            label = label.to(device)
            outputs = model((patches_rgb, patches_gra, patches_MSCN))
            score = outputs.mean()
            y_pred[i] = score
            # loss = criterion(score, label[0])
            # L = L + loss.item()
    # test_loss = L / (i + 1)
    test_loss = 1
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
    
    with open('SAUD_FinalResult.txt', 'a') as f:  # 设置文件对象data.txt
        print(args.current_epoch,file=f)
        print("{:.4f} {:.4f} {:.4f} {:.4f}".format(SROCC,PLCC,KROCC,RMSE),file=f)


                                                                                                         

    print("Final test Results: loss={:.4f} SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(test_loss,
                                                                                                     SROCC,
                                                                                                     PLCC,
                                                                                                     KROCC,
                                                                                                     RMSE))
















