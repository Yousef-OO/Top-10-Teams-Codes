#!/usr/bin/env python

import os, sys
from scipy.io import loadmat

# 注：该文件可以编辑
# 但是train_12ECG_classifier(input_directory, output_directoory)函数的输入输出不可改动
# 只可以将模型及其参数保存在另一个文件

# 以下是official phase第一次提交所需的package
# 07/16/2020，gym
import torch, shutil
from config import config
import time
import models_B, utils
import numpy as np
from torch.utils.data import DataLoader
from dataset import ECGDataset
from torch import nn, optim
from tensorboard_logger import Logger
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)

# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(pre_state, new_state, is_best, model_save_dir):
    # best_f1 < val_f1
    best_w = os.path.join(model_save_dir, config.best_w)
    if is_best:
        torch.save(pre_state, best_w)
    else:
        torch.save(pre_state, best_w)

def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in tqdm(train_dataloader):
        inputs = inputs.to(device)
#        sss=inputs.cpu().detach().numpy()
#        print(sss)
#        print(np.where(np.isnan(sss)))
#         print(inputs.shape)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
#        print(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
#        print(f1)
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count

def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()        # model.eval()是保证BN用全部训练数据的均值和方差，Dropout利用到了所有网络连接
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in tqdm(val_dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
#            print(output)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            y_pre = output.view(-1).cpu().detach().numpy() > threshold
            # print(y_pre)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    return loss_meter / it_count, f1_meter / it_count

def train_12ECG_classifier(input_directory, output_directory):
    # train_model.py直接调用的是本文件
    # input_directory是训练集文件的目录
    # output_directory是模型的输出文件

    # **************************************************************************
    # ********************************训练模型***********************************
    # **************************************************************************

    # '''
    model = getattr(models_B, config.model_name)()

    pre_state = torch.load(config.current_w, map_location='cpu')  # 由GPU保存的模型加载到CPU上
    model.load_state_dict(pre_state['state_dict'])

    model = model.to(device)
    # data
    train_dataset = ECGDataset(data_path=input_directory, data_info_pth_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset  =  ECGDataset(data_path=input_directory, data_info_pth_path=config.train_data, train=False)
    val_dataloader  =  DataLoader(val_dataset, batch_size=config.batch_size)

    print("train_datasize: ", len(train_dataset))
    print("val_datasize: ", len(val_dataset))

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    if config.loss:
        criterion = utils.signLoss()
    else:
        criterion = utils.WeightedMultilabel(w)

    # 模型保存文件夹:  ./ckpt_all_exclude_3_500hz_re/se_resnet34
    model_save_dir = output_directory

    # if args.ex: model_save_dir += args.ex
    start_epoch = pre_state['epoch']
    val_loss = pre_state['loss']
    stage = pre_state['stage']
    best_f1 = pre_state['f1']
    # lr = pre_state['lr']
    print('#epoch:%02d stage:%d val_loss:%0.3e val_f1:%.3f \n'
          % (start_epoch, stage, val_loss, best_f1))


    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    # =========>开始训练<=========
    for epoch in range(start_epoch, start_epoch + config.add_epoch):
        epoch += 1
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=500)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)

        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': config.lr,
                 'stage': stage}
        save_ckpt(pre_state, state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)

        if epoch in config.stage_epoch:
            stage += 1
            lr = config.lr/config.lr_decay       # 学习率衰减
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)


    # for epoch in range(config.max_epoch):
    #     since = time.time()
    #     train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=50)
    #     print('#epoch:%02d train_loss:%.3e train_f1:%.3f  time:%s\n'
    #           % (epoch, train_loss, train_f1, utils.print_time_cost(since)))
    #     logger.log_value('train_loss', train_loss, step=epoch)
    #     logger.log_value('train_f1', train_f1, step=epoch)
    #
    # state = {"state_dict": model.state_dict()}
    # torch.save(state, os.path.join(output_directory, config.best_w))

    # '''
    # **************************************************************************
    # ********************************离线模型***********************************
    # **************************************************************************
    '''
    # 加载模型权重文件
    model_save_dir = config.best_w # model_save_dir = './ckpt_all_exclude_3_500hz_re/se_resnet34_202007031745/best_w.pth'  # config.model_save_dir
    model = getattr(models_A, config.model_name)()        # 加载模型

    model.load_state_dict(torch.load(model_save_dir, map_location='cpu')['state_dict'])

    # 保存模型
    state = {"state_dict": model.state_dict()}
    filename = os.path.join(output_directory, 'finalized_model.pth')
    torch.save(state, filename)
    '''
    # 以下是官网代码
    # filename = os.path.join(output_directory, 'finalized_model.sav')
    # joblib.dump(model, filename, protocol=0)

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)
