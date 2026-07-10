# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:45

@ author: javis
'''
import os


class Config:
    # for train
    # 训练的模型名称
    model_name = 'se_resnet50'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [10, 15, 30]
    stage_lr = [1e-5, 1e-6, 1e-7]
    # stage_epoch = [10, 15, 30]
    # stage_lr = [1e-6, 1e-7, 1e-8]
    #训练时的batch大小
    batch_size = 64
    #label的类别数
    num_classes = 27
    #最大训练多少个epoch
    max_epoch = 40
    # 采样频率
    fs = 500
    # 采样时间
    ecg_time = 10
    #目标的采样长度  2048
    target_point_num = 4000
    #保存模型的文件夹
    ckpt = 'ckpt'
    #保存日志数据
    log_dir = 'logger'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 3e-4
    #保存模型当前epoch的权重
    current_w = 'current_w.pth'
    #保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10
    fine_turning = False


config = Config()
