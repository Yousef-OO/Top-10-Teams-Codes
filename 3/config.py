# -*- coding: utf-8 -*-
#s
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Config:


    # 预处理flag
    preprocess = True           # 预处理

    loss = True

    brady_rule = True           # 心动过缓
    unet = False                 # unet来修正PVC
    qrs_rule = False            # rule_based来修正QRS

    prob_revise = True

    # databases information
    # train_data = 'train_all_exclude_3_500hz_re.pth'
    train_data = 'data_offline_relabel.pth'
    # train_data = 'train_all_exclude_3_500hz_relabel.pth'

    # PVC(unet)模型的权重文件
    best_cp_unet = 'best_cp_5000.pt'

    #数据长度
    seq_length=5000


    weight_A = 0.5
    weight_B = 1 - weight_A

    # 使用8导联的索引
    lead_nums_flag = True
    lead_index = [0, 1, 6, 7, 8, 9, 10, 11]

    # 定制阈值
    two_threshold = False
    thresh_H_list = [1, 0, 2, 3, 5, 7, 9, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23, 24, 25]
    threshold_H = 0.27

    thresh_L_list = [6, 4, 8, 10, 15, 16, 19, 26]
    threshold_L = 0.27

    # 模型融合
    ensemble = True
    best_w_15K_8L = 'best_w_15K_8L.pth'
    
    #label的类别数
    num_classes = 27

    #训练的模型名称
    model_name = 'se_resnet34'

    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [13, 20, 30]

    #训练时的batch大小
    batch_size = 16

    #最大训练多少个epoch
    max_epoch = 18
    add_epoch = 3

    #目标的采样长度
    target_point_num = 2048

    #初始的学习率
    lr = 1e-9

    #保存模型当前epoch的权重
    current_w = 'current_w.pth'

    #保存最佳的权重
    best_w = 'best_w.pth'

    # 学习率衰减 lr/=lr_decay
    lr_decay = 10


config = Config()
