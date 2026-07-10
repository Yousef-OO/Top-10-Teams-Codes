# -*- coding: utf-8 -*-
'''
@time: 2020/6/1
数据预处理：
    1.切割数据长度为10s的片段
    2.划分五折交叉验证
    3.生成arrythmia.txt，存放各种疾病的代号
    4.生成label.txt，每行数据格式为  交叉验证折号  文件名称 label
    5. 建立每折的测试集
@ author: Gui
'''

import numpy as np, os, sys, torch
from scipy.io import loadmat
from config import config


def cut_data_folds(input_directory, output_directory, folds):
    """
    切割数据长度，同时设置fold数
    :param input_directory: 输入文件路径
    :param output_directory: 输出文件路径
    :return:
    """
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
    num_files = len(input_files)
    num_val = 0.1 * num_files

    folds_count = [0 for f in range(folds)]

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)

        # 从数量最小的fold中随机选择一个
        min_count = min(folds_count)
        np.random.seed(i)
        fold = np.random.choice([f for f, count in enumerate(folds_count) if count == min_count])
        folds_count[fold] += 1

        data = data.transpose()
        ecg_len = data.shape[0]
        if ecg_len % 5000 == 0:
            cut_pos = int(ecg_len / 5000)
        else:
            cut_pos = int(ecg_len / 5000)
            if cut_pos == 0:
                cut_pos = 1
            else:
                cut_pos += 1
        for i in range(cut_pos):
            cut_start = 5000 * i
            cut_end = cut_start + 5000
            ecg_tmp = np.zeros(shape=(1, 5000, 12))

            if cut_end > ecg_len and ecg_len < 5000:
                sig = data[cut_start: ecg_len, :]
                ecg_tmp = Pad_2d(sig, 5000 - (ecg_len - cut_start))
                print("Padding")
            elif cut_end > ecg_len and ecg_len > 5000:
                ecg_tmp = data[ecg_len - 5000: ecg_len, :]
            else:
                ecg_tmp = data[cut_start: cut_end, :]

            if not os.path.exists(output_directory):
                os.mkdir(output_directory)

            output_file = os.path.join(output_directory, f.split('.')[0] + '_{}.txt'.format(i))
            np.savetxt(output_file, ecg_tmp, fmt='%d')

            output_label = os.path.join(output_directory, 'label.txt')
            with open(output_label, 'a') as e:
                e.write(str(fold) + " " + f.split('.')[0] + '_{}'.format(i) + header_data[-4].split(':')[1])
    print(folds_count)


def write_classes(input_directory, output_directory, files):
    """
    得到全部类别，写入arrythmia.txt文件
    :param input_directory: path
    :param output_diretory: path
    :return:
    """

    classes = get_classes()
    output_label = os.path.join(output_directory, 'arrythmia.txt')
    with open(output_label, 'a') as e:
        for i in classes:
            e.write(i + '\n')


def split_data_fold_27(input_directory, file2fold, folds, delete_record):
    '''
    按照fold划分数据集，避免属于同一记录的片段分别划分给训练集和测试集
    :param file2fold:
    :param folds:将fold值为folds的数据划分给测试集
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(input_directory))
    val = set()
    train = set()
    for file, fold_idx in file2fold.items():
        if file not in delete_record:
            if int(fold_idx) == folds:
                val.add(file)
            else:
                train.add(file)
    # train = data.difference(val)
    print(len(train))
    print(len(val))
    return list(train), list(val)


def get_classes():
    classes = set()
    score = ['10370003', '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
             '164947007', '17338001', '251146004', '270492004', '284470004', '39732003', '426627000',
             '426177001', '426783006', '427084000', '427172004', '427393009', '445118002', '47665007',
             '59118001', '59931005', '63593006', '698252002', '713426002', '713427006']
    for a in score:
        classes.add(a)
    return sorted(classes)


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    file2fold = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split(' ')
        id = arr[1] + '.txt'
        labels = [name2idx[name] for name in arr[2].split(',')]
        fold = arr[0]
        # print(id, labels)
        file2index[id] = labels
        file2fold[id] = fold
    return file2index, file2fold


def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name) if len(name) > 0}
    return name2indx


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def Pad_2d(sig, target_length):
    """
    # 对小于target_length的信号进行补零
    :param sig: 2-D numpy Array，输入信号
    :param target_length: int，目标长度
    :return:  2-D numpy Array，输出补零后的信号
    """
    if target_length > 0:
        sig = np.concatenate((sig, np.zeros(shape=(target_length, 12))))
    return sig


def file2index_27(path, name2idx):
    '''
    获取文件id对应的标签类别,仅保留计分的27类标签，删除其余标签，若删除后标签数为0则删除该记录
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    score = ['10370003','111975006','164889003','164890007','164909002','164917005','164934002',
            '164947007','17338001','251146004','270492004','284470004','39732003','426627000',
            '426177001','426783006','427084000','427172004','427393009','445118002','47665007',
            '59118001','59931005','63593006','698252002','713426002','713427006']
    file2index = dict()
    file2fold = dict()
    delete_record = []
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split(' ')
        id = arr[1] + '.txt'
        labels = [name2idx[name] for name in arr[2].split(',') if name in score]
        if len(labels) == 0:
            delete_record.append(id);
            continue
        fold = arr[0]
        file2index[id] = labels
        file2fold[id] = fold

    return file2index, file2fold, delete_record


def train_27(name2idx, idx2name, folds, input_directory, output_directory):
    '''
    一键生成包含训练集测试集等信息的pth文件
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2idx, file2fold, delete_record = file2index_27(os.path.join(output_directory, 'label.txt'), name2idx)
    train, val = split_data_fold_27(input_directory, file2fold, folds, delete_record)
    # wc = count_labels(train, file2idx)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    torch.save(dd, os.path.join(output_directory, 'train.pth'))
