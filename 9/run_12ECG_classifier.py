#!/usr/bin/env python

import numpy as np, os, sys
#import joblib
import tensorflow as tf 
#import torch

#torch.cuda.set_device(7)
#from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,loaded_model):

    data1 = list()
    num_classes = 24
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    
    def extData(data2, ext_len= 4864):
        ext= np.zeros([12,ext_len])
        for i in range(0,12):
            ext[i][0:len(data2[i])]=data2[i]
        return ext
    
    def cutData(data2, cut_len = 4864):
        cut = np.zeros([12,cut_len])
        for i in range(12):
            lenght = len(data2[i])- cut_len
            cut[i] = data2[i][lenght:]
        return cut
    
    #data2 = list()    
    #for i in range(len(data)):
    if len(data[0])<=5120:
        data1.append(extData(data, ext_len=5120).T)
    else:
        data1.append(cutData(data, cut_len=5120).T)
    # Use your classifier here to obtain a label and score for each class. 
    '''if data.shape[1]>=32000:
        ext = data[:,0:32000]
    elif data.shape[1]<32000:
        ext = np.zeros([12,32000])
        for i in range(0,11):
            ext[i][0:len(data[i])]= data[i]'''

    #data1.append(ext.T)
    data1 = np.stack(data1,axis=0)

    score2 = loaded_model.predict(data1)
    score3 = np.zeros((score2.shape[0],24), dtype=int)
    
    for j in range(score2.shape[0]):
        found = False
        max=-1
        max_i=-1
        for i in range(len(score2[j])):

            if(score2[j][i]>max):
                max = score2[j][i]
                max_i=i
            if(score2[j][i]>=0.2):
                score3[j][i]=1
                found=True
        if found==False:
            score3[j][max_i]=1
            
    for i in range(num_classes):
        current_score[i] = np.array(score2[0][i])    
        
    for i in range(num_classes):
        current_label[i] = np.array(score3[0][i])  
 
    # Use your classifier here to obtain a label and score for each class.
    '''model = loaded_model['model']
    imputer = loaded_model['imputer']
    classes = loaded_model['classes']

    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1, -1)
    feats_reshape = imputer.transform(feats_reshape)
    current_label = model.predict(feats_reshape)[0]
    current_label=current_label.astype(int)
    current_score = model.predict_proba(feats_reshape)
    current_score=np.asarray(current_score)
    current_score=current_score[:,0,1]'''
    
    scored = list()
    with open('dx_mapping_scored.csv', 'r') as f:
        for l in f:
            dxs = (l.split(','))
            scored.append(dxs[1])
    scored = (sorted(scored[1:]))
    
    
    equivalent_classes_collection = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in scored]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [scored.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            #labels[:, representative_index] = np.any(labels[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        scored.remove(x)

    return current_label, current_score,scored

def load_12ECG_model(input_directory):
    # load the model from disk 
    f_out = 'model.h5'
    #f_out = 'model.pt'
    filename = os.path.join(input_directory,f_out)

    #loaded_model = joblib.load(filename)
    loaded_model = tf.keras.models.load_model(filename)
    #loaded_model = torch.load(filename)
    #loaded_model.eval()

    return loaded_model
