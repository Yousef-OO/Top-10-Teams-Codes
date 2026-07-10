import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import network
import json


def read_data(input_directory):
    def find_challenge_files(label_directory):#, output_directory):
        label_files = list()
        data_files  = list()
        #output_files = list()
        for f in sorted(os.listdir(label_directory)):
            F = os.path.join(label_directory, f) # Full path for label file
            if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
                root, ext = os.path.splitext(f)
                #g = root + '.csv'
                #G = os.path.join(output_directory, g) # Full path for corresponding output file
                #if os.path.isfile(G):
                label_files.append(F)
                    #output_files.append(G)
            #else:
                 #raise IOError('Output file {} not found for label file {}.'.format(g, f))
            if os.path.isfile(F) and F.lower().endswith('.mat') and not f.lower().startswith('.'):
                data_files.append(F)


        if label_files:# and output_files:
            return label_files, data_files#, output_files
        else:
            raise IOError('No label or output files found.')

        
        
    lb, data = find_challenge_files(input_directory)
    
    
    scored = list()
    with open('dx_mapping_scored.csv', 'r') as f:
        for l in f:
            dxs = (l.split(','))
            scored.append(dxs[1])
    scored = (sorted(scored[1:]))
    
    
    normal_class = '426783006'
    equivalent_classes_collection = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]


    def load_labels(label_files, data_list, normal_class, equivalent_classes_collection):
        # The labels should have the following form:
        #
        # Dx: label_1, label_2, label_3
        #
        num_recordings = len(label_files)

        def load_matFile(matF, data2d=False):
            dict = sio.loadmat(matF)
            data = dict['val']
            length = data.shape[1]
            return data

        data = list()

        for i in range(num_recordings):
            data.append(load_matFile(data_list[i]))


        # Load diagnoses.
        tmp_labels = list()
        for i in range(num_recordings):
            with open(label_files[i], 'r') as f:
                for l in f:
                    if l.startswith('#Dx'):
                        dxs = set(arr.strip() for arr in l.split(': ')[1].split(','))
                        tmp_labels.append(dxs)

        # Identify classes.
        classes = set.union(*map(set, tmp_labels))
        #print('classes= ', classes)
        if normal_class not in classes:
            classes.add(normal_class)
            print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal_class))
        classes = sorted(classes)
        num_classes = len(classes)

        classes2 = list()
        for i in range(num_recordings):
            dxs = tmp_labels[i]
            for dx in dxs:
                if dx in scored:
                    #print('dx=', dx)
                    classes2.append(dx)

        classes3 = list()
        for x in classes2:
            if x not in classes3:
                classes3.append(x)


        classes3 = sorted (classes3)
        #print('number classes=', len(classes3))


        index = list()
        # Use one-hot encoding for labels.
        labels = np.zeros((num_recordings, len(classes3)))#, dtype=np.bool)
        for i in range(num_recordings):
            dxs = tmp_labels[i]
            flag = np.zeros((1,len(dxs)), dtype = np.bool)
            count = 0
            for dx in dxs:
                if dx in classes3:
                    j = classes3.index(dx)
                    labels[i, j] = 1
                    flag [0 ,count] = True

                count += 1

            if np.any(flag) == False:
                index.append(i)


        # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
        # The label for the representative class is positive if any of the labels in the set is positive.
        remove_classes = list()
        remove_indices = list()
        for equivalent_classes in equivalent_classes_collection:
            equivalent_classes = [x for x in equivalent_classes if x in classes3]
            if len(equivalent_classes)>1:
                representative_class = equivalent_classes[0]
                other_classes = equivalent_classes[1:]
                equivalent_indices = [classes3.index(x) for x in equivalent_classes]
                representative_index = equivalent_indices[0]
                other_indices = equivalent_indices[1:]

                labels[:, representative_index] = np.any(labels[:, equivalent_indices], axis=1)
                remove_classes += other_classes
                remove_indices += other_indices

        for x in remove_classes:
            classes3.remove(x)
        labels = np.delete(labels, remove_indices, axis=1)

        # If the labels are negative for all classes, then change the label for the normal class to positive.
        normal_index = classes3.index(normal_class)
        for i in range(num_recordings):
            num_positive_classes = np.sum(labels[i, :])
            if num_positive_classes==0:
                labels[i, normal_index] = 1

        return classes, classes3, labels, data, index, tmp_labels
    
    alli, label_classes, labels, data , ind, tmp = load_labels(lb, data, normal_class, equivalent_classes_collection)
    
    labels = [labels[i] for i in range(len(labels)) if i not in ind]
    data = [data[i] for i in range(len(data)) if i not in ind]

    
    def extData(data, ext_len= 5120):
        ext= np.zeros([12,ext_len])
        for i in range(0,12):
            ext[i][0:len(data[i])]=data[i]
        return ext
    
    def cutData(data, cut_len = 5120):
        cut = np.zeros([12,cut_len])
        for i in range(12):
            lenght = len(data[0])- cut_len
            cut[i] = data[i][lenght:]
        return cut
    
    data2 = list()    
    for i in range(len(data)):
        if len(data[i][0])<=5120:
            data2.append(extData(data[i], ext_len=5120).T)
        else:
            data2.append(cutData(data[i], cut_len=5120).T)
        
    
   
    labels = np.stack(labels, axis =0)

    #print('shape-labels=', (labels.shape))
    data2 = np.stack(data2, axis =0)

    #print('shape-data=', (data2.shape))
    
    
    
    train_x, test_x, train_y, test_y = train_test_split(data2, labels, test_size=0.1, random_state=42)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
    
    #print('train_x.shape=', train_x.shape)
    #print('train_y.shape=', train_y.shape)
    #print('dev_x.shape=', dev_x.shape)
    #print('dev_y.shape=', dev_y.shape)

    return train_x,train_y,dev_x,dev_y
    #print('score=',score)
    #return test_x,test_y
