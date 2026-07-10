import sys
sys.path.append('..')
import os
import re
import torch
from glob import glob
from network.main import get_network
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_auc_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from learn.tester import Tester
import utils.dataset as ds_utils


def get_dataformat(model_path):
    if 'dataformat=rhythm' in model_path:
        return 'rhythm'
    return 'median'


def get_lead12(model_path):
    if 'lead12=True' in model_path:
        return True
    return False


def get_model_info(model_path):
    device = torch.device('cuda')
    checkpoint = torch.load(model_path, map_location=device)
    model_info = dict((k, v) for k, v in checkpoint.items() if k not in [
      'model_state_dict', 'optimizer_state_dict',
    ])
    return model_info

  
def init_model(model_path):
    device = torch.device('cuda')
    checkpoint = torch.load(model_path, map_location=device)
    model = get_network(checkpoint['network_name'])(**checkpoint['network_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    model_info = dict((k, v) for k, v in checkpoint.items() if k not in [
      'model_state_dict', 'optimizer_state_dict',
    ])
    return model, model_info


def get_model_dir_path(model_dir, root_dir='models', data_dir='cardiomyopathy'):
    return os.path.join(
      '/media/data_ssd_1t/ECGnet/data_students',
      data_dir,
      root_dir,
      model_dir,
    )

  
def get_best_model(model_dir_path):
    models = os.listdir(model_dir_path)
    high_score, best_model = 0, None
    for model in models:
        model_path = os.path.join(model_dir_path, model)
        if not os.path.splitext(model_path)[1] == '.pt':
          continue
        model_info = get_model_info(model_path)
        if model_info['report']['roc_auc'] > high_score:
            high_score = model_info['report']['roc_auc']
            best_model = model
    return init_model(os.path.join(model_dir_path, best_model))


def keep_best_model(model_dir_path, seed=None):
    def remove_for_seed(model_dir_path, seed, models):
        epochs = set([int(re.search(f'seed_{seed}.epoch_(.*).pt', m).group(1))
                 for m in models if re.search(f'seed_{seed}.epoch_(.*).pt', m)])
        max_epoch = max(epochs)
        seed_models = [re.search(f'seed_{seed}.epoch_(.*).pt', m).group()
                 for m in models if re.search(f'seed_{seed}.epoch_(.*).pt', m)]
        best_model = f'seed_{seed}.epoch_{str(max_epoch)}.pt'
        seed_models.remove(best_model)
        # delete every file that is not max for seed
        for file in seed_models:
            path = os.path.join(model_dir_path, file)
            os.remove(path)
      
    models = os.listdir(model_dir_path)
    # get a list of each seeds
    seeds = set([re.search('seed_(.+?).epoch_(.*).pt', m).group(1)
             for m in models if re.search('seed_(.+?).epoch_(.*).pt', m)])
    # for each seed find max
    if seed is None:
      for seed in seeds:
          remove_for_seed(model_dir_path, seed, models)
    else:
      remove_for_seed(model_dir_path, seed, models)
  

def get_best_model_path(model_dir_path):
    models = os.listdir(model_dir_path)
    epochs = [int(re.search(r'\d+', m).group()) for m in models]
    epoch = max(epochs)
    return os.path.join(model_dir_path, f'epoch_{epoch}.pt')
  
          
def get_model_report(testset, model):
    tester = Tester(model)
    testloader = ds_utils.get_dataloader(testset, batch_size=128)
    y_trues, y_preds, y_scores, y_probs = tester.test(testloader)
    report = classification_report(y_trues, y_preds, y_scores, y_probs)
    return report


def get_best_model_info(model_dir, root_dir='models', data_dir='cardiomyopathy'):
    dataformat = get_dataformat(model_dir)
    lead12 = get_lead12(model_dir)
    model_dir_path = get_model_dir_path(model_dir, root_dir, data_dir)
    model_path = get_best_model_path(model_dir_path)
    model, model_info = init_model(model_path)
    return model_info
  

def sensitivity(tn, fp, fn, tp):
    return tp/(tp+fn)


def specificity(tn, fp, fn, tp):
    return tn/(tn+fp)
    

def false_positive_rate(tn, fp, fn, tp):
    return fp/(tn+fp)


def positive_predictive_value(tn, fp, fn, tp):
    return tp/(tp+fp)


def negative_predictive_value(tn, fp, fn, tp):
    return tn/(tn+fn)


def f1(tn, fp, fn, tp):
    recall = sensitivity(tn, fp, fn, tp)
    precision = positive_predictive_value(tn, fp, fn, tp)
    return 2*((recall*precision)/(recall+precision))


def fbeta(tn, fp, fn, tp, beta=2):
    return ((1+beta**2)*tp)/((1+beta**2)*tp+fp+(beta**2)*fn)


def jaccard(tn, fp, fn, tp, beta=2):
    return (tp)/(tp+fp+beta*fn)


def classification_report(y_true, y_pred, y_scores, y_probs):
    """
    `y_true` - true class labels
    `y_pred` - prediction scores binarized
    `y_scores` - the prediction scores
    `y_probs` - prediction scores converted into probabilities
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    extra = {}
    is_multilabel = len(y_true.shape) > 1 and y_true.shape[1] > 1
    if is_multilabel:
          confusionmatrices = multilabel_confusion_matrix(y_true, y_pred)
    else:
          confusionmatrices = [confusion_matrix(y_true, y_pred)]
          roc = roc_curve(y_true, y_probs)
          extra['roc'] = roc
    reports = []
    for confusionmatrix in confusionmatrices:
      confusion = confusionmatrix.ravel()
      reports.append({
          'confusion_matrix': confusionmatrix,
          'confusion': confusion,
          'sensitivity': sensitivity(*confusion),
          'specificity': specificity(*confusion),
          'fpr': false_positive_rate(*confusion),
          'ppv': positive_predictive_value(*confusion),
          'npv': negative_predictive_value(*confusion),
          'f1': f1(*confusion),
          'f2': fbeta(*confusion, beta=2),
          'g2': jaccard(*confusion, beta=2),
#           'roc_auc': roc_auc_score(y_true, y_probs),
          'ap': average_precision_score(y_true, y_probs),
          **extra,
      })
    return reports if is_multilabel else reports[0]


def calculate_report(confusion):
    return {
        'sensitivity': sensitivity(*confusion),
        'specificity': specificity(*confusion),
        'fpr': false_positive_rate(*confusion),
        'ppv': positive_predictive_value(*confusion),
        'npv': negative_predictive_value(*confusion),
    }


def confusion_testids(y_true, y_pred, y_tids):
    tp_tids = y_tids[(y_true == 1) & (y_pred == 1)]
    fp_tids = y_tids[(y_true == 0) & (y_pred == 1)]
    tn_tids = y_tids[(y_true == 0) & (y_pred == 0)]
    fn_tids = y_tids[(y_true == 1) & (y_pred == 0)]
    return tn_tids, fp_tids, fn_tids, tp_tids
