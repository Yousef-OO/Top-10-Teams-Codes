# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

""" Utility functions """

import os
import sys
import time
import random
import datetime
import subprocess
import numpy as np

import torch


def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else: 
        time = None
        line = string

    print(line, file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_args(args):
    """ sanity check for arguments """
    if args["checkpoint"] is not None and not os.path.exists(args["checkpoint"]):
            sys.exit("checkpoint [%s] does not exists" % (args["checkpoint"]))


def set_output(args, string):
    """ set output configurations """
    output, writer, save_prefix = sys.stdout, None, None
    if args["output_path"] is not None:
        save_prefix = args["output_path"]
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(args["output_path"] + "/" + string + ".txt", "a")
        if "eval" not in string:
            tb = args["output_path"] + "/tensorboard/"
            if not os.path.exists(tb):
                os.makedirs(tb, exist_ok=True)
            #writer = SummaryWriter(tb)

    return output, writer, save_prefix


def get_training_results(paths, metrics, sort_idx, delete_paused=False, delete_checkpoints=False):
    """ get training results """
    if not isinstance(paths, list): paths = [paths]

    config_results, all_done = {}, True
    for p, path in enumerate(paths):
        if not os.path.exists(path): all_done = False; continue

        for config in os.listdir(path):
            if config.startswith("."): continue
            elif not os.path.exists(path + config + "/train_model_log.txt"): all_done = False; continue

            if config not in config_results:
                config_results[config] = [[] for _ in range(len(metrics)+1)]

            FILE = open(path + config + "/train_model_log.txt", "r")
            lines = FILE.readlines()
            FILE.close()
            idxs, results, last_epoch, skip, done = [], [], -1, True, False
            for line in lines:
                tokens = line.strip().split("\t")
                if len(tokens) < 2: continue
                elif tokens[1].startswith("end training"): done = True
                elif tokens[1] == "ep":
                    skip = False
                    for metric in metrics:
                        val_offset = tokens.index("|")
                        idxs.append(tokens[val_offset:].index(metric) + val_offset)
                        results.append([])
                elif not skip and not done:
                    last_epoch += 1
                    for i, idx in enumerate(idxs):
                        results[i].append(float(tokens[idx]))

            if last_epoch < 0:
                print(config, "initializing")
            elif len(metrics) > 0:
                best_epoch = np.argmax(results[sort_idx])
                for i in range(len(results)):
                    config_results[config][i].append(results[i][best_epoch])
                if not done: config_results[config][-1].append("P%d_EP%d_RUN(%d)" % (p, best_epoch + 1, last_epoch + 1))
                else:        config_results[config][-1].append("P%d_EP%d" % (p, best_epoch + 1))
            else:
                if not done: config_results[config][-1].append("P%d_RUN(%d)" % (p, last_epoch + 1))
                else:        config_results[config][-1].append("P%d" % (p))
            if not done: all_done = False

            if delete_paused and not done:
                os.system("rm -rf %s/%s" % (path, config))
            if delete_checkpoints and done and os.path.exists("%s/%s/checkpoints/1.pt" % (path, config)):
                if len(metrics) > 0:
                    os.system("cp %s/%s/checkpoints/%d.pt %s/%s/best.pt" % (path, config, best_epoch + 1, path, config))
                os.system("cp %s/%s/checkpoints/%d.pt %s/%s/last.pt" % (path, config, last_epoch + 1, path, config))
                os.system("rm -rf %s/%s/checkpoints/*" % (path, config))
                os.system("mv %s/%s/*.pt %s/%s/checkpoints/." % (path, config, path, config))

    strings = {}
    for config, results in config_results.items():
        if len(results[-1]) == 0: continue
        tokens = config.split("-")
        preprocess, model, run = tokens[:3]
        if len(tokens) > 3: model += "-" + tokens[3]

        string = [preprocess, model, run]
        for i in range(len(results)-1):
            string.append("%.3f" % np.average(results[i]))
        string.append(results[-1][-1])
        if len(metrics) > 0: strings[np.average(results[sort_idx]) + np.random.rand() / 1000000.0] = string
        else:                strings[np.random.rand() / 1000000.0] = string

    return strings, all_done

