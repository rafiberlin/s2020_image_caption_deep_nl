import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
from collections import OrderedDict
import preprocessing

def get_hyper_parameters():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    parameters = OrderedDict([("lr", [0.01, 0.001]),
                              ("batch_size", [10, 100, 100]),
                              ("shuffle", [True, False]),
                              ("epochs", [10, 100]),
                              ("device", device)
                              ])
    return parameters


def get_dataset_file_args():
    file_args = {"train": {"img": "./data/train2017", "inst": "./data/annotations/instances_train2017.json",
                           "capt": "./data/annotations/captions_train2017.json"},
                 "val": {"img": "./data/val2017", "inst": "./data/annotations/instances_val2017.json",
                         "capt": "./data/annotations/captions_val2017.json"}
                 }
    return file_args


DATASET_FILE_PATHS_CONFIG = "dataset_file_args.json"
HYPER_PARAMETER_CONFIG = "hyper_parameters.json"


def main():
    file_args = preprocessing.read_json_config(DATASET_FILE_PATHS_CONFIG)
    hyper_parameters = preprocessing.read_json_config(HYPER_PARAMETER_CONFIG)

    # TODO create a testing split, there is only training and val currently...
    coco_train_set = dset.CocoDetection(root=file_args["train"]["img"],
                                        annFile=file_args["train"]["capt"],
                                        transform=transforms.Compose([preprocessing.CenteringPad(),
                                                                      transforms.Resize((640, 640)), transforms.ToTensor()])
                                        )

    train_loader = torch.utils.data.DataLoader(coco_train_set, hyper_parameters["batch_size"][0])

    batch_one = next(iter(train_loader))
    img, capt = batch_one[0], batch_one[1]


if __name__ == '__main__':
    main()
