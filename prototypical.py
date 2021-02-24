"""
""" # TODO: docstring
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Resize, RandomRotation, ToTensor

torch.manual_seed(1234)

samples = []
labels = []
N = len(samples)
classes = torch.unique(labels)
K = len(classes)

# distance metric
d = lambda a, b: torch.sum(torch.square(a - b))
# number of classes per episode
N_C = 5
# number of support examples per class
N_S = 5
# number of query examples per class
N_Q = 5

# typical recommendations: try to match params with test-time specs. E.g. for 1-shot learning with 5 classes in test set, use N_C = 5 and N_S = 1
# it can be beneficial to use higher N_C
# usually best to have N_S equal for training and testing


# episode(samples, labels, classes):
# 1. pick N_C classes from `classes`
# 2. for every class:
# 3.    S_K <- select N_S points from samples with class
# 4.    Q_K <- select N_S points from (samples\S_K) with class (XXX: shouldn't it just be the remaining instead of sampling N_Q points?)
# 5.    c_k <- average of f_theta(x_i) # compute prototype
# 6. for every query sample:
# 7.    loss += -log p(y=k|x)


# Results:
# - Dataset: Omniglot [7] (1263 chars, 20 examples of each char, from 50 alphabets)
#     - Vinyals et. al: resize grayscale imgs to 28x28 and augment with rotations in multiples of 90 degrees 
#         - 1200 chars plus rotations for training (4800 classes[?] in total)
#         - embedding architecture: 4 conv blocks of [64-filter 3x3 conv, BN, ReLU, 2x2 max-pooling]
#         - Train using Adam with initial learning rate of 10^-3, halved every 2000 episodes.
# - Accuracy averaged over 1000 randomly generated episodes from test set.
# - 5-way accuracy:
#     - 1-shot: 98.8% 
#     - 5-shot: 99.7%
# - 20-way accuracy:
#     - 1-shot: 96.0%
#     - 5-shot: 98.9%

data = torchvision.datasets.Omniglot(root='omniglot', download=True)
data = list(data)
X, y = zip(*data) # TODO: use this to make code cleaner
train_data = data[:1200]
test_data = data[1200:]

augmented_train_data = []
for deg in [0, 90, 180, 270]:
    transform = Compose([
        Resize((28,28)),
        RandomRotation((deg,deg)),
        ToTensor()
    ])
    augmented_data = [(transform(x[0]), x[1]) for x in train_data]
    augmented_train_data += augmented_data

# TODO: Figure out padding_mode for conv2d layer
conv_block = [
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(2)
]
embed = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    *(3*conv_block) # repeat conv_block 3 times
)