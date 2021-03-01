"""
""" # TODO: docstring
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, Resize, RandomRotation, ToTensor

torch.manual_seed(1234)

class PrototypicalNetwork(nn.Module):
    def __init__(self, embed, embed_shape, d=torch.cdist):
        """
        :param: embed () - embedding function 
        :param: embed_shape () - output shape of the embedding function
        :param: d () - distance metric (default: euclidean)
        """ # TODO: docstring
        super(PrototypicalNetwork, self).__init__()
        self.embed = embed
        self.embed_shape = embed_shape
        self.d = d
        self.prototypes = None
    
    def forward(self, support, queries):
        n_way = support.shape[0] #len(torch.unique(support_labels))
        n_shot = support.shape[1] #len(support) // n_way

        support = support.flatten(0,1)
        queries = queries.flatten(0,1)

        support = self.embed(support).view(n_way, n_shot, -1)
        self.prototypes = support.mean(dim=1)

        dist = self.d(self.embed(queries), self.prototypes)
        return torch.softmax(-dist, dim=1)

    def episode(self, support, queries):
        """
        :param n_way: - no. of classes used in episode (N_c)
        :param n_shot: - no. of support samples used for each class (N_S)
        :param n_query: - no. of query samples used for each class (N_Q)
        typical recommendations: try to match params with test-time specs. E.g. for 1-shot learning with 5 classes in test set, use N_C = 5 and N_S = 1
        it can be beneficial to use higher N_C
        usually best to have N_S equal for training and testing
        """ # TODO: write docstring
        n_way = support.shape[0] #len(torch.unique(support_labels))
        n_shot = support.shape[1] #len(support) // n_way
        n_query = queries.shape[1] #len(X_queries) // n_way

        support = support.flatten(0,1)
        queries = queries.flatten(0,1)
        query_labels = torch.arange(n_way).repeat_interleave(n_query)

        support = self.embed(support).view(n_way, n_shot, -1)
        self.prototypes = support.mean(dim=1)
        dist = self.d(self.embed(queries), self.prototypes)
        loss = F.cross_entropy(-dist, query_labels)

        return loss

        # 1. pick N_C classes from `classes`
        # 2. for every class:
        # 3.    S_K <- select N_S points from samples with class
        # 4.    Q_K <- select N_S points from (samples\S_K) with class (XXX: shouldn't it just be the remaining instead of sampling N_Q points?)
        # 5.    c_k <- average of f_theta(x_i) # compute prototype
        # 6. for every query sample:
        # 7.    loss += -log p(y=k|x)
    
    def fit(self, x, y):
        pass
    
    def predict(self, x, y):
        pass

def episode_split(samples, labels, n_way, n_shot, n_query):
    """
    """# TODO: docstring
    X = torch.empty(n_way, n_shot+n_query, *samples.shape[1:])
    y = torch.zeros(n_way, n_shot+n_query, dtype=torch.long)

    classes = torch.unique(labels)
    n_classes = len(classes)
    chosen_classes = torch.randperm(n_classes)[:n_way]
    chosen_classes = classes[chosen_classes]
    for ci, c in enumerate(chosen_classes):
        c_samples = samples[labels == c]
        c_samples = c_samples[torch.randperm(len(c_samples))]
        X[ci] = c_samples[:n_shot+n_query]
        y[ci, :] = c
        
    X_support = X[:, :n_shot]
    X_queries = X[:, n_shot:]
    y_support = y[:, :n_shot]
    y_queries = y[:, n_shot:]
    return X_support, y_support, X_queries, y_queries

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

# TODO: make main() and fix these via args:
#validation_size = 500
#epochs = 1000
batch_size = 250
n_way = 60
n_shot = 5
n_query = 5
test_way = 5

transform = Compose([
    Resize((28,28)),
    ToTensor()
])

data = torchvision.datasets.Omniglot(root='omniglot', download=True, transform=transform)
X, y = zip(*data) # TODO: use this to make code cleaner

X = torch.stack(X)
y = torch.tensor(y)

X_train, y_train = X[:14460], y[:14460] # 75 % split (there aren't >1200 chars in the dataset as described in the paper)
X_test, y_test = X[14460:], y[14460:]

rotations = torch.empty(4, *X_train.shape)
for i, deg in enumerate([0, 90, 180, 270]):
    transform = RandomRotation((deg,deg))
    rotated = transform(X_train)
    rotations[i] = rotated
X_train = rotations.flatten(0,1)
y_train = y_train.repeat(4)

#train_loader = DataLoader(augmented_train_data, batch_size=batch_size)

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
    *(3*conv_block), # repeat conv_block 3 times
    nn.Flatten()
)

model = PrototypicalNetwork(embed, embed_shape=(64,))

optim = Adam(model.parameters(), lr=0.001)
lr_scheduler = StepLR(optim, step_size=2000, gamma=0.5)

model.train()
for epoch in range(1000):
    support, support_labels, queries, queries_labels = episode_split(X_train, y_train, n_way, n_shot, n_query)
    loss = model.episode(support, queries)
    print(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

model.eval()
eval_acc = []
for test_it in range(1000):
    support, support_labels, queries, query_labels = episode_split(X_test, y_test, test_way, n_shot, n_query)
    preds = model(support, queries).argmax(dim=-1)
    preds = support_labels[preds,0]
    truth = query_labels.flatten()
    acc = (preds == truth).sum() / len(preds)
    eval_acc.append(acc)
    print(test_it, "\t- accuracy:", acc.item())
print(np.mean(eval_acc))
