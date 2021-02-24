"""
""" # TODO: docstring
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, Resize, RandomRotation, ToTensor

torch.manual_seed(1234)

class PrototypicalNetwork(nn.Module):
    def __init__(self, n_classes, embed, embed_shape, d=torch.cdist):
        """
        :param: n_classes () - number of classes
        :param: embed () - embedding function 
        :param: embed_shape () - shape of the embedded output
        :param: d () - distance metric (default: euclidean)
        """ # TODO: docstring
        self.embed = embed
        self.d = d
        self.prototypes = torch.empty(n_classes, *embed_shape)
    
    def forward(self, x):
        x = self.embed(x)
        n_samples = len(x)
        n_prototypes = len(self.prototypes)
        # XXX: this reshaping trick may no longer be necessary
        dist = self.d(x.view(n_samples, -1), self.prototypes.view(n_prototypes, -1))
        return torch.softmax(-dist, dim=1)

    def episode(samples, labels, n_way, n_shot, n_query):
        """
        :param: n_way - no. of classes used in episode (N_c)
        :param: n_shot - no. of support samples used for each class (N_S)
        :param: n_query - no. of query samples used for each class (N_Q)
        typical recommendations: try to match params with test-time specs. E.g. for 1-shot learning with 5 classes in test set, use N_C = 5 and N_S = 1
        it can be beneficial to use higher N_C
        usually best to have N_S equal for training and testing
        """ # TODO: write docstring
        classes = torch.unique(labels)
        n_samples = len(samples) # N
        n_classes = len(classes) # K

        chosen_classes = classes[torch.randperm(n_classes)][:n_way]
        for c in chosen_classes:
            c_samples = samples[labels == c]
            # pick random samples by shuffling and then choosing first few
            c_samples = c_samples[torch.randperm(len(c_samples))]
            support = c_samples[:n_shot]
            self.prototypes[c] = torch.mean(self.embed(support), dim=0) 
            queries = c_samples[n_shot:n_shot+n_query]
            # TODO: aggregate prototypes and queries
            # TODO: figure out how to handle new classes
        
        dist = self.d(queries, self.prototypes)
        loss = F.cross_entropy(-dist, labels) #XXX: is this a true rewrite of the loss?
        #for query in Q:
        #    loss += -self(query) # XXX: should add log

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
#X = torch.cat(X)
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
    *(3*conv_block), # repeat conv_block 3 times
    nn.Flatten()
)

model = PrototypicalNetwork(len(torch.unique(y)), embed, embed_shape=(64,))

optim = Adam(model.parameters(), lr=0.001)
lr_scheduler = StepLR(optim, step_size=2000, gamma=0.5)

#for i, batch in enumerate(train_loader):
#    loss = model.episode(batch_labels, )
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
