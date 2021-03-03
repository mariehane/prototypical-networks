"""A re-implementation of "Prototypical Networks for Few-shot Learning"
by Jake Snell, Kevin Swersky, and Richard S. Zemel.

See: https://arxiv.org/abs/1703.05175
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, Resize, RandomRotation, ToTensor
from tqdm import tqdm

class PrototypicalNetwork(nn.Module):
    def __init__(self, embed, embed_shape, d=torch.cdist):
        """Initialize a Prototypical Network.

        Args:
            embed (callable): embedding function
            embed_shape (tuple): output shape of the embedding function
            d (callable): distance metric (default: euclidean)
        """
        super(PrototypicalNetwork, self).__init__()
        self.embed = embed
        self.embed_shape = embed_shape
        self.d = d
        self.prototypes = None
    
    def forward(self, support, queries):
        """Compute prediction logits for query samples.

        Args:
            support: tensor of shape [n_way, n_shot, *in_dim]
            queries: tensor of shape [n_way, n_query, *in_dim]

        Returns:
            logits: tensor of shape [n_way*n_query, n_way]
        """
        n_way = support.shape[0] #len(torch.unique(support_labels))
        n_shot = support.shape[1] #len(support) // n_way

        support = support.flatten(0,1)
        queries = queries.flatten(0,1)

        support = self.embed(support).view(n_way, n_shot, -1)
        self.prototypes = support.mean(dim=1)

        dist = self.d(self.embed(queries), self.prototypes)
        return torch.softmax(-dist, dim=1)

    def episode(self, support, queries):
        """Perform a training episode and compute negative log-likelihood loss.

        Args:
            support: tensor of shape [n_way, n_shot, *in_dim]
            queries: tensor of shape [n_way, n_query, *in_dim]

        Returns:
            loss (float): tensor containing single value
        """
        n_way = support.shape[0]
        n_shot = support.shape[1]
        n_query = queries.shape[1]

        support = support.flatten(0,1)
        queries = queries.flatten(0,1)
        query_labels = torch.arange(n_way).repeat_interleave(n_query)

        support = self.embed(support).view(n_way, n_shot, -1)
        self.prototypes = support.mean(dim=1)
        dist = self.d(self.embed(queries), self.prototypes)
        loss = F.cross_entropy(-dist, query_labels)

        return loss
    
    def predict(self, support, support_labels, queries):
        """Compute prediction labels for query samples.

        Args:
            support: tensor of shape [n_way, n_shot, *in_dim]
            support_labels: tensor of shape [n_way, n_shot]
            queries: tensor of shape [n_way, n_query, *in_dim]

        Returns:
            predictions: tensor of shape [n_way*n_query]
        """
        preds = self(support, queries).argmax(dim=-1)
        preds = support_labels[preds,0]
        return preds

def episode_split(samples, labels, n_way, n_shot, n_query):
    """Sample data to facilitate n-way k-shot learning.

    Typical recommendations are to match params with test-time specs [0].
    E.g. for 1-shot learning with 5 classes in test set, use n_way = 5 and n_shot = 1.
    However, [0] found that it can be beneficial to use higher n_way for training than testing.
    For Prototypical Networks it is usually best to have n_shot equal for training and testing.

    [0]: https://arxiv.org/abs/1703.05175

    Args:
        samples: tensor of shape [n_samples, *in_dim]
        labels: tensor of shape [n_samples]
        n_way (int): no. of classes used in episode (N_c)
        n_shot (int): no. of support samples used for each class (N_S)
        n_query (int): no. of query samples used for each class (N_Q)

    Returns:
        X_support: support samples of shape [n_way, n_shot, *in_dim]
        y_support: support labels of shape [n_way, n_shot]
        X_queries: query samples of shape [n_way, n_query, *in_dim]
        y_queries: query labels of shape [n_way, n_query]
    """
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

def main():
    """Trains and evaluates Prototypical Network on Omniglot dataset as specified in the original paper.

    - Resize grayscale images to 28x28 and augment with rotations in multiples of 90 degrees.
    - Embedding architecture: 4 conv blocks of [64-filter 3x3 conv, BN, ReLU, 2x2 max-pooling].
    - Train using Adam with initial learning rate of 10^-3, halved every 2000 episodes.
    - Average accuracy over 1000 randomly generated episodes from test set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=4000)
    #parser.add_argument('--validation-percent', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--train-way', type=int, default=60)
    parser.add_argument('--test-way', type=int, default=5)
    #parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    print("Args:", args)

    torch.manual_seed(1234)

    transform = Compose([
        Resize((28,28)),
        ToTensor()
    ])

    background = torchvision.datasets.Omniglot(root='.', download=True, transform=transform)
    evaluation = torchvision.datasets.Omniglot(root='.', background=False, download=True, transform=transform)
    X_train, y_train = zip(*background)
    X_test, y_test = zip(*evaluation)

    X_train = torch.stack(X_train)
    X_test = torch.stack(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # augment training images with rotations in multiples of 90 degrees
    rotations = torch.empty(4, *X_train.shape)
    for i, deg in enumerate([0, 90, 180, 270]):
        transform = RandomRotation((deg,deg))
        rotated = transform(X_train)
        rotations[i] = rotated
    X_train = rotations.flatten(0,1)
    y_train = y_train.repeat(4)

    # TODO: Figure out padding_mode for conv2d layer
    def conv_block(in_dim=64, out_dim=64):
        return [
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(num_features=out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
    conv_blocks = conv_block(1,64) + 3*conv_block(64,64)
    embed = nn.Sequential(
        *conv_blocks,
        nn.Flatten()
    )

    model = PrototypicalNetwork(embed, embed_shape=(64,))

    optim = Adam(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optim, step_size=2000, gamma=0.5)

    model.train()
    for train_it in tqdm(range(args.episodes)):
        support, _, queries, _ = episode_split(X_train, y_train, args.train_way, args.shot, args.query)
        loss = model.episode(support, queries)
        tqdm.write(f"Episode {train_it}:\tloss: {loss.item()}")
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()

    model.eval()
    eval_acc = []
    for test_it in tqdm(range(1000)):
        support, support_labels, queries, query_labels = episode_split(X_test, y_test, args.test_way, args.shot, args.query)
        preds = model.predict(support, support_labels, queries)
        truth = query_labels.flatten()
        acc = (preds == truth).sum() / len(preds)
        eval_acc.append(acc)
        tqdm.write(f"{test_it}\t- accuracy: {acc.item()}")
    print(np.mean(eval_acc))

if __name__=="__main__":
    main()