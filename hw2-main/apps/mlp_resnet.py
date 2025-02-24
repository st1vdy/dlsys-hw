import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    stem = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    res_block = nn.Sequential(
        nn.Residual(stem),
        nn.ReLU(),
    )
    return res_block
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    return resnet
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    criterion = nn.SoftmaxLoss()
    num_acc, num_total, loss_average, num_batches = 0, 0, 0, 0

    for i, batch in enumerate(dataloader):
        x, y = batch
        out = model(x)
        if opt:
            opt.reset_grad()
            loss = criterion(out, y)
            loss_average += float(loss.detach().numpy())
            loss.backward()
            opt.step()
        else:
            loss = criterion(out, y)
            loss_average += float(loss.detach().numpy())

        num_acc += (out.numpy().argmax(1) == y.numpy()).sum()
        num_total += y.shape[0]
        num_batches += 1

    acc = num_acc / num_total
    loss_average /= num_batches
    return 1 - acc, loss_average
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(
        os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
        os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'),
    )
    test_set = ndl.data.MNISTDataset(
        os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
        os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'),
    )
    train_loader = ndl.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(epochs):
        train_err, train_loss  = epoch(train_loader, model, opt)
        test_err, test_loss = epoch(test_loader, model, None)

    # Returns a tuple of the training error, training loss, test error, test loss computed in the last epoch of training
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
