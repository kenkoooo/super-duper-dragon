import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, n_unit: int):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(28*28*1, n_unit)
        self.l2 = nn.Linear(n_unit, n_unit)
        self.l3 = nn.Linear(n_unit, 10)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", '-b', type=int, default=100)
    parser.add_argument("--epoch", "-e", type=int, default=20)
    parser.add_argument("--unit", "-u", default=1000, type=int)
    parser.add_argument("--gpu", "-g", type=bool, default=False)
    parser.add_argument("--initmodel", "-m", default="")
    parser.add_argument("--resume", "-r", default="")
    args = parser.parse_args()

    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset1 = datasets.MNIST('../mnist',
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())
    dataset2 = datasets.MNIST('./mnist',
                              train=False,
                              download=True,
                              transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=args.batchsize)
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=args.batchsize)

    model = MLP(args.unit).to(device)
    optimizer = optim.SGD(model.parameters(),  lr=0.01)

    if args.initmodel:
        print(f"Loading {args.initmodel}")
        model.load_state_dict(torch.load(args.initmodel))
    if args.resume:
        print(f"Loading {args.resume}")
        optimizer.load_state_dict(torch.load(args.resume))

    for epoch in range(1, args.epoch+1):
        model.train()
        sum_loss = 0
        itr = 0
        for data, target in train_loader:
            x, t = data.view(-1, 28*28*1).to(device), target.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = nn.CrossEntropyLoss()(y, t)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            itr += 1

        model.eval()
        sum_test_loss = 0
        sum_test_accuracy = 0
        test_itr = 0
        with torch.no_grad():
            for data, target in test_loader:
                x_test, t_test = data.view(-1, 28 *
                                           28*1).to(device), target.to(device)
                y_test = model(x_test)
                sum_test_loss += nn.CrossEntropyLoss()(x_test, t_test).item()

                pred = y_test.argmax(dim=1, keepdim=True)
                sum_test_accuracy += pred.eq(t_test.view_as(pred)).sum().item()
                test_itr += 1
        print(
            f"epoch={epoch} train_loss={sum_loss/itr} test_loss={sum_test_loss/test_itr} accuracy={sum_test_accuracy/test_itr}")

    torch.save(model.state_dict(), "./mnist/mlp.model")
    torch.save(optimizer.state_dict(), "./mnist/mlp.state")


if __name__ == "__main__":
    main()
