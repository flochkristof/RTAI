import argparse
import os
from unicodedata import decimal
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view((-1, 28 * 28))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


def pgd(model, x_batch, target, k, eps, eps_step):
    """Batch-wise implementation of the PGD attack
    
    :param model: the model to attack
    :param x_batch: the input batch
    :param target: the target labels
    :param k: the number of steps
    :param eps: the epsilon value
    :param eps_step: the epsilon in one step
    :return: the adversarial examples
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    x_adv = x_batch + eps * (2*torch.rand_like(x_batch) - 1)
    x_adv.clamp_(min=0., max=1.)
    
    for _ in range(k):
        x_adv.detach_().requires_grad_(True)
        model.zero_grad()
        out = model(x_adv)
        loss = loss_fn(out, target)
        loss.backward()
    
    
        nu = eps_step * torch.sign(x_adv)
        x_adv = x_adv + nu

        x_adv.clamp_(min=0, max=1)

    return x_adv.detach()


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--seed', type=int, default='42', help='seed')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--defense', type=str, choices=['none', 'PGD', 'TRADES'], default='TRADES', help='defense')
parser.add_argument('--num_epochs', type=int, default=10, help='epochs')
parser.add_argument('--eps', type=float, default=0.1, help='pgd epsilon')
parser.add_argument('-k', type=int, default=7, help='pgd steps')
parser.add_argument('--trades_fact', type=float, default=1.0, help='TRADES lambda')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting the random number generator
torch.manual_seed(args.seed)

# Datasets
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), Net())
model = model.to(device)


opt = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(opt, 15)
ce_loss = torch.nn.CrossEntropyLoss()


for epoch in range(1, args.num_epochs + 1):
    # Training
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader)):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if args.defense == 'PGD':
            model.eval()
            x_adv = pgd(model, x_batch, y_batch, args.k, args.eps, 2.5 * args.eps / args.k)
            model.train() 
            out_pgd = model(x_adv)

            loss = ce_loss(out_pgd, y_batch)


        elif args.defense == 'TRADES':
            model.train()
            out_nat = model(x_batch)
            target = F.softmax(out_nat.detach(), dim=1)

    
            model.eval() 
            x_adv = pgd(model, x_batch, target, args.k, args.eps, 2.5 * args.eps / args.k)

            model.train()
            out_adv = model(x_adv)
            loss_nat = ce_loss(out_nat, y_batch)
            loss_adv = ce_loss(out_adv, target)

            loss = loss_nat + args.trades_fact * loss_adv


        elif args.defense == 'none':
            # standard training
            model.train()  # switch to training mode
            out_nat = model(x_batch)
            loss = ce_loss(out_nat, y_batch)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    # Testing
    model.eval()
    tot_test, tot_acc, tot_adv_acc = 0.0, 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        out = model(x_batch)
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()

        adv = pgd(model, x_batch, y_batch, args.k, args.eps, 2.5 * args.eps / args.k)
        out_adv = model(adv)
        pred_adv = torch.max(out_adv, dim=1)[1]
        acc_adv = pred_adv.eq(y_batch).sum().item()

        # Accumulate accuracies
        tot_acc += acc
        tot_adv_acc += acc_adv
        tot_test += x_batch.size()[0]
    
    scheduler.step()

    print('Epoch %d: Accuracy %.5lf, Adv Accuracy %.5lf' % (epoch, tot_acc / tot_test, tot_adv_acc / tot_test))

# optionally save the model
#os.makedirs("models", exist_ok=True)
#torch.save(model.state_dict(), f"models/Net_{args.num_epochs}_{args.defense}")
