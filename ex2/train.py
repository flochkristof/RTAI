import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from attack import pgd
from modules import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--defense', type=str, choices=['none', 'PGD', 'TRADES'], default='none', help='defense type')
parser.add_argument('--num_epochs', type=int, default=10, help='epochs')
parser.add_argument('--eps', type=float, default=0.1, help='pgd epsilon')
parser.add_argument('-k', type=int, default=7, help='pgd steps')
parser.add_argument('--trades_fact', type=float, default=1.0, help='TRADES lambda')
parser.add_argument('--save_model', action='store_true', default=False, help='save the model after training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting the random number generator
torch.manual_seed(args.seed)

# Datasets
train_dataset = datasets.MNIST(
    'mnist_data/', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_dataset = datasets.MNIST(
    'mnist_data/', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = get_model()
model = model.to(device)


opt = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(opt, 15)
ce_loss = torch.nn.CrossEntropyLoss()


for epoch in range(1, args.num_epochs + 1):
    # Training
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader)):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if args.defense == 'PGD':
            # raise NotImplementedError
            # TODO: Problem 4.1 implement PGD training
            model.eval() # switch net to evaluation mode, to ensure it is deterministic
            x_adv = pgd(model, x_batch, y_batch, args.k, args.eps, 2.5 * args.eps / args.k)

            model.train() # switch to training mode
            out_pgd = model(x_adv)

            loss = ce_loss(out_pgd, y_batch)


        elif args.defense == 'TRADES':
            # raise NotImplementedError
            # TODO: Problem 4.2 implement TRADES training
            model.train()  # switch to training mode
            out_nat = model(x_batch)
            target = F.softmax(out_nat.detach(), dim=1)  # outputs as probabilities

            # do PGD attack to generate adversarial examples
            model.eval()  # switch net to evaluation mode, to ensure it is deterministic
            x_adv = pgd(model, x_batch, target, args.k, args.eps, 2.5 * args.eps / args.k)

            # Calculate natural and adversarial loss, add them
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

        # TODO: Problem 4.1 and 4.2 - calculate accuracy under PGD attack
        adv = pgd(model, x_batch, y_batch, args.k, args.eps, 2.5 * args.eps / args.k)
        out_adv = model(adv)
        pred_adv = torch.max(out_adv, dim=1)[1]
        acc_adv = pred_adv.eq(y_batch).sum().item()

        # Accumulate accuracies
        tot_acc += acc
        tot_adv_acc += acc_adv
        tot_test += x_batch.size()[0]

    scheduler.step()

    print('Epoch {}: Accuracy {:7.5f}, Adv Accuracy {:7.5f}'.format(
        epoch, tot_acc / tot_test, tot_adv_acc / tot_test
    ))

if args.save_model:
    # optionally save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/net_{args.num_epochs}_{args.defense}.pt')
