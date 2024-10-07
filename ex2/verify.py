import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from attack import pgd
from modules import Normalize, View, get_model


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='pgd attack seed')
parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(__file__),'models/net_10_none.pt'), help='path to model')
parser.add_argument('--eps', type=float, default=0.025, help='certification epsilon')
parser.add_argument('-k', type=int, default=7, help='pgd steps')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting the random number generator
torch.manual_seed(args.seed)

# Dataset and data loader
test_dataset = datasets.MNIST(
    'mnist_data/', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the model
model = get_model()
print(model)
model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
model.eval()


class AbstractBox:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub


    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        lb = x - eps
        ub = x + eps

        lb.clamp_(min=0, max=1)
        ub.clamp_(min=0, max=1)

        return AbstractBox(lb, ub)

    def propagate_normalize(self, normalize: Normalize) -> 'AbstractBox':
        lb = normalize(self.lb)
        ub = normalize(self.ub)

        return AbstractBox(lb, ub)
        

    def propagate_view(self, view: View) -> 'AbstractBox':
        lb = view(self.lb)
        ub = view(self.ub)

        return AbstractBox(lb, ub)


    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        w = fc.weight
        b = fc.bias
        
        lb = self.lb.repeat(w.shape[0], 1) 
        ub = self.ub.repeat(w.shape[0], 1)
        
        m_lb = torch.where(w > 0, lb, ub)
        m_ub = torch.where(w > 0, ub, lb)

        lb = (m_lb * w).sum(dim=1) + b
        ub = (m_ub * w).sum(dim=1) + b

        return AbstractBox(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        lb = relu(self.lb)
        ub = relu(self.ub)

        return AbstractBox(lb, ub)

    def check_postcondition(self, y) -> bool:
        target = y.item()

        lb =self.lb[target].item()

        for i in range(len(self.lb)):
            if i != target:
                ub = self.ub[i].item()
                if lb <= ub:
                    return False
            
        return True


def certify_sample(model, x, y, eps) -> bool:
    box = AbstractBox.construct_initial_box(x, eps)
    for layer in model:
        if isinstance(layer, Normalize):
            box = box.propagate_normalize(layer)
        elif isinstance(layer, View):
            box = box.propagate_view(layer)
        elif isinstance(layer, nn.Linear):
            box = box.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box.check_postcondition(y)


num_samples, acc, adv_acc, cert_acc = 0, 0, 0, 0
for x, y in (pbar := tqdm(test_loader)):
    # x is a batch of a single sample
    # y is a tensor containing single element (the target of the sample x)
    x, y = x.to(device), y.to(device)

    out = model(x)
    pred = torch.max(out, dim=1)[1]
    is_correct_clean = pred.eq(y).sum().item()

    # Perform the attack in a slightly smaller box to allow for small numerical imprecision
    adv = pgd(model, x, y, args.k, args.eps - 1e-6, 2.5 * (args.eps - 1e-6) / args.k)
    assert (adv - x).abs().max() <= args.eps
    out_adv = model(adv)
    pred_adv = torch.max(out_adv, dim=1)[1]
    is_correct_adv = pred_adv.eq(y).sum().item()

    with torch.no_grad():
        certified = certify_sample(model, x, y, args.eps)

    num_samples += x.shape[0]
    acc += is_correct_clean
    adv_acc += is_correct_adv
    if certified:
        assert is_correct_clean and is_correct_adv, \
            'Sample was certified but adversarial example is found'
        cert_acc += 1

    pbar.set_postfix(
        acc=acc / num_samples,
        adv_acc=adv_acc / num_samples,
        cert_acc=cert_acc / num_samples
    )

print('Clean Acc {:7.5f}, Adversarial Acc {:7.5f}, Certified Acc {:7.5f}'.format(
    acc / num_samples,
    adv_acc / num_samples,
    cert_acc / num_samples
))
