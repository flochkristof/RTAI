import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torchvision import datasets, transforms
from model import Net, ConvNet

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

import matplotlib.pyplot as plt

# hard-code random seeds for deterministic outcomes
np.random.seed(40)
torch.manual_seed(40)

# loading the dataset
# note that this time we do not perfrom the normalization operation, see next
test_dataset = datasets.MNIST(
    'mnist_data/',
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307)/0.3081


# we load the body of the neural net
model = torch.load(os.path.join(os.path.dirname(__file__),'model.net'), map_location='cpu') 

# ... and add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), model)

# and here we also create a version of the model that outputs the class probabilities
model_to_prob = nn.Sequential(model, nn.Softmax())

# we put the neural net into evaluation mode (this disables features like dropout)
model.eval()
model_to_prob.eval()


# define a show function for later
def show(original, adv, model_to_prob):
    p0 = model_to_prob(original).detach().numpy()
    p1 = model_to_prob(adv).detach().numpy()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[0].set_title("Original, class: " + str(p0.argmax()))
    axarr[1].imshow(adv.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[1].set_title("Attacked, class: " + str(p1.argmax()))
    print("Class\t\tOrig\tAdv")
    for i in range(10):
        print("Class {}:\t{:.2f}\t{:.2f}".format(i, float(p0[:, i]), float(p1[:, i])))
    plt.show(block=False)


# x: input image
# target: target class
# eps: size of l-infinity ball
def fgsm_targeted(model, x, target, eps, **kwargs):
    """Perform targeted FGSM attack on the input.

    .. math::

        x_\mathrm{adv}=x-\epsilon \cdot \mathrm{sign}(\nabla_x \ell(x))
    
    :param model: the model to attack
    :type model: nn.Module
    :param x: the input data
    :type x: torch.Tensor
    :param target: the (expected) target class
    :type target: int
    :param eps: the size of the l-infinity ball
    :type eps: float
    :return: the adversarial example
    :rtype: torch.Tensor
    """
    input_ = x.clone().detach().requires_grad_(True)

    output = model(input_)
    model.zero_grad()
    target = torch.LongTensor([target])

    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    out = input_ - eps * torch.sign(input_.grad)


    if 'clip_min' in kwargs:
        out = torch.clamp(out, min=kwargs['clip_min'])
    if 'clip_max' in kwargs:
        out = torch.clamp(out, max=kwargs['clip_max'])


    return out


# x: input image
# label: current label of x
# eps: size of l-infinity ball
def fgsm_untargeted(model, x, label, eps, **kwargs):
    """Perform untargeted FGSM attack on the input.

        .. math::

        x_\mathrm{adv}=x-\epsilon \cdot \mathrm{sign}(\nabla_x \ell(x))
    
    :param model: the model to attack
    :type model: nn.Module
    :param x: the input data
    :type x: torch.Tensor
    :param eps: the size of the l-infinity ball
    :type eps: float
    :return: the adversarial example
    :rtype: torch.Tensor
    """

    input_ = x.clone().detach_().requires_grad_(True)
    
    output = model(input_)
    label = torch.LongTensor([label])
    model.zero_grad()

    loss = nn.CrossEntropyLoss()(output, label)
    loss.backward()
    
    out = input_ + eps * torch.sign(input_.grad)
    
    if 'clip_min' in kwargs:
        out = torch.clamp(out, min=kwargs['clip_min'])
    if 'clip_max' in kwargs:
        out = torch.clamp(out, max=kwargs['clip_max'])
    return out



# try out our attacks
original = torch.unsqueeze(test_dataset[0][0], dim=0)

adv_untargeted = fgsm_untargeted(model, original, label=test_dataset[0][1], eps=0.25, clip_min=0, clip_max=1.0)
show(original, adv_untargeted, model_to_prob)

adv_targeted = fgsm_targeted(model, original, target=2, eps=0.25, clip_min=0, clip_max=1.0)
show(original, adv_targeted, model_to_prob)
plt.show()