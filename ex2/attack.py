import torch


def pgd(model, x_batch, target, k, eps, eps_step):
    # TODO: Problem 4.1 and 4.2 - implement PGD attack
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # Initialize random input around x_batch within an eps sized box
    x_adv = x_batch + eps * (2 * torch.rand_like(x_batch) - 1)
    # clamp values back down to the [0,1] range
    x_adv.clamp_(min=0., max=1.)

    for _ in range(k):
        x_adv.detach_().requires_grad_()

        # Calculate the loss and calculate the gradients.
        model.zero_grad()
        out = model(x_adv)
        loss_fn(out, target).backward()

        # Calculate the step and add the step and project back to the eps
        # sized box around x_batch
        step = eps_step * x_adv.grad.sign()
        x_adv = x_batch + (x_adv + step - x_batch).clamp_(min=-eps, max=eps)

        # Clamp back to image domain; in contrast to the previous exercise we clamp at each step
        # (so this is part of the projection)
        # Both implementations are valid; this tends to work slightly better
        x_adv.clamp_(min=0, max=1)

    return x_adv.detach()
