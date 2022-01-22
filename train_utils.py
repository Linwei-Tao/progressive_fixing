'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score

loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score
}


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       loss_mean=False):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 50
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        if ('mmce' in loss_function):
            loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda,
                                                                  device=device))
        else:
            loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader) * len(data),
        #         100. * batch_idx / len(train_loader),
        #         loss.item()))

    return train_loss / num_samples


def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda,
                                                                       device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda,
                                                          device=device).item()
            num_samples += len(data)

    return loss / num_samples


def PF_fix(model, PF_round, epoch):
    PF_round = PF_round + 1
    print(f"PF_fix round {PF_round}")
    model.PF_epochs.append(epoch)

    for i in range(PF_round):
        print(f'fix {model.blocks_name[-(i + 1)]} parameters')
        for param in getattr(model, model.blocks_name[-(i + 1)]).parameters():
            param.requires_grad = False
    return PF_round


class ProgressiveFixer:
    def __init__(self):
        self.PF_round = 0
        self.patience_steps = 0

    def fix(self, args, model, epoch, training_set_loss, val_set_loss, val_set_ece):
        training_set_loss = list(training_set_loss.values())
        val_set_loss = list(val_set_loss.values())
        val_set_ece = list(val_set_ece.values())

        if args.PF_criterion == "patience":
            if val_set_loss[-1] < min(val_set_loss):
                self.patience_steps = 0

            if val_set_loss[-1] > min(val_set_loss):
                self.patience_steps += 1

            if self.patience_steps >= args.PF_patience and self.PF_round <= 2:
                self.PF_round = PF_fix(model, self.PF_round, epoch)

        elif args.PF_criterion == "force":
            if epoch + 1 == args.PF_epoch_1:
                self.PF_round = PF_fix(model, self.PF_round, epoch)
            if epoch + 1 == args.PF_epoch_2:
                self.PF_round = PF_fix(model, self.PF_round, epoch)

        elif args.PF_criterion == "GL":
            self.GL = 100 * (val_set_loss[-1] / min(val_set_loss) - 1)
            if self.GL > args.GL_alpha and self.PF_round <= 2:
                self.PF_round = PF_fix(model, self.PF_round, epoch)
        elif args.PF_criterion == "PQ":
            k = 5
            if len(training_set_loss) < k:
                return
            self.GL = 100 * (val_set_loss[-1] / min(val_set_loss) - 1)
            self.P_k = 1000 * (sum(training_set_loss[-k:]) / (k * min(training_set_loss[-k:])) - 1)
            self.PQ = self.GL / self.P_k
            if self.PQ > args.PQ_alpha and self.PF_round <= 2:
                self.PF_round = PF_fix(model, self.PF_round, epoch)
        elif args.PF_criterion == "UP":
            UP = 0
            k = 5
            if args.UP_alpha * k + 1 > len(val_set_loss):
                return
            for i in range(args.UP_alpha):
                if val_set_loss[-1 - i * k] > val_set_loss[-1 - (i + 1) * k]:
                    UP += 1
                else:
                    break
            if UP >= args.UP_alpha and self.PF_round <= 2:
                self.PF_round = PF_fix(model, self.PF_round, epoch)
