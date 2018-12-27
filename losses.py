import torch
import torch.nn as nn

"""Loss calc:"""
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class dice_coef(nn.Module):
    def __init__(self):
        super(dice_coef,self).__init__()
        self.flatten = Flatten()

    def forward(self, y_pred, y_real, eps=0.00001):

        y_pred_f = self.flatten(y_pred)
        y_real_f = self.flatten(y_real)
        # binary values so sum the same as sum of squares
        interaction = 2 * torch.sum(torch.mul(y_pred_f,y_real_f))
        y_pred_sum = torch.sum(torch.pow(y_pred_f, 2))
        y_real_sum = torch.sum(y_real_f)
        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        dc = interaction / (y_pred_sum + y_real_sum + eps)
        return dc


class MultiClass_Dice_loss(nn.Module):
    def __init__(self, threshold=0.5):
        super(MultiClass_Dice_loss, self).__init__()
        self.threshold = threshold
        self.dice = dice_coef()

    def forward(self, net_out, target):
        """
        :param net_out:
        :param target:
        :return:
        """
        eps = 0.000001
        target = (target > self.threshold).float() * 1
        dcl = torch.zeros([net_out.shape[1], 1])
        for channel in range(net_out.shape[1]):  # [batch, channels, x, y]
            dcl[channel] = (self.dice(net_out[:, channel, ...], target[:, channel, ...]))

        out = 1 - (torch.mean(dcl))  # 1-D is better then -D
        return out

class MultiClass_Dice_acc(nn.Module):
    def __init__(self, threshold=0.5):
        super(MultiClass_Dice_acc, self).__init__()
        self.threshold = threshold
        self.dice = dice_coef()

    def forward(self, net_out, target):
        """
        :param net_out:
        :param target:
        :return:
        """
        eps = 0.000001
        target = (target > self.threshold).float() * 1
        net_out = (net_out > self.threshold).float() * 1
        dca = torch.zeros([net_out.shape[1], 1])
        for channel in range(net_out.shape[1]):  # [batch, channels, z, x, y]
            dca[channel] = (self.dice(net_out[:, channel, ...], target[:, channel, ...]))

        out = torch.mean(dca)
        return out