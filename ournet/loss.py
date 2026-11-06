import torch
from torch import nn
import torch.nn.functional as F

class dice_bce_loss(nn.Module):

    def __init__(self, batch=True, num_classes=1):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-6
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

class dice_ce_loss(nn.Module):

    def __init__(self, batch=True, num_classes=3):
        super(dice_ce_loss, self).__init__()
        self.batch = batch
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def one_hot_encode(self, y_true, num_classes):
        y_true_one_hot = torch.zeros(y_true.size(0), num_classes, y_true.size(2), y_true.size(3)).to(y_true.device)
        y_true_one_hot.scatter_(1, y_true.long(), 1)
        return y_true_one_hot

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-601
        if self.batch:
            intersection = torch.sum(y_true * y_pred, dim=(0, 2, 3))
            i = torch.sum(y_true, dim=(0, 2, 3))
            j = torch.sum(y_pred, dim=(0, 2, 3))
        else:
            intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
            i = torch.sum(y_true, dim=(1, 2, 3))
            j = torch.sum(y_pred, dim=(1, 2, 3))
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, y_pred, y_true):
        y_true_one_hot = self.one_hot_encode(y_true, self.num_classes)
        ce_loss = self.ce_loss(y_pred, y_true.squeeze(1).long())
        y_pred_softmax = F.softmax(y_pred, dim=1)
        dice_loss = self.soft_dice_loss(y_true_one_hot, y_pred_softmax)
        return ce_loss + dice_loss

class B2B1E1Loss(nn.Module):

    def __init__(self, batch=True, num_classes=5):
        super(B2B1E1Loss, self).__init__()
        self.B2Loss = dice_ce_loss(num_classes=num_classes)
        self.B1Loss = dice_bce_loss()
        self.l1loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        b2loss = self.B2Loss(y_pred['roadocc_masks'], y_true['label'])
        pc = y_pred['roadwo_edge_masks'].type(torch.float32)
        dc = y_true['dist_maps'].type(torch.float32)
        edgeloss = self.l1loss(pc, dc)
        roadwo_mask_true = (y_true['label'] >= 1).float()
        b1loss = self.B1Loss(y_pred['roadwo_masks'], roadwo_mask_true)
        return b2loss + b1loss + edgeloss
