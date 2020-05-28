import torch
import torch.nn as nn

residual = torch.tensor(1e-4)


class Loss(nn.Module):

    def __init__(self, center_weight=1.0, size_weight=0.1, off_weight=1.0):
        super(Loss, self).__init__()
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.off_weight  = off_weight

        self.focal_loss  = refined_focal_loss
        self.regre_loss  = regre_loss

    def forward(self, pred, gt):
        # 96*96
        heats_P2        = pred[0]
        obj_size_P2     = pred[1]

        # 48*48
        heats_P3        = pred[2]
        obj_size_P3     = pred[3]

        # 24*24
        heats_P4        = pred[4]
        obj_size_P4     = pred[5]

        center_heats_P2, gt_obj_size_P2, center_pos_P2 = gt[0:3]
        center_heats_P3, gt_obj_size_P3, center_pos_P3 = gt[3:6]
        center_heats_P4, gt_obj_size_P4, center_pos_P4 = gt[6:9]

        batch = len(heats_P2)
        # refined focal loss
        focal_loss_1 = 0
        focal_loss_2 = 0
        focal_loss_3 = 0

        heats_P2 = torch.clamp(heats_P2, min=1e-4, max=1-1e-4)
        heats_P3 = torch.clamp(heats_P3, min=1e-4, max=1-1e-4)
        heats_P4 = torch.clamp(heats_P4, min=1e-4, max=1-1e-4)

        focal_loss_1 += self.focal_loss(heats_P2, center_heats_P2) * self.center_weight
        focal_loss_2 += self.focal_loss(heats_P3, center_heats_P3) * self.center_weight
        focal_loss_3 += self.focal_loss(heats_P4, center_heats_P4) * self.center_weight

        focal_loss = focal_loss_1 + focal_loss_2 + focal_loss_3
        # size loss
        size_loss_1 = 0
        size_loss_2 = 0
        size_loss_3 = 0

        size_loss_1 += self.regre_loss(obj_size_P2, gt_obj_size_P2, center_pos_P2) * self.size_weight
        size_loss_2 += self.regre_loss(obj_size_P3, gt_obj_size_P3, center_pos_P3) * self.size_weight
        size_loss_3 += self.regre_loss(obj_size_P4, gt_obj_size_P4, center_pos_P4) * self.size_weight

        size_loss = size_loss_1 + size_loss_2 + size_loss_3

        print('focal loss: ' + str(focal_loss))
        print('size loss: ' + str(size_loss))

        loss = (focal_loss + size_loss) / batch

        return loss.unsqueeze(0)


def refined_focal_loss(pred, gt, alpha=2, beta=4):
    loss = 0
    batch = gt.size()[0]

    for i in range(batch):
        pos_inds = gt[i].eq(1)
        neg_inds = gt[i].lt(1)

        pos_pred = pred[i][pos_inds]
        pos_loss = torch.pow(1-pos_pred, alpha) * torch.log(pos_pred)

        neg_weight = torch.pow(1 - gt[i][neg_inds], beta)
        neg_pred = pred[i][neg_inds]
        neg_loss = neg_weight * torch.pow(neg_pred, alpha) * torch.log(1 - neg_pred)

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        num_pos = pos_inds.float().sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def regre_loss(pred, gt, pos):
    loss = 0

    batch = gt.size()[0]
    height, width = gt.size()[2:4]

    pos = pos.gt(0)

    pos_inds = pos.expand(batch, 4, height, width)

    x = pred[pos_inds]
    gt_x = gt[pos_inds]

    x_loss = nn.functional.smooth_l1_loss(x, gt_x, size_average=False)

    obj_num = len(x) / 4

    if obj_num == 0:
        loss = 0
    else:
        loss += x_loss / (obj_num + residual)

    return loss

