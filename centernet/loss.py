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
        center_heats = pred[0]
        obj_size     = pred[1]

        gt_center_heats = gt[0]
        gt_obj_size     = gt[1]
        gt_pos          = gt[2]
        gt_obj_mask     = gt[3]

        batch = len(center_heats)
        # refined focal loss
        focal_loss = 0
        center_heats = torch.clamp(center_heats, min=1e-4, max=1-1e-4)

        focal_loss += self.focal_loss(center_heats, gt_center_heats)
        focal_loss = self.center_weight * focal_loss

        # size loss
        size_loss = 0

        size_loss += self.regre_loss(obj_size, gt_obj_size, gt_pos, gt_obj_mask)
        size_loss = self.size_weight * size_loss

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


def regre_loss(pred, gt, pos, mask):
    loss = 0

    batch = gt.size()[0]
    pos = pos.gt(0)
    pos_inds = pos.expand(batch, 4, 96, 96)

    obj_num = mask.float().sum()

    x = pred[pos_inds]
    gt_x = gt[pos_inds]

    x_loss = nn.functional.smooth_l1_loss(x, gt_x, size_average=False)

    loss += x_loss / (obj_num + residual)

    return loss

