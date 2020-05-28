import numpy as np
import torch
import torch.nn as nn
import math


def draw_center(heatmap, center, obj_size, ratio=0.8):
    x, y = center
    height, width = obj_size

    w_range = math.floor(width * ratio / 2)
    h_range = math.floor(height * ratio / 2)

    ness = np.ones((h_range+1, w_range+1), dtype=np.float32)
    for i in range(h_range+1):
        for j in range(w_range+1):
            l = (width / 2) - w_range + j
            r = (width / 2) + w_range - j
            t = (height / 2) - h_range + i
            b = (height / 2) + h_range - i

            ness[i, j] = np.sqrt((l/r) * (t/b))

    origin_heatmap = heatmap[y-h_range:y+h_range+1, x-w_range:x+w_range+1]
    mask_heatmap = np.zeros((2*h_range+1, 2*w_range+1), dtype=np.float32)

    mask_heatmap[0:h_range+1, 0:w_range+1] = ness
    mask_heatmap[h_range:2*h_range+1, 0:w_range+1] = ness[::-1, :]
    mask_heatmap[0:h_range+1, w_range:2*w_range+1] = ness[:, ::-1]
    mask_heatmap[h_range:2*h_range+1, w_range:2*w_range+1] = ness[::-1, ::-1]
    '''
    mask_heatmap[y-h_range:y+1, x-w_range:x+1] = ness
    mask_heatmap[y:y+h_range+1, x-w_range:x+1] = ness[::-1, :]
    mask_heatmap[y-h_range:y+1, x:x+w_range+1] = ness[:, ::-1]
    mask_heatmap[y:y+h_range+1, x:x+w_range+1] = ness[::-1, ::-1]
    '''
    np.maximum(origin_heatmap, mask_heatmap, out=origin_heatmap)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    '''
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    
    return min(r1, r2, r3)
    '''
    return r1


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_clses, topk_ys, topk_xs

