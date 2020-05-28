import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

from data_produce import classes, voc_produce
from dataset import VOC_data, test_data
from centernet.resnet_fpn import resnet18
from centernet.utils import _nms, _topk
import time

device = torch.device('cuda:0')

img_dir = 'dataset/VOCdevkit/VOC2007/JPEGImages/'

transform = transforms.Compose([
    transforms.ToTensor()
])

trains, tests = voc_produce()

trainset = VOC_data(trains, img_dir, transform=transform, test_flag=True)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

testset = test_data(tests, img_dir, if_flip_test=True)
testloader = DataLoader(testset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def flip_heatmap(tensor):
    flip_tensor = tensor.detach().cpu().numpy()
    flip_tensor = flip_tensor[:, :, :, ::-1].copy()
    flip_tensor = torch.from_numpy(flip_tensor)
    flip_tensor = flip_tensor.to(device)

    return flip_tensor


if __name__ == '__main__':
    #inference('test/004819.jpg')

    net = resnet18(20)
    net.load_state_dict(torch.load('fpn_models/resnet.pkl'))

    net.to(device)
    net.eval()

    img_num = 0
    for i, data in enumerate(testloader):
        images, flip_images, img_names = data
        images = images.to(device)
        flip_images = flip_images.to(device)

        outs = net(images)
        flip_outs = net(flip_images)

        batch = len(outs[0])

        center_heat_1 = torch.clamp(outs[0], min=1e-4, max=1 - 1e-4)
        center_heat_2 = torch.clamp(outs[2], min=1e-4, max=1 - 1e-4)
        center_heat_3 = torch.clamp(outs[4], min=1e-4, max=1 - 1e-4)

        flip_center_heat_1 = torch.clamp(flip_outs[0], min=1e-4, max=1 - 1e-4)
        flip_center_heat_2 = torch.clamp(flip_outs[2], min=1e-4, max=1 - 1e-4)
        flip_center_heat_3 = torch.clamp(flip_outs[4], min=1e-4, max=1 - 1e-4)

        flip_center_1 = flip_heatmap(flip_center_heat_1)
        flip_center_2 = flip_heatmap(flip_center_heat_2)
        flip_center_3 = flip_heatmap(flip_center_heat_3)

        flip_size_1 = flip_heatmap(flip_outs[1])
        flip_size_2 = flip_heatmap(flip_outs[3])
        flip_size_3 = flip_heatmap(flip_outs[5])

        center_heat_1 = (center_heat_1 + flip_center_1) / 2
        center_heat_2 = (center_heat_2 + flip_center_2) / 2
        center_heat_3 = (center_heat_3 + flip_center_3) / 2

        flip_obj_size_1 = (outs[1] + flip_size_1) / 2
        flip_obj_size_2 = (outs[3] + flip_size_2) / 2
        flip_obj_size_3 = (outs[5] + flip_size_3) / 2

        center_heat_1 = _nms(center_heat_1, kernel=3)
        center_heat_2 = _nms(center_heat_2, kernel=3)
        center_heat_3 = _nms(center_heat_3, kernel=3)

        center_heats = [center_heat_1, center_heat_2, center_heat_3]
        obj_size = [flip_obj_size_1, flip_obj_size_2, flip_obj_size_3]

        for i in range(batch):
            img_name = img_names[i]
            print(img_name)
            out_file = open('dataset/detections/%s.txt' % (img_name), 'w')
            for j, center_heat in enumerate(center_heats):
                obj_scores, obj_clses, obj_ys, obj_xs = _topk(center_heat, K=100)

                scale = 2**j
                _obj_size = obj_size[j]
                for obj_num in range(15):
                    confidence = float(obj_scores[i, obj_num])
                    center_x = int(obj_xs[i, obj_num])
                    center_y = int(obj_ys[i, obj_num])

                    category = obj_clses[i, obj_num]
                    x_offset, y_offset = _obj_size[i, 2:4, center_y, center_x]
                    width, height = _obj_size[i, 0:2, center_y, center_x]

                    tlx = int((center_x + x_offset - width / 2) * 8 * scale)
                    tly = int((center_y + y_offset - height / 2) * 8 * scale)
                    brx = int((center_x + x_offset + width / 2) * 8 * scale)
                    bry = int((center_y + y_offset + height / 2) * 8 * scale)

                    box = [tlx, tly, brx, bry]

                    out_file.write(classes[category] + " " + str(confidence) + " " + " ".join([str(a) for a in box]) + '\n')

        print('the number is: ' + str(img_num))
        img_num += 2

    print('done with the detection')

    '''
    net = Net()
    net.load_state_dict(torch.load('models/69_499_params.pkl'))

    net.to(device)
    net.eval()

    img_num = 0
    for i, data in enumerate(testloader):
        inputs, labels, img_names = data
        inputs = inputs.to(device)
        labels = [y.to(device) for y in labels]

        center_heat_1, center_heat_2, center_heat_3, offsets, obj_size = net(inputs)

        center_heat_1 = torch.clamp(torch.sigmoid(center_heat_1), min=1e-4, max=1 - 1e-4)
        center_heat_2 = torch.clamp(torch.sigmoid(center_heat_2), min=1e-4, max=1 - 1e-4)
        center_heat_3 = torch.clamp(torch.sigmoid(center_heat_3), min=1e-4, max=1 - 1e-4)

        center_heat_1 = _nms(center_heat_1, kernel=3)
        center_heat_2 = _nms(center_heat_2, kernel=3)
        center_heat_3 = _nms(center_heat_3, kernel=3)

        obj_scores = {}
        obj_clses = {}
        obj_ys = {}
        obj_xs = {}

        obj_scores[1], obj_clses[1], obj_ys[1], obj_xs[1] = _topk(center_heat_1, K=30)
        obj_scores[2], obj_clses[2], obj_ys[2], obj_xs[2] = _topk(center_heat_2, K=30)
        obj_scores[3], obj_clses[3], obj_ys[3], obj_xs[3] = _topk(center_heat_3, K=30)

        batch = len(center_heat_1)

        for i in range(batch):
            img_name = img_names[i]
            print(img_name)
            out_file = open('dataset/detections/%s.txt' % (img_name), 'w')
            for num in range(3):
                for obj_num in range(30):
                    if obj_scores[num + 1][i, obj_num] > 0.1:
                        confidence = float(obj_scores[num + 1][i, obj_num])
                        center_x = int(obj_xs[num + 1][i, obj_num])
                        center_y = int(obj_ys[num + 1][i, obj_num])

                        category = obj_clses[num + 1][i, obj_num]
                        x_offset, y_offset = offsets[i, :, center_y, center_x]
                        width, height = obj_size[i, :, center_y, center_x]

                        tlx = int((center_x + x_offset - width / 2) * 4)
                        tly = int((center_y + y_offset - height / 2) * 4)
                        brx = int((center_x + x_offset + width / 2) * 4)
                        bry = int((center_y + y_offset + height / 2) * 4)

                        box = [tlx, tly, brx, bry]

                        out_file.write(classes[category] + " " + str(confidence) + " " + " ".join([str(a) for a in box]) + '\n')
                    else:
                        break

        print('the number is: ' + str(img_num))
        img_num += 2

    print('done with the detection')
    '''
