import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from centernet.resnet_fpn import resnet18
from centernet.fpn_loss import Loss
from centernet.config import cfg
from data_produce import voc_produce
from fpn_dataset import VOC_data
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    img_dir = 'dataset/VOCdevkit/VOC2007/JPEGImages/'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trains, tests = voc_produce()

    trainset = VOC_data(trains, img_dir)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    testset = VOC_data(tests, img_dir, test_flag=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)

    net = resnet18(20, pretrained=False)
    net.load_state_dict(torch.load('/model/DavidJia/resnet/resnet18-5c106cde.pth'), strict=False)
    
    if torch.cuda.device_count() > 1:
        print('using ', torch.cuda.device_count(), 'GPUs!')
        net = nn.DataParallel(net)
    else:
        print('using single GPU')

    pre_model = cfg['pre_model']
    if pre_model != None:
        net.module.load_state_dict(torch.load(pre_model))
        start_iter = cfg['start_iter']
    else:
        start_iter = 0

    writer = SummaryWriter('/output/logs')

    net.to(device)
    net.train()

    loss = Loss(center_weight=1.0, size_weight=0.1, off_weight=1.0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.00025)

    train_loss = 0
    test_loss = 0
    for epoch in range(start_iter, 70):
        print('num of epoch: ' + str(epoch))

        if epoch == 45:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
        
        if epoch == 60:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
    
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            length = len(trainloader)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = [y.to(device) for y in labels]

            center_heats, obj_size = net(inputs)
            preds = [center_heats, obj_size]

            all_loss = loss(preds, labels)
            all_loss = all_loss.mean()
            all_loss.backward()

            optimizer.step()
            print("training step {}: ".format(i), all_loss.item())

            train_loss += all_loss.item()
            if (i + 1) % 15 == 0:
                writer.add_scalar('training loss',
                                  train_loss / 15,
                                  i + epoch * length)
                train_loss = 0
            
        if epoch > 40:
            torch.save(net.module.state_dict(), '/output/{}_params.pkl'.format(epoch))
        
        net.eval()
        with torch.no_grad():
            test_num = 0
            for data in testloader:
                test_num += 1

                inputs, labels = data
                inputs = inputs.to(device)
                labels = [y.to(device) for y in labels]

                center_heats, obj_size = net(inputs)
                preds = [center_heats, obj_size]

                all_loss = loss(preds, labels)
                all_loss = all_loss.mean()
                test_loss += all_loss.item()

            writer.add_scalar('test loss', test_loss / test_num, epoch)
            test_loss = 0
        net.train()
