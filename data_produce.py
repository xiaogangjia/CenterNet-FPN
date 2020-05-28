import xml.etree.ElementTree as ET
import random

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_bbox(file):
    tree = ET.parse(file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    bbox = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        bbox.append([xmin,ymin,xmax,ymax,cls])

    return bbox


def voc_produce():
    anns_2007 = '/data/bitahub/VOC2007/Annotations/'
    anns_2012 = '/data/bitahub/VOC2012/Annotations/'

    train_2007 = '2007trainval.txt'
    train_2012 = '2012trainval.txt'
    test_txt  = 'test.txt'

    with open(train_2007) as f:
        files_2007 = f.readlines()
        files_2007 = [file.strip() for file in files_2007]

    with open(train_2012) as f:
        files_2012 = f.readlines()
        files_2012 = [file.strip() for file in files_2012]

    with open(test_txt) as f:
        test_files = f.readlines()
        test_files = [file.strip() for file in test_files]

    train_2007_detections = {}
    train_2012_detections = {}
    test_detections       = {}

    for img in files_2007:
        bbox = get_bbox(anns_2007 + img + '.xml')
        train_2007_detections[img] = bbox

    for img in files_2012:
        bbox = get_bbox(anns_2012 + img + '.xml')
        train_2012_detections[img] = bbox

    for img in test_files:
        bbox = get_bbox(anns_2007 + img + '.xml')
        test_detections[img] = bbox

    train_2007_detections.update(train_2012_detections)

    return train_2007_detections, test_detections


if __name__ == '__main__':
    voc_produce()
