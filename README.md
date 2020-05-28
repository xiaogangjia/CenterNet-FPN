# CenterNet-FPN
This is an implementation of multi-scale CenterNet based on FPN. The results are verified on VOC2007.

Here I do not use DCNv2 and deconvolution. The feature fusion method follows a standard FPN. The model
is testd on VOC2007. Single-scale model only utilizes feature C2 to predict. Multi-scale model utilizes
C2, C3, C4 to predict. In addition, I adopt the training trick in SNIP and Trident Networks. Objects of
different scales are assigned to C2, C3 and C4 respectively.


| Model               | resolution    |  AP  |
| --------            | :-------:   | :----: |
| resnet18-fpn        | 384      |   72.46    |
| resnet18-fpn        | 384      |   74.97    |
| resnet101-fpn       | 384      |   79.16    |

