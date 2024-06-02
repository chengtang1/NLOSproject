# 全部使用python来做自相关吧

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# 把我们的label和image 读取进来看对不对
import numpy as np
import os
import cv2
from PIL import  Image
# 感觉图片对应位置不对   #TODO： 怎么解决啊！！！！！！！ ：  尝试处理自相关和存储图片一起进行  全用python处理吧
# 数据归一化存储才行 或者读取的时候归一化
class phy_constraint(nn.Module):
    def __init__(self, F):
        super().__init__()
    def forward(self,F):
        f_1 = torch.fft.fft2(F)   # 默认是取最后两个维度傅里叶变化 torch.mutiply()对应元素相乘
        auto_corr = torch.fft.fftshift(torch.fft.ifft2(f_1*torch.resolve_conj(f_1)))
        auto_corr = torch.abs(auto_corr)
        # 归一化

        return auto_corr

def phy_cro(F):
    f_1 = torch.fft.fft2(F)  # 默认是取最后两个维度傅里叶变化 torch.mutiply()对应元素相乘
    auto_corr = torch.fft.fftshift(torch.fft.ifft2(f_1 * torch.resolve_conj(f_1)))
    auto_corr = torch.abs(auto_corr)
    # 归一化

    return auto_corr
# 数据读取的问题
img = Image.open("./data/Image/test/0/1.png")
label_1 = Image.open("./data/label/test/0/1.png")
path = r'C:\tangcheng\PATN\physical-aware'
# # 或者是img=cv2.imread("./img/cat1.jpg")
# # 又或者是#img=matplotlib.image.imread("./img/cat1.jpg")
# # 2：读取一些图片信息，比如图像的宽，高，通道数，最大像素值，最小像素值
label_1 = torch.from_numpy(np.array((label_1)))   # 转换成tensor 方便执行函数
phy_1= np.array(phy_cro(label_1))
path = os.path.join(path,str(2)+'.png')
# 还得归一化才行
phy_1=cv2.resize(np.array(phy_1),(112,112))
Max = np.max(np.max(phy_1))
Min = np.min(np.min(phy_1))
phy_1 = (phy_1 - Min) / (Max - Min)*255
cv2.imwrite(path,phy_1)
print(np.mean(np.abs(phy_1-img)))
# print(img.shape)
# print(img.shape[0])
# print(img.shape[1])
# print(img.mean())
# print(img.min(), img.max())
# # 3：显示图片
#
# plt.figure(figsize=(10,10))
# plt.subplot(121);plt.imshow(img)
# plt.subplot(122);plt.imshow(phy_1)
# plt.show()

