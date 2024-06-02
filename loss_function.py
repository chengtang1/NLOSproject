import os

import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def type_trans(window,img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mcs_map  = (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        h, w = img1.shape[0],img1.shape[1]
        b=1
        img1 = torch.reshape(img1,(b,1,h,w))
        img2 = torch.reshape(img2,(b,1,h,w))
        channel = 1
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)
        ssim_map, mcs_map =_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = np.array(img1.cpu().detach().numpy())
    img2 =np.array(img2.cpu().detach().numpy())
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11,size_average = True):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # self.channel = 3

    def forward(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

        msssim = Variable(torch.Tensor(levels,))
        mcs    = Variable(torch.Tensor(levels,))

        if torch.cuda.is_available():
            weight =weight.cuda()
            msssim=msssim.cuda()
            mcs=mcs.cuda()

        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)

        for i in range(levels): #5 levels
            ssim_map, mcs_map = _ssim(img1, img2,window,self.window_size, channel, self.size_average)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            # print(img1.shape)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1 #refresh img
            img2 = filtered_im2

        return torch.prod((msssim[levels-1]**weight[levels-1] * mcs[0:levels-1]**weight[0:levels-1]))
        # return torch.prod((msssim[levels-1] * mcs[0:levels-1]))
        #torch.prod: Returns the product of all elements in the input tensor

def physical_auto(F,img=None):
    ## 默认是取最后两个维度傅里叶变化 但是现在导致一个bacth内的图片全部倒置了 TODO: 解决办法，拆开来做 fft 然后再cat 一起 注意output是bx1xhxw，记得去转换一下
    bacth_size,w = F.shape[0],F.shape[-1]
    temp = torch.zeros((1,w,w)).cuda()
    for i in range(bacth_size):
        f_1 = F[i,:,:]
        f_1 = torch.fft.fft2(f_1)
        auto_corr = torch.fft.fftshift(torch.fft.ifft2(f_1 * torch.conj(f_1)))
        phy_1 = torch.abs(auto_corr)
        Max = torch.max(torch.max(phy_1))
        Min = torch.min(torch.min(phy_1))
        phy_1 = (phy_1 - Min) / (Max - Min)
        phy_1 = torch.reshape(phy_1,(1,w,w))
        temp = torch.cat((temp,phy_1),dim=0)
    #迭代的时候使用
    # phy_1 = temp[1:,:,:]
    phy_1 = temp[1:, :, :]
    path = r'D:\tangcheng\physical-aware\physical-aware'
    path_label = r'D:\tangcheng\physical-aware\physical-aware'
    # path_img = r'C:\tangcheng\PATN\physical-aware'
    for i in range(1):
        phy_1_0 = np.array(phy_1[i,:,:].detach().cpu().numpy())
        label_1_0 = np.array(F[i,:,:].detach().cpu().numpy())
        Max = np.max(np.max(phy_1_0 ))
        Min = np.min(np.min(phy_1_0 ))
        phy_1_0 = (phy_1_0  - Min) / (Max - Min)*255
        Max = np.max(np.max(label_1_0))
        Min = np.min(np.min(label_1_0 ))
        label_1_0= (label_1_0  - Min) / (Max - Min)*255
        path = os.path.join(path, str(i)+'cro_cat' + '.png')
        path_label = os.path.join(path_label, str(i) +'output_cat '+'.png')
        # path_img = os.path.join(path_img, str(i) +'img_cat '+'.png')

        cv2.imwrite(path, phy_1_0)
        cv2.imwrite(path_label, label_1_0)
        # cv2.imwrite(path_img, img_1_0)
    return phy_1

