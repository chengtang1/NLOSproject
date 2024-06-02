import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        sample = {'image': image, 'label': label}
        return sample


class MyDataset(Dataset):
    def __init__(self, dataset_path_auto_cro,dataset_path_label, num_class=1, transforms=None):
        super(MyDataset,self).__init__()
        images = []
        labels = []
        #获得image
        txt_path_auto_cro = self.dataset2txt(dataset_path_auto_cro,num_class)
        txt_path_label = self.dataset2txt(dataset_path_label,num_class)

        #把image 和label 放到对应的位置上
        with open(txt_path_auto_cro, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) > num_class:
                    break
                line = line.strip('\n')
                images.append(line)    # 获得image 的txt列表

        with open(txt_path_label, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) > num_class:
                    break
                line = line.strip('\n')
                labels.append(line)    #label对应 的txt列表 按照顺序读取
                # labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])
        if self.transforms is not None:
            sample = {'image': image, 'label': label}
            sample = self.transforms(sample)
            image,label = sample['image'], sample['label']
        image = np.array(image).astype('float32')
        label = np.array(label).astype('float32')
        # 归一化
        Max = np.max(np.max(image))
        Min = np.min(np.min(image))
        image = (image - Min) / (Max - Min)

        Max = np.max(np.max(label))
        Min = np.min(np.min(label))
        label = (label - Min) / (Max - Min)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image, label

    def __len__(self):
        return len(self.labels)


    def dataset2txt(self,dataset_path, class_num=None):
        '''
        transform dataset into a txt file which contain every Image
        :param In_path: path of dataset
        :param num_class: classes
        :return:path of txt file
        '''

        # 1.创建文件
        # 一下两行代码目的是与数据集同级目录下新建dataset-text.txt文件
        txt_path = os.path.abspath(os.path.dirname(dataset_path))
        txt_path = txt_path + '/dataset-text.txt'
        # 删除已经存在的文件，要保证每次操作的文件是一个空的txt文件
        if os.path.exists(txt_path):
            os.remove(txt_path)

        f = open(txt_path, 'w')
        f.close()
        # 2.写入文件
        # 打开数据集，将主目录下所有文件夹放入list中
        dirs = os.listdir(dataset_path)
        # 将文件夹按从小到大排序，文件夹的名字是按照数字命名的，01，02，03...
        dirs.sort()

        # 打开第二级每个文件夹，将并将每个文件的绝对路径写入到上面新建的txt文件
        for i, dir in enumerate(dirs):
            file = os.path.abspath(dataset_path) + '/' + dirs[i]
            DIRLIST = os.listdir(file)
            for j, d in enumerate(DIRLIST):
                content = file + '/' + d + '\n'
                # 每次执行前一定确保要写入的文件是空的
                with open(txt_path, 'a') as f:
                    f.write(content)

        return txt_path
