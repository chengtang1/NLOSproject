import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import configs.TransFnet_configs as configs
from trainer import trainer
from networks.TransFnet import transfnet
import time


parser = argparse.ArgumentParser()
parser.add_argument('--train_auto_cro', type=str,
                    default=r'D:\tangcheng\physical-aware\physical-aware\data\iteror2_car\image', help='root dir for data')
parser.add_argument('--train_path_label', type=str,
                    default=r'D:\tangcheng\physical-aware\physical-aware\data\iteror2_car\label', help='root dir for data')
# parser.add_argument('--test_auto_cro', type=str,
#                     default='./data/finetune_mnist/image', help='root dir for data')
# parser.add_argument('--test_path_label', type=str,
#                     default=r'./data/finetune_mnist/label', help='root dir for data')
# hen you save the result
parser.add_argument('--test_auto_cro', type=str,
                    default=r'D:\tangcheng\physical-aware\physical-aware\data\iteror2_car\image', help='root dir for data')
parser.add_argument('--test_path_label', type=str,
                    default=r'D:\tangcheng\physical-aware\physical-aware\data\iteror2_car\label', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='mnsit', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_mnist', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=0,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=112, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results-previous', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='Trans-Uet-tc', help='[hiformer-s, hiformer-b, hiformer-l]')
parser.add_argument('--eval_interval', type=int,
                    default=1, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--do_load_checkpoint', type=bool,
                    default=True, help='z_spacing')
parser.add_argument('--save_model_path', type=str,
                    default=r'', help='model')
#预训练的模型
# parser.add_argument('--save_model_path', type=str,
#                     default='', help='model')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)



    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


    model = transfnet(num_classes=args.num_classes)#.cuda()
    # start_time = time.time()
    trainer(args, model, args.output_dir)
    # end_time = time.time()

    # print("cost %f second"% (end_time- start_time))
