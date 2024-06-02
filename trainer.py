import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
#导入均方根误差
from torch.nn.modules.loss import MSELoss,L1Loss
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, test_single_volume,test_evaluate
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch.nn.functional as F
import cv2
from datasets.dataset_mnist import MyDataset, RandomGenerator
import torchvision.transforms as transforms
from loss_function import SSIM,calculate_psnr,physical_auto
# from audtorch.metrics.functional import  pearsonr
from scipy.stats import stats

# 计算指标 psnr mae ssim TODO: 不敢保证没问题，还没调试过测算指标的
def inference_mnist(model, testloader, args, test_save_path,epoch_num =None):
    model.eval()
    metric_list = 0.0
    ssim_list = []
    psnr_list = []
    mae_list = []
    epoch_num = epoch_num
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_bacth, label_batch = sampled_batch
        ssim_i,psnr_i,mae_i = test_evaluate(image_bacth,label_batch,model,test_save_path=test_save_path,epoch=epoch_num)
        ssim_list.append(ssim_i)
        psnr_list.append(psnr_i)
        mae_list.append(mae_i)
        if i_batch % 20 == 0:
            logging.info(' idx %d case  mean_ssim %f mean_psnr %f mean_mae %f' % (
                i_batch, np.mean(ssim_i), np.mean(psnr_i),np.mean(mae_i)))

    metric_list = metric_list/len(testloader.dataset)
    mean_ssim = np.mean(ssim_list)
    mean_psnr = np.mean(psnr_list)
    mean_mae = np.mean(mae_list)
    logging.info('Testing performance in best val model: mean_ssim : %f mean_psnr : %f mean_mae : %f' % (mean_ssim, mean_psnr , mean_mae))
    return mean_ssim,mean_psnr,mean_mae

def inference(model, testloader, args, test_save_path):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        # 怎么去把每个类给表现出来 TODO：基本上得重写
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (
        i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(testloader.dataset)

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

    return performance, mean_hd95


def plot_result(ssim, psnr,mae,flitting_error,snapshot_path, args):
    dict = {'mean_ssim': ssim, 'mean_psnr': psnr,'mean_mae': mae,'preidcted_error':flitting_error}
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_ssim'].plot()
    resolution_value = 1200
    plt.title('Mean ssim')
    date_and_time = datetime.datetime.now().date()
    filename = f'{args.model_name}_' + str(date_and_time) + 'ssim' + '.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_psnr'].plot()
    plt.title('Mean psnr')
    filename = f'{args.model_name}_' + str(date_and_time) + 'psnr' + '.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(2)
    df['mean_mae'].plot()
    plt.title('Mean mae')
    filename = f'{args.model_name}_' + str(date_and_time) + 'mae' + '.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(3)
    df['preidcted_error'].plot()
    plt.title('preidcted error')
    filename = f'{args.model_name}_' + str(date_and_time) + 'preidcted_error' + '.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    # save csv
    filename = f'{args.model_name}_' + str(date_and_time) + 'results-previous' + '.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')

def trainer(args, model, snapshot_path):
    date_and_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True) # 保存模型
    test_save_path = os.path.join(snapshot_path, 'test')

    # Save logs
    logging.basicConfig(filename=snapshot_path + f"/{args.model_name}" + str(date_and_time) + "_log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    # db_train = MyDataset(dataset_path_auto_cro=args.train_auto_cro,dataset_path_label=args.train_path_label, transforms=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    #不加扰动
    db_train = MyDataset(dataset_path_auto_cro=args.train_auto_cro,dataset_path_label=args.train_path_label)

    db_test = MyDataset(dataset_path_auto_cro=args.test_auto_cro,dataset_path_label=args.test_path_label)

    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False)

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.cuda()
    model.train()

    if args.do_load_checkpoint:
        if args.save_model_path == '':
            print("the pretrain model path is empty")

        else:
            model.load_state_dict(torch.load(args.save_model_path))
# define the loss function

    mse_loss = MSELoss(reduction='mean')
    mae_loss =  L1Loss(reduction='mean')
    kl_loss =  nn.KLDivLoss()

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)  # 出现梯度消失
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr,weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    psnr_ = []
    mae_ = []
    ssim_ = []
    predicted_error= []
    flitting_error = []
    #TODO:调试作用
    # mean_psnr, mean_mae, mean_ssim = inference_mnist(model, testloader, args, test_save_path=test_save_path)
    for epoch_num in iterator:
        for sampled_batch in trainloader:
            image_batch, label_batch = sampled_batch

            image_batch = image_batch.to(torch.float32).cuda()
            label_batch = label_batch.to(torch.float32).cuda()


            #  loss 为什么这么大？ 训练不出来啊 添加物理模型  自相关回去:reason 图片对应关系是错乱的，是完全一个batch内反着的
            # 查一下：图像重建一般用啥loss啊,感觉像mse,kl 这种一致性的loss，数值太大了。加入psnr
            outputs = model(image_batch)
            outputs = torch.reshape(outputs,(outputs.shape[0],outputs.shape[2],outputs.shape[2]))

            Max = torch.max(torch.max(outputs))
            Min = torch.min(torch.min(outputs))
            outputs = (outputs - Min) / (Max - Min)

            auto_cro = physical_auto(outputs)
            # auto_cro_label = physical_auto(label_batch)
            loss_mse = mse_loss(outputs, label_batch)

            loss_phy = mae_loss(auto_cro, image_batch)

            # loss_phy = kl_loss(auto_cro,image_batch)
            #pearson相关系数
            var_a = torch.var(auto_cro.reshape(-1,1),unbiased=False)
            var_b = torch.var(image_batch.reshape(-1, 1),unbiased=False)
            npcc = torch.cov(torch.cat((auto_cro.reshape(1,-1),auto_cro.reshape(1,-1)),dim=0),correction=0)
            npcc = -1*npcc[0][1]/var_a/var_b
            loss = ( 0.1*loss_mse+npcc).to(torch.float32)
# add the flitting error
            # 无监督的微调 phy_aware loss
            # loss = (npcc).to(torch.float32)

            # loss = loss_phy.to(torch.float32)

            error = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_mse', loss_mse, iter_num)
            writer.add_scalar('info/loss_phy', loss_phy, iter_num)

            logging.info('iteration %d : loss : %f, loss_mse: %f loss_phy: %f' % (
            iter_num, loss.item(), loss_mse.item(), loss_phy.item()))

            try:
                if iter_num % 10 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    # TODO：我们output需要argmax吗？
                    # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
            except:
                pass

        # Test TODO: 应该存在问题，不是dice指标 换成psnr 和 mse,测试时间 把下面注释
        if (epoch_num + 1) % args.eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")

            #计算指标  psnr SSIM  MAE
#add the every epoch loss
            mean_ssim, mean_psnr,mean_mae= inference_mnist(model, testloader, args, test_save_path=test_save_path,epoch_num = epoch_num)
            psnr_.append(mean_psnr)
            mae_.append(mean_mae)
            ssim_.append(mean_ssim)
            error = error.cpu().detach().numpy()
            flitting_error.append(error)


            model.train()

        # if epoch_num >= max_epoch - 1 or epoch_num == 40:
        #     filename = f'{args.model_name}_epoch_{epoch_num}.pth'
        #     save_mode_path = os.path.join(snapshot_path, filename)
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #
        #     if not (epoch_num + 1) % args.eval_interval == 0:
        #         logging.info("*" * 20)
        #         logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
        #         print(f"Epoch {epoch_num}, Last Epcoh")
        #         mean_ssim, mean_psnr, mean_mae = inference_mnist(model, testloader, args, test_save_path=test_save_path,epoch_num = epoch_num)
        #         psnr_.append(mean_psnr)
        #         mae_.append(mean_mae)
        #         ssim_.append(mean_ssim)
        #
        #     iterator.close()
        #     break
    flitting_error = np.array(flitting_error)
    plot_result(ssim_, psnr_, mae_,flitting_error,snapshot_path, args)
    writer.close()
    return "Training Finish]ed!"