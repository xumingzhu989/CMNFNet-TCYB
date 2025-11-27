import torch
import argparse
from torch import nn
from util import dataset, transform, config
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from util.util import check_makedirs
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model import basicnetRes256NestFuse
import torch.optim as optim
import datetime
import math
import os
import pytorch_iou
import cv2
eps = math.exp(-10)
from torch_scatter import gather_coo


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cod_mgl50.yaml', help='config file')
    parser.add_argument('opts', help='see config/cod_mgl50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

# 0~1 normalization.
def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x


class MyLoss:
    def __init__(self):
        self.x = None
        self.y = None
        self.IOU = pytorch_iou.IOU(size_average=True)

    def loss(self, X, y):  # Q is B node_num n
        lossAll = 0
        for x in X:
            loss = (-y.mul(torch.log(x + eps)) - (1 - y).mul(torch.log(1 - x + eps))).sum()  # + (abs(x-y)).sum()
            num_pixel = y.numel()
            lossAll = lossAll + torch.div(loss, num_pixel) + self.IOU(x, y)

        return lossAll


def img_predict_tra(salmap, label):
    a = salmap[0].cpu()
    aa = a.detach().numpy()
    b = label[0].cpu()
    bb = b.detach().numpy()
    plt.subplot(121)
    plt.imshow(aa)
    plt.subplot(122)
    plt.imshow(bb)
    plt.show()


def loss_curve(counter, losses):
    fig = plt.figure()
    plt.plot(counter, losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number')
    plt.ylabel('loss')
    plt.show()


def train(loss_fn, args):
    train_losses = []
    train_counter = []

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Resize((args.img_h, args.img_w)),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_data = dataset.SemData(split='train', data_root=None, data_list=args.train_list, transform=train_transform)
    train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    # #define the graph model, loss function, optimizer.
    net = basicnetRes256NestFuse.CMNFNet(args.block_num, args.block_nod, args.img_dim, args.cov_loop,
                                      args.cov_bias, train_data, args.knn, res_pretrained=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0000,
                           amsgrad=False)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    date_str = str(datetime.datetime.now().date())  # 获取当前时间的日期
    # 训练模型
    model.train()
    lossAvg = 0
    # interNum = 10  # 120
    for epoch in range(args.epoch_num):
        for i, (input, gt, _) in enumerate(train_loader):
            input =input.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()
            sal1, sal2, sal3, sal4, sal5, sal6, sal7, sal8, sal9 = model(input)
            Out = [sal1, sal2, sal3, sal4, sal5, sal6, sal7, sal8, sal9]
            gt = MaxMinNormalization(gt)
            loss = loss_fn.loss(Out, gt)
            loss.backward()
            optimizer.step()
            # if ((i + 1) % interNum == 0) or (i + 1 == len(train_loader)):  # 每处理10个batch的数据显示一次检测结果
            if (i + 1 == len(train_loader)):
                img_predict_tra(sal1, gt)
                print('Train Epoch: {} [{:.0f}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(epoch + 1, (i + 1) * len(input),
                                                                                len(train_data),
                                                                                100. * (i + 1) * args.train_batch_size / len(train_data),
                                                                                lossAvg / (len(train_data)/args.train_batch_size)))
                lossAvg = 0
            lossAvg = lossAvg + loss.item()
            train_losses.append(loss.item())
            train_counter.append(((epoch) * len(train_data) / args.train_batch_size) + i)
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        scheduler.step()
        if ((epoch + 1) % 1 == 0) or (epoch + 1 == args.epoch_num):
            model_folder = args.model_path + date_str + '/'
            check_makedirs(model_folder)
            torch.save(model.state_dict(), model_folder + 'CMNFNet-resnet-' + str(epoch + 1) + '.pth')
    loss_curve(train_counter, train_losses)


def test(args):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    test_transform = transform.Compose([
        transform.Resize((args.img_h, args.img_w)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    to_pil_img = transforms.ToPILImage()

    date_str = str(datetime.datetime.now().date())
    results_folder = args.results_folder + date_str

    test_data = dataset.SemData(split='test', data_root=None, data_list=args.test_list, transform=test_transform)
    img_path_list = test_data.data_list
    img_name_list = []
    n_imgs = len(test_data)
    for i in range(n_imgs):
        img_name = img_path_list[i][1].split('/')[-1]  # img_path_list[i][0] is the image path, img_path_list[i][1] is the gt path
        img_name_list.append(img_name)

    test_sampler = None
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=test_sampler,
                                               drop_last=True)
    net = basicnetRes256NestFuse.CMNFNet(args.block_num, args.block_nod, args.img_dim, args.cov_loop,
                                                args.cov_bias, test_data, args.knn, res_pretrained=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epoch_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for now in epoch_list:
        model = net.to(device)
        model_dir = args.model_path + '2025-11-22/' + 'CMNFNet-resnet-' + str(now) + ".pth"  # the path/file to save the trained model params.
        results_folder_now = results_folder + "/" + 'CMNFNet-resnet-' + str(now)
        model.load_state_dict(torch.load(model_dir))
        print('The network parameters are loaded!')

        model.eval()
        for i, (input, _, img_size) in enumerate(test_loader):
            input = input.to(device)
            sal1, sal2, sal3, sal4, sal5, sal6, sal7, sal8, sal9 = model(input)
            n_img, _, _ = sal1.size()
            for j in range(n_img):
                salmaps = to_pil_img(sal1[j].cpu())
                salmaps = salmaps.resize((int(img_size[j][1]), int(img_size[j][0])))  # PIL.resize(width, height)
                file_name = img_name_list[i * args.test_batch_size + j]  # get the corresponding image name.
                if not os.path.isdir(results_folder_now):
                    os.makedirs(results_folder_now)
                salmaps.save(results_folder_now + '/' + file_name)
                print('Testing {} th image'.format(i * args.test_batch_size + j))



if __name__ == '__main__':
    args = get_parser()
    Flag_train_test = 1 # 0 is train; and 1 is test.
    if Flag_train_test == 0:
        criterion = MyLoss()
        train(loss_fn=criterion, args=args)
    else:
        ##########################################################################################
        # to test the network.
        test(args=args)