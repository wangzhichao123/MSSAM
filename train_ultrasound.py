import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataset_ultrasound import get_loader, test_dataset
from utils.polyp_utils import clip_gradient, adjust_lr, AvgMeter, diy_lr
import torch.nn.functional as F
import numpy as np
import logging
from torch.nn.modules.loss import CrossEntropyLoss
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from importlib import import_module

import matplotlib.pyplot as plt

ce_loss = CrossEntropyLoss()

def ce_dice(pred, mask, dice_weight=0.7, smooth=1e-8):
    ###########
    ce = ce_loss(pred, mask)
    ###########
    pred = torch.sigmoid(pred)           # activation
    intersection = (pred * mask).sum(dim=(2, 3))        # [B, 1, H, W] -> [B, 1]
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    loss = (1.0 - dice_weight) * ce + dice_weight * dice
    return loss.mean()
    

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 512)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        outputs = model(image, True, 512)
        
        mask = outputs['masks']    # iout = outputs['masks']
        decoder_mask = outputs['decoder_mask']
        
        res = mask + decoder_mask             # 融合
        # eval Dice
        res = F.interpolate(res, size=gt.shape, mode='bilinear')    # 还原GT
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1



def train(train_loader, model, optimizer, epoch, args):
    model.train()
    global best
    loss_P1_record, loss_P2_record = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        
        images, gts, low_res_gt = pack
        
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        low_res_gt = Variable(low_res_gt).cuda()
        
        # ---- forward ----
        multimask_output = False  # don't use
        outputs = model(images, multimask_output, args.trainsize)
            
        mask = outputs['low_res_logits']    # iout = outputs['masks'] # 128  
        diy_decoder_mask = outputs['diy_decoder_mask']
        
        P1 = mask
        P2 = diy_decoder_mask
        P3 = P1 + P2
        
        # ---- loss function ----
        loss_P1 = structure_loss(P1, low_res_gt)
        loss_P2 = structure_loss(P2, low_res_gt)
        # loss_P3 = structure_loss(P3, low_res_gt)         # Impact on Performance
        
        loss = loss_P1 + loss_P2
        # ---- backward ----
        optimizer.zero_grad()
        loss.backward()
        # clip_gradient(optimizer, args.clip)      # gradient clipping
        optimizer.step()
        # ---- recording loss ----
        loss_P1_record.update(loss_P1.data, args.batchsize)
        loss_P2_record.update(loss_P2.data, args.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:0.4f}, lateral-2: {:0.4f}]'.
                  format(datetime.now(), epoch, args.epoch, i, total_step,
                         loss_P1_record.show(), loss_P2_record.show()))
    # save model 
    save_path = (args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path +str(epoch)+ 'SAM_Polyp.pth')
    # choose the best model

    global dict_plot
   
    test1path = './data/BUSI/'
    if (epoch + 1) % 1 == 0:
        meandice = test(model, test1path, 'test')
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'SAM_BUSI-best-r8.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))


def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
    
if __name__ == '__main__':
    ##################model_name#############################
    model_name = 'SAM_BUSI'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300, help='epoch number')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    
    # If you want use the "box" prompt, please don't employ any enhanced strategies.
    parser.add_argument('--augmentation', default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')

    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str, default='./data/BUSI/train/', help='path to train dataset')

    parser.add_argument('--test_path', type=str, default='./data/BUSI/test/', help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str, default='./model_BUSI_pth/'+ model_name + '/')
    
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    
    parser.add_argument('--ckpt', type=str, default='sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    
    parser.add_argument('--warmup', default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # don't use
    
    parser.add_argument('--warmup_period', type=int, default=20, help='Warp up iterations, only valid whrn warmup is activated') # don't use

    args = parser.parse_args()
    logging.basicConfig(filename='train_BUSI_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.trainsize,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    low_res = img_embedding_size * 4  # 32 * 4 = 128  32 * 7 = 224 
    
    net = sam.cuda()
    pkg = import_module(args.module)
    model = pkg.LoRA_Sam(sam, 8).cuda()  # r = 8

    best = 0

    params = model.parameters()

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.trainsize,
                              augmentation=args.augmentation)
    
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, args.epoch):
        adjust_lr(optimizer, args.lr, epoch, 0.1, 300)
        # diy_lr(args, epoch, args.lr, optimizer)
        train(train_loader, model, optimizer, epoch, args)
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
