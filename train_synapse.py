import argparse
import logging
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from trainer import trainer_synapse

from torchsummaryX import summary
from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
# from ptflops import get_model_complexity_info


parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate') #0.001
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input') #224
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--ckpt', type=str, default='sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
args = parser.parse_args()


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
    
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    
    
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    args.exp = 'MSSAM_w3_7_' + dataset_name + str(args.img_size)
    snapshot_path = "model_ATT_Adapter_DiYDecoder_Conv_pth/{}/{}".format(args.exp, 'MSSAM_w3_7')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    current_time = time.strftime("%H%M%S")
    print("The current time is", current_time)
    snapshot_path = snapshot_path +'_run'+current_time

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    low_res = img_embedding_size * 4  # 32 * 4 = 128  32 * 7 = 224 
    
    net = sam.cuda()
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, 4).cuda()     # 基于此版本进行修改

    
    print('Model %s created, param count: %d' % ('MSSAM: ', sum([m.numel() for m in net.parameters()])))
   
    # macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
