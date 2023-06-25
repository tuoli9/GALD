import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from torchvision import transforms

from tqdm import tqdm
import numpy as np
import argparse
from utils import Normalize, clip_by_tensor, drop_patch
from dataset import AdvDataset
from vit_models import swin_transformer, re_vit, re_deit, re_cait, tnt, re_pit

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/', help='Input directory with images.')

    parser.add_argument('--white', type=str, default='vit', help='target label.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')

    parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
    
    parser.add_argument("--N", type=int, default=5, help="")
    parser.add_argument("--ra", type=float, default=0.7, help="attention ratio")
    parser.add_argument("--rd", type=float, default=0.1, help="patch drop ratio")
    
    return parser


def GALD(images, gt, model, model_name, min, max, opt):
    
    num_iter = opt.num_iter
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    device = x.device
    # model = model.to(device)
    delta = torch.zeros_like(x, requires_grad=True).to(device)
    bs,c,h,w = images.shape

    if model_name in ['vit_deit_base_distilled_patch16_224','levit_256','pit_b_224',
                                    'cait_s24_224','convit_base', 'visformer_small', 
                                    'deit_base_distilled_patch16_224','T2t_vit_t_14', 
                                    'swin_base_patch4_window7_224', 'deit_base_patch16_224']:
        norm = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    else:
        norm = Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    
    for i in range(num_iter):
        
        grad_c = 0.0
        for n in range(opt.N):
            drop_img = drop_patch(x + delta,shuffle_size=14,ratio=opt.rd)
            if i == 0:
                outputs, attns, layer_wise = model(norm(drop_img))
            else:
                outputs, attns, layer_wise = model(norm(drop_img),opt.ra,attns)
            # outputs, attns, embeds = model(norm(x + delta))
    
            loss = F.cross_entropy(outputs, gt)
            loss.backward()
            grad_c += delta.grad.clone().to(device)
            delta.grad.zero_()

        delta.data = delta.data + alpha * torch.sign(grad_c)
        delta.data = delta.data.clamp(-eps, eps)
        delta.data = clip_by_tensor(x.data + delta.data, min, max) - x.data
        # delta.data = ((x + delta.data).clamp(0, 1)) - x
    adv_img = x + delta
    return adv_img.detach()


def main():
    
    opt = argument_parsing().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    data = AdvDataset(adv_path = opt.input_dir)
    data_loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    norm0 = Normalize(mean = IMAGENET_DEFAULT_MEAN, std = IMAGENET_DEFAULT_STD)
    norm1 = Normalize(mean = IMAGENET_INCEPTION_MEAN, std = IMAGENET_INCEPTION_STD)

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda')
    
    vit_b = re_vit.vit_base_patch16_224(pretrained=True).eval().to(device)

    for param in vit_b.parameters():
        param.requires_grad = False
    vit = nn.Sequential(norm1, vit_b)
    deit_b = re_deit.deit_base_patch16_224(pretrained=True).eval().to(device)
    deit = nn.Sequential(norm0, deit_b)

    cait_s = re_cait.cait_s24_224(pretrained=True).eval().to(device)
    cait = nn.Sequential(norm0, cait_s)

    tnt_s = tnt.tnt_s_patch16_224(pretrained=True).eval().to(device)
    TNT = nn.Sequential(norm1, tnt_s)

    swin_b = swin_transformer.swin_base_patch4_window7_224(pretrained=True).eval().to(device)
    for param in swin_b.parameters():
        param.requires_grad = False
    swin = nn.Sequential(norm0, swin_b)  

    pit_b = re_pit.pit_b_224(pretrained=True).eval().to(device)
    for param in pit_b.parameters():
        param.requires_grad = False
    pit = nn.Sequential(norm0, pit_b) 
    
    model_lst = [vit, deit, cait, TNT, swin, pit]
    model_names = ['vit', 'deit', 'cait', 'tnt', 'swin', 'pit']
        
   
    asr = torch.zeros(len(model_lst))

        
    if opt.white == 'vit':
        white = vit_b
        
        white_name = 'vit_base_patch16_224'
        
    elif opt.white == 'deit':
        white = deit_b
        
        white_name = 'deit_base_patch16_224'
        
    elif opt.white == 'tnt':
        white = tnt_s
        
        white_name = 'tnt_s_patch16_224'
        
    elif opt.white == 'pit':
        white = pit_b
        
        white_name = 'pit_b_224'
        
    for imgs, labels, _, _ in tqdm(data_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        images_min = clip_by_tensor(imgs - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(imgs + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = GALD(imgs, labels, white, white_name, images_min, images_max, opt)
        
        with torch.no_grad():
            for idx, model in enumerate(model_lst):
                outs = model(adv_img)
                
                if isinstance(outs, tuple) or isinstance(outs, list):
                    outs = outs[0]
                    if isinstance(outs, list):
                        outs = outs[-1]
                asr[idx] += (outs.argmax(1) != labels).detach().sum().cpu()
               

    for i, model in enumerate(model_lst):
        print('{} = {:.2%}'.format(model_names[i], asr[i] / 1000.0))
    print()


if __name__ == '__main__':
    main()