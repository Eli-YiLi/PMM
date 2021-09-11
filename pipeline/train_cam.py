import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import tool.data
from tool import pyutils, imutils, torchutils, visualization
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def train(args):
    pyutils.Logger(os.path.join(args.log_dir, args.session_name) + '.log')
    
    if args.network == "ScaleNet50_SEAM":
        from network.scalenet_SEAM import ScaleNet50_SEAM
        model = ScaleNet50_SEAM(args.structure, args.weights, [int(i) for i in args.dilations.split('_')], args.num_cls)
    elif args.network == "ScaleNet101_SEAM":
        from network.scalenet_SEAM import ScaleNet101_SEAM
        model = ScaleNet101_SEAM(args.structure, args.weights, [int(i) for i in args.dilations.split('_')], args.num_cls)
    else:
        model = getattr(importlib.import_module(args.network), 'Net')(args.num_cls)

    print(model)

    tblogger = SummaryWriter(args.log_dir)	

    tool.data.NUM_CLS = args.num_cls
    tool.data.IMG_FOLDER_NAME = args.img_dir
    tool.data.ANNOT_FOLDER_NAME = args.gt_dir
    tool.data.CLS_LABEL = args.cls_label
    if args.heatmap_root == '':
        train_dataset = tool.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root, pseudo_gt=args.pseudo_list,
                                                   transform=transforms.Compose([
                            imutils.RandomResizeLong(448, 768),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            np.asarray,
                            model.normalize,
                            imutils.RandomCrop(args.crop_size),
                            imutils.HWC_to_CHW,
                            torch.from_numpy
                        ]))
    else:
        train_dataset = tool.data.VOC12ClsHeatCropDataset(args.train_list, voc12_root=args.voc12_root,
                            heatmap_root=args.heatmap_root, heat_type='npy', scale=(0.04, 1), ratio=(3. / 5., 5. / 3.),
                            label_match_thresh=0.1, cut_scale=(0.04, args.cut_s), cut_p=args.cut_p,
                            crop_scales=[float(i) for i in args.scales.split(',')] if args.scales != "" else [],
                            crop_size=448, stride=args.stride,
                            transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            transforms.Resize((args.crop_size, args.crop_size)),
                            np.asarray,
                            model.normalize,
                            imutils.HWC_to_CHW,
                            torch.from_numpy
                        ]))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights != "":
        weights_dict = torch.load(args.weights, map_location=torch.device('cpu'))

    if 'ScaleNet' not in args.network:
        if 'model' in weights_dict:
            weights_dict = weights_dict['model']['main']
        model.load_state_dict(weights_dict, strict=False)
        del weights_dict
        torch.cuda.empty_cache()
    else:
        if args.weights != "":
            model.load_state_dict(weights_dict, strict=False)
            del weights_dict
            torch.cuda.empty_cache()

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):
            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1,scale_factor=scale_factor,mode='bilinear',align_corners=True) 
            N,C,H,W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1 = model(img1)
            label1 = F.adaptive_avg_pool2d(cam1, (1,1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
            cam1 = F.interpolate(visualization.max_norm(cam1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label

            cam2, cam_rv2 = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1,1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
            cam2 = visualization.max_norm(cam2)*label
            cam_rv2 = visualization.max_norm(cam_rv2)*label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])

            ns,cs,hs,ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
            cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
            cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)#*eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)#*eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(args.num_cls*hs*ws*0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(args.num_cls*hs*ws*0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2 
            loss = loss_cls + loss_er + loss_ecr
            torch.clamp(loss, 0, 2)

            optimizer.zero_grad()
            if torch.isnan(loss):
                torch.cuda.empty_cache()
                continue
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item()})

            if (optimizer.global_step - 1) % 50 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                loss_dict = {'loss':loss.item(), 
                             'loss_cls':loss_cls.item(),
                             'loss_er':loss_er.item(),
                             'loss_ecr':loss_ecr.item()}
                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)

            else:
                timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.train_res_dir, args.session_name + '.pth'))
