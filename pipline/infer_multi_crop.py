import numpy as np
import torch
import os
import tool.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
from PIL import Image
import torch.nn.functional as F

def infer_multi_crop(args):
    if args.network == "ScaleNet50_SEAM":
        from network.scalenet_SEAM import ScaleNet50_SEAM
        model = ScaleNet50_SEAM(args.structure, ckpt=None, dilations=[int(i) for i in args.dilations.split('_')], num_cls=args.num_cls)
    elif args.network == "ScaleNet101_SEAM":
        from network.scalenet_SEAM import ScaleNet101_SEAM
        model = ScaleNet101_SEAM(args.structure, ckpt=None, dilations=[int(i) for i in args.dilations.split('_')], num_cls=args.num_cls)
    else:
        model = getattr(importlib.import_module(args.network), 'Net')(args.num_cls)
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    tool.data.NUM_CLS = args.num_cls
    tool.data.IMG_FOLDER_NAME = args.img_dir
    tool.data.ANNOT_FOLDER_NAME = args.gt_dir
    tool.data.CLS_LABEL = args.cls_label
    infer_dataset = tool.data.VOC12ClsDatasetMultiCrop(args.infer_list, voc12_root=args.voc12_root, pseudo_gt=args.pseudo_list,
                                                  scales=[float(s) for s in args.scales.split(',')], crop_size=448, stride=args.stride,
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    print(n_gpus, flush=True)
    for iter, (img_name, img_list, label, location) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = tool.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path).convert('RGB'))

        def _work(i, img, loc):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    _, cam = model_replicas[i%n_gpus](img.cuda())
                    cam = F.interpolate(cam[:,1:,:,:], (loc[3], loc[2]), mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(args.num_cls-1, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return [cam, loc]

        thread_pool = pyutils.BatchThreader(_work, [[i, img_list[i], location[i]] for i in range(len(img_list))],
                                            batch_size=len(img_list), prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        # merge crops
        sum_cam = np.zeros((args.num_cls-1, orig_img.shape[0], orig_img.shape[1]))
        sum_counter = np.zeros_like(sum_cam)
        for i in range(len(cam_list)):
            x, y, w, h = cam_list[i][1]
            crop = cam_list[i][0]
            sum_cam[:, y:y+h, x:x+w] += crop
            sum_counter[:, y:y+h, x:x+w] += 1
        sum_counter[sum_counter < 1] = 1

        sum_cam = sum_cam / sum_counter
        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(args.num_cls-1):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i].astype(np.float16)

        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if iter % 500 == 0:
            print(iter, flush=True)
