import numpy as np
import torch
import os
import cv2
import tool.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
from PIL import Image
import torch.nn.functional as F

def infer_multi_scale(args):
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
    infer_dataset = tool.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root, pseudo_gt=args.pseudo_list,
                                                 scales=[0.5, 1.0, 1.5, 2.0],
                                                 inter_transform=torchvision.transforms.Compose(
                                                    [np.asarray,
                                                    model.normalize,
                                                    imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    crf_alpha = [int(i) for i in args.crf_threshs.split('_')]
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    print("gpu num: " + str(n_gpus), flush=True)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0] 

        img_path = tool.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path).convert('RGB'))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    _, cam = model_replicas[i%n_gpus](img.cuda())
                    cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(args.num_cls - 1, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(args.num_cls - 1):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        def _crf_with_alpha(cam_dict, alpha, num_cls):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()
            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t, args.num_cls)
                folder = args.out_crf 
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)

        if iter % 500 == 0:
            print('iter: ' + str(iter), flush=True)
