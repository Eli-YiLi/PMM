import argparse
import os

from tool import pyutils

from pipline.train_cam import train
from pipline.infer_multi_scale import infer_multi_scale
from pipline.infer_multi_crop import infer_multi_crop
from pipline.refine_cam import refine_cam
from pipline.evaluation import evaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--local_rank", type=int, default=0)

    # Dataset
    parser.add_argument("--dataset", default="voc12", type=str)
    parser.add_argument("--voc12_root", default='voc12/VOC2012', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--img_dir", default="JPEGImages", type=str)
    parser.add_argument("--gt_dir", default="voc12/VOC2012/SegmentationClass", type=str)
    parser.add_argument("--cls_label", default="voc12/cls_labels.npy", type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)

    # Train
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--num_cls", default=21, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--pseudo_list", default="", type=str) # not used
    parser.add_argument("--heatmap_root", default="", type=str)
    parser.add_argument("--dilations", default="1_1_1_1", type=str)
    parser.add_argument("--structure", default="models/scalenet/structures/scalenet101.json", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default='models/resnet38_SEAM.pth', type=str)
    parser.add_argument("--cut_p", default=0, type=float) # not used
    parser.add_argument("--cut_s", default=0.25, type=float)
    parser.add_argument("--scales", default="0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3", type=str)
    parser.add_argument("--stride", default=300, type=int)
    parser.add_argument("--session_name", default="", type=str)

    # Inference
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str) 

    # CRFs
    parser.add_argument("--crf_threshs", default='24', type=str)
    parser.add_argument("--cam_thresh", default=0.05, type=float)
    parser.add_argument("--cv_scale", default=0.3, type=float)

    # Evaluation
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument("--eval_thresh", default=0.1, type=float)
    parser.add_argument("--predict_dir", default="", type=str)

    # Output Path
    parser.add_argument("--train_res_dir", default="data/results/train", type=str)
    parser.add_argument("--test_res_dir", default="data/results/test", type=str)
    parser.add_argument("--log_dir", default="data/results/log", type=str)
    parser.add_argument("--eval_log_name", default="", type=str)

    # Pipline
    parser.add_argument("--train_multi_scale", default=False, action='store_true')
    parser.add_argument("--gen_mask_for_multi_crop", default=False, action='store_true')
    parser.add_argument("--train_multi_crop", default=False, action='store_true')
    parser.add_argument("--eval", default=False, action='store_true')
    parser.add_argument("--gen_seg_mask", default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs("data/results", exist_ok=True)
    os.makedirs(args.train_res_dir, exist_ok=True)
    os.makedirs(args.test_res_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.dataset == 'coco14':
        args.session_name = 'coco'
        args.voc12_root = 'coco14'
        args.train_list = 'coco14/voc_format/train.txt'
        args.val_list = 'coco14/voc_format/val.txt'
        args.img_dir = 'images'
        args.gt_dir = 'coco14/voc_format/class_labels'
        args.cls_label = 'coco14/voc_format/cls_labels.npy'
        args.num_cls = 91 # 81 classes with data
        args.batch_size = 8 # 16 out of memory
        args.lr = 0.01
        args.train_multi_scale = False
        args.gen_mask_for_multi_crop = False

    print(vars(args))

    # two purposes: 1.comparison with baseline 2.avoid noise samples in multi-crop training
    if args.train_multi_scale:
        timer = pyutils.Timer('train in multi-scale strategy:')

        args.session_name = args.dataset + '_wr38'
        train(args)

    if args.gen_mask_for_multi_crop:
        timer = pyutils.Timer('infer multi_scale cam and make rough mask:')

        args.session_name = args.dataset + '_wr38'
        args.infer_list = args.train_list
        args.weights = os.path.join(args.train_res_dir, args.session_name + '.pth')
        args.out_cam = os.path.join(args.test_res_dir, args.session_name, 'cam')
        args.out_crf = os.path.join(args.test_res_dir, args.session_name, 'train_mask')
        infer_multi_scale(args)

    if args.train_multi_crop:
        timer = pyutils.Timer('train in multi-crop strategy:')

        args.batch_size = 8 if args.dataset == 'coco14' else 16
        args.max_epoches = 20
        args.network = 'ScaleNet101_SEAM'
        args.weights = 'models/scalenet/weights/scalenet101.pth'
        args.session_name = args.dataset + '_s101_mc'
        args.heatmap_root = os.path.join(args.test_res_dir, args.dataset + '_wr38', 'train_mask')
        # simplify and speed up training of COCO via ignore heatmap
        if args.dataset == 'coco14':
            args.heatmap_root = ''
        train(args)

    if args.eval:
        timer = pyutils.Timer('infer, eval cam and PPMG:')

        args.network = 'ScaleNet101_SEAM'
        args.session_name = args.dataset + '_s101_mc'
        args.infer_list = args.val_list # change val_list to eval test and train
        set_name = args.infer_list.split('/')[-1].split('.')[0]
        args.weights = os.path.join(args.train_res_dir, args.session_name + '.pth')
        args.out_cam = os.path.join(args.test_res_dir, args.session_name, 'cam_' + set_name)
        args.out_crf = None
        infer_multi_crop(args)

        args.eval_log_name = os.path.join(args.log_dir, args.session_name + '_cam_' + set_name)
        args.predict_dir = args.out_cam
        args.type = 'npy'
        args.curve = True
        evaluation(args)

        args.type = 'png'
        args.crf_threshs = '11'
        args.out_crf = os.path.join(args.test_res_dir, args.session_name, 'ppmg_' + set_name)
        refine_cam(args)
        
        args.eval_log_name = os.path.join(args.log_dir, args.session_name + '_ppmg_' + set_name)
        args.predict_dir = args.out_crf
        args.curve = False
        evaluation(args)

    if args.gen_seg_mask:
        timer = pyutils.Timer('infer multi_crop cam and apply ppmg for segmentation:')

        args.network = 'ScaleNet101_SEAM'
        args.session_name = args.dataset + '_s101_mc'
        args.infer_list = args.train_list
        args.weights = os.path.join(args.train_res_dir, args.session_name + '.pth')
        args.out_cam = os.path.join(args.test_res_dir, args.session_name, 'cam')
        args.out_crf = None
        infer_multi_crop(args)

        args.out_crf = os.path.join(args.test_res_dir, args.session_name, 'ppmg')
        args.crf_threshs = '11'
        refine_cam(args)
