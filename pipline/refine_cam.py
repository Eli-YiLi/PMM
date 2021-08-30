import cv2, sys, os, random
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import itertools
from multiprocessing import Pool

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(3/scale_factor, compat=10)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def crf_with_alpha(img, cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(img, bgcam_score, labels=bgcam_score.shape[0])
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]

    # return npy
    return n_crf_al

def ppmg(img, cam_dict, alpha, cam_thresh=0, k_add=True, cv={}):

    # init bg
    foreground = [np.ones((img.shape[0], img.shape[1])).astype(np.uint8)]
    importance = [foreground[0]]
    keys = [0]

    for k, v in cam_dict.items():
        # cvs
        v = np.array([v])
        cvs = np.copy(v)
        cvs[cvs < cam_thresh] = 0
        cvs = cvs ** (1 - cv[k])
        bg_score = np.power(1 - np.max(cvs, axis=0, keepdims=True), alpha)

        # crf
        bgcam_score = np.concatenate((bg_score, cvs), axis=0)
        crf_k = crf_inference(img, bgcam_score, labels=bgcam_score.shape[0])[1]

        fg_k = np.zeros_like(crf_k)
        fg_k[crf_k > 0.5] = 1
        fg_k[crf_k <= 0.5] = 0
        foreground.append(fg_k)
        importance.append(v[0] / (1 + (v[0] * fg_k).sum())) # +1 aviod 0
        keys.append(k + 1 if k_add else k)

        # update bg
        foreground[0] = foreground[0] * (fg_k == 0)
        importance[0] = foreground[0]

    # mask to numpy
    out = {}
    mask = np.argmax(np.stack(foreground, 0) * np.stack(importance, 0), 0)
    for i, k in enumerate(keys):
        out[k] = (mask == i).astype('bool')

    return out

def get_CoV(heat, scale=0.3, t=0.05):
    CoV = {}
    for k, v in heat.items():
        if (v > t).sum() == 0:
            CoV[k] = 0
        else:
            CoV[k] = v[v > t].std() / (v[v > t] + 0.00000001).mean() * scale
    return CoV

def work(pid, ps, imgs, imgd, maskd, outd, crf_thresh, cam_thresh, cv_scale, png=False):
    for idx, i in enumerate(imgs):
        if idx % ps != pid:
            continue
        if not os.path.exists(os.path.join(maskd, i.split('.')[0] + '.npy')):
            continue

        img = cv2.imread(os.path.join(imgd, i))
        mask = np.load(os.path.join(maskd, i.split('.')[0] + '.npy'), allow_pickle=True)
        cv = get_CoV(mask.item(), cv_scale, cam_thresh)
        mask = ppmg(img, mask.item(), crf_thresh, cam_thresh, True, cv)

        if png:
            out = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
            for k, v in mask.items():
                out[v != 0] = k
            cv2.imwrite(os.path.join(outd, i.replace('.jpg', '.png')), out)
        else:
            with open(os.path.join(outd, i.replace('.jpg', '.npy')), 'wb') as f:
                np.save(f, mask)

def throw_error(e):
    raise e

def refine_cam(args):
    print(os.path.join(args.voc12_root, args.img_dir), args.out_cam, args.out_crf)
    imgs = os.listdir(os.path.join(args.voc12_root, args.img_dir))
    if not os.path.exists(args.out_crf):
        os.makedirs(args.out_crf)

    pnum = args.num_workers
    pool = Pool(pnum)
    for i in range(pnum):
        pool.apply_async(work, \
          (i, pnum, imgs, os.path.join(args.voc12_root, args.img_dir), args.out_cam, \
            args.out_crf, int(args.crf_threshs), args.cam_thresh, args.cv_scale, args.type=='png',), \
          error_callback=throw_error)
    pool.close()
    pool.join()
