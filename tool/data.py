import math
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
import sys
sys.path.append('../')
from tool import imutils
from torchvision import transforms
from PIL import Image

NUM_CLS=21
IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
CLS_LABEL = 'voc12/cls_labels.npy'
CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load(CLS_LABEL, allow_pickle=True).item()

    return [cls_labels_dict[img_name].astype(np.float32) for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] if 'jpg' in img_gt_name else img_gt_name.strip() for img_gt_name in img_gt_name_list]

    #img_name_list = img_gt_name_list
    return img_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = []
        if img_name_list_path != '':
            self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        if 'yfcc100m' in name:
            img = PIL.Image.open(name).convert("RGB")
        else:
            img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None, pseudo_gt=""):
        super().__init__(img_name_list_path, voc12_root, transform)
        if self.img_name_list != '':
            self.label_list = load_image_label_list_from_npy(self.img_name_list)
        if pseudo_gt != "":
            lines = open(pseudo_gt).readlines()
            for l in lines:
                tokens = l.strip().split()
                #labels = np.zeros_like(self.label_list[0])
                labels = np.zeros(500).astype(np.float32)
                for t in tokens[1:]:
                    if int(t) < labels.shape[0]:
                        labels[int(t) - 1] = 1
                self.label_list.append(labels)
                self.img_name_list.append(tokens[0])

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1, pseudo_gt=''):
        super().__init__(img_name_list_path, voc12_root, transform=None, pseudo_gt=pseudo_gt)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ClsDatasetMS(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list, label

class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        label_la = np.load(label_la_path, allow_pickle=True).item()
        label_ha = np.load(label_ha_path, allow_pickle=True).item()

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0))

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label

class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label

class VOC12ClsHeatCropDataset(Dataset):

    def __init__(self, img_name_list_path, heatmap_root, heat_type, voc12_root, transform=None, \
                       scale=(0.04, 1), ratio=(3. / 5., 5. / 3.), label_match_thresh=0.1, \
                       cut_scale=(0.02, 0.25), cut_p=0.5,
                       crop_scales=[], crop_size=448, stride=300):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

        self.voc12_root = voc12_root
        self.heatmap_root = heatmap_root
        self.heat_type = heat_type
        self.transform = transform
        self.scale = scale
        self.ratio = ratio
        self.label_match_thresh = label_match_thresh
        self.cut_scale = cut_scale
        self.cut_p = cut_p
        self.crop_scales = crop_scales
        self.crop_size = crop_size
        self.stride = stride

    def get_params(self, img, scale, ratio):
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                param = [int(j), int(i), int(j + w), int(i + h)]
                return param

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

        param = [int(j), int(i), int(j + w), int(i + h)]
        return param

    def get_multi_crop_params(self, img, crop_scales, crop_size, stride):
        params_list = []

        # multi crop proposals
        width, height = img.size
        for s in crop_scales:
            s_list = []
            w, h = int(width * s), int(height * s)
            w_num = 1 + int(math.ceil(max(0.0, float(w - crop_size)) / stride))
            h_num = 1 + int(math.ceil(max(0.0, float(h - crop_size)) / stride))
            for w_idx in range(w_num):
                for h_idx in range(h_num):
                    if w_idx == (w_num - 1):
                        x2 = w
                        x1 = max(0, x2 - crop_size)
                    else:
                        x2 = min(w, (w_idx + 1) * stride)
                        x1 = w_idx * stride

                    if h_idx == (h_num - 1):
                        y2 = h
                        y1 = max(0, y2 - crop_size)
                    else:
                        y2 = min(h, (h_idx + 1) * stride)
                        y1 = h_idx * stride

                    x1 = int(float(x1) / s)
                    x2 = int(float(x2) / s)
                    y1 = int(float(y1) / s)
                    y2 = int(float(y2) / s)
                    s_list.append([x1, y1, x2, y2])
            params_list.append(s_list)

        return params_list
        

    def update_label(self, label, heat, param, label_thresh=0.1):
        new_label = np.zeros_like(label)
        for k, v in heat.items():
            k_first = k
            break
        selected = np.zeros_like(heat[k_first])
        selected[int(param[1]):int(param[3]), int(param[0]):int(param[2])] = 1
        for i in range(label.shape[0]):
            if int(i + 1) not in heat:
                continue
            mask = heat[int(i) + 1]
            if (mask > 0.9).sum() == 0:
                continue
            intersection = float(((mask > 0.9) * selected).sum())
            # box inside mask, or mask inside box
            if (intersection / (mask > 0.9).sum()) > label_thresh or (intersection / selected.sum()) > label_thresh:
               new_label[i] = 1
        return  new_label

    def check_cut_label(self, label, heat, param, cut_param, label_thresh=0.1):
        new_label = np.zeros_like(label)
        for k, v in heat.items():
            k_first = k
            break
        selected = np.zeros_like(heat[k_first])
        selected[int(param[1]):int(param[3]), int(param[0]):int(param[2])] = 1

        # cut part in selected
        w, h = param[2] - param[0], param[3] - param[1]
        x1 = param[0] + float(w) / selected.shape[1] * cut_param[0]
        y1 = param[1] + float(h) / selected.shape[0] * cut_param[1]
        x2 = param[0] + float(w) / selected.shape[1] * cut_param[2]
        y2 = param[1] + float(h) / selected.shape[0] * cut_param[3]
        selected[int(y1):int(y2), int(x1):int(x2)] = 0

        for i in range(label.shape[0]):
            if int(i + 1) not in heat:
                continue
            mask = heat[int(i) + 1]
            if (mask > 0.9).sum() == 0:
                continue
            intersection = float(((mask > 0.9) * selected).sum())
            # box inside mask, or mask inside box
            if (intersection / (mask > 0.9).sum()) > label_thresh or (intersection / selected.sum()) > label_thresh:
               new_label[i] = 1
        return  new_label, [int(x1), int(y1), int(x2), int(y2)]

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        if self.heat_type == 'png':
            try:
                png = Image.open(os.path.join(self.heatmap_root, name + '.png'))
                png_np = np.array(png)
            except:
                return self.__getitem__(idx + 1)
            uniq = set(png_np.reshape(png_np.shape[0] * png_np.shape[1]).tolist())
            heat = {}
            for l in uniq:
                m = np.zeros_like(png_np)
                m[png_np == l] = 1
                heat[int(l)] = m
        elif self.heat_type == 'npy':
            heat = np.load(os.path.join(self.heatmap_root, name + '.npy'), allow_pickle=True).item()
        else:
            print('error heatmap type!')

        if self.crop_scales == []:
            counter = 0
            while True:
                counter += 1
                param = self.get_params(img, self.scale, self.ratio)
                new_label = self.update_label(label, heat, param, self.label_match_thresh)
                if new_label.sum() > 0 or counter >= 100:
                    break
        else:
            params = self.get_multi_crop_params(img, self.crop_scales, self.crop_size, self.stride)
            scale_idx = int(torch.rand(1) * len(self.crop_scales))
            ins_len = len(params[scale_idx])
            for _ in range(ins_len):
                ins_idx = int(torch.rand(1) * ins_len)
                param = params[scale_idx][ins_idx]
                new_label = self.update_label(label, heat, param, self.label_match_thresh)
                if new_label.sum() > 0:
                    break

        if torch.rand(1) < self.cut_p:
            cut_param = self.get_params(img, self.scale, self.ratio)
            new_label, fill_param = self.check_cut_label(label, heat, param, cut_param, self.label_match_thresh)
            fill_img = Image.new('RGB', [fill_param[2] - fill_param[0], fill_param[3] - fill_param[1]], (255,255,255))
            img.paste(fill_img, fill_param)

        img = img.crop(param)
        if self.transform:
            img = self.transform(img)

        return name, img, torch.from_numpy(new_label)


class VOC12ClsDatasetMultiCrop(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales=[0.75, 1, 1.25, 1.5, 2], crop_size=448, stride=300, inter_transform=None, unit=1, pseudo_gt=''):
        super().__init__(img_name_list_path, voc12_root, transform=None, pseudo_gt=pseudo_gt)
        self.scales = scales
        self.crop_size = crop_size
        self.stride=stride
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)
        width, height = img.size

        ms_img_list = []
        location_list = []
        for s in self.scales:
            w, h = int(width * s), int(height * s)
            w_num = 1 + int(math.ceil(max(0.0, float(w - self.crop_size)) / self.stride))
            h_num = 1 + int(math.ceil(max(0.0, float(h - self.crop_size)) / self.stride))
            for w_idx in range(w_num):
                for h_idx in range(h_num):
                    if w_idx == (w_num - 1):
                        x2 = w
                        x1 = max(0, x2 - self.crop_size)
                    else:
                        x2 = min(w, (w_idx + 1) * self.stride)
                        x1 = w_idx * self.stride

                    if h_idx == (h_num - 1):
                        y2 = h
                        y1 = max(0, y2 - self.crop_size)
                    else:
                        y2 = min(h, (h_idx + 1) * self.stride)
                        y1 = h_idx * self.stride

                    x1 = int(float(x1) / s)
                    x2 = int(float(x2) / s)
                    y1 = int(float(y1) / s)
                    y2 = int(float(y2) / s)

                    s_img = img.crop((x1, y1, x2, y2))
                    s_img = s_img.resize((self.crop_size, self.crop_size), resample=PIL.Image.BILINEAR)
                    ms_img_list.append(s_img)
                    location_list.append([x1, y1, x2 - x1, y2 - y1])

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        location_list_f = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            location_list_f.append(location_list[i])
            location_list_f.append(location_list[i])

        return name, msf_img_list, label, location_list_f

