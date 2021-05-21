# -*- coding: utf-8 -*-
# General utils

import glob
import logging
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torchvision
import yaml

from utils.google_utils import gsutil_getsize
from utils.metrics import fitness
from utils.torch_utils import init_torch_seeds
from utils.rotated.oriented_iou_loss import cal_iou

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 53))  # check host accesability
        return True
    except OSError:
        return False


def check_git_status():
    # Suggest 'git pull' if YOLOv5 is out of date
    print(colorstr('github: '), end='')
    try:
        if Path('.git').exists() and check_online():
            url = subprocess.check_output(
                'git fetch && git config --get remote.origin.url', shell=True).decode('utf-8')[:-1]
            n = int(subprocess.check_output(
                'git rev-list $(git rev-parse --abbrev-ref HEAD)..origin/master --count', shell=True))  # commits behind
            if n > 0:
                print(f"⚠️ WARNING: code is out of date by {n} {'commits' if n > 1 else 'commmit'}. "
                      f"Use 'git pull' to update or 'git clone {url}' to download latest.")
            else:
                print(f'up to date with {url} ✅')
    except Exception as e:
        print(e)


def check_requirements(file='requirements.txt'):
    # Check installed dependencies meet requirements
    import pkg_resources
    requirements = pkg_resources.parse_requirements(Path(file).open())
    requirements = [x.name + ''.join(*x.specs) if len(x.specs) else x.name for x in requirements]
    pkg_resources.require(requirements)  # DistributionNotFound or VersionConflict exception if requirements not met


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def check_dataset(dict):
    # Download dataset if not found locally
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h, a] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x[...,:4].clone() if isinstance(x, torch.Tensor) else np.copy(x[...,:4])
    angle = x[:, 4]
    cos_a = torch.cos(angle) if isinstance(x, torch.Tensor) else np.cos(angle)
    sin_a = torch.sin(angle) if isinstance(x, torch.Tensor) else np.sin(angle)
    # y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    x_c = x[:, 0]
    y_c = x[:, 1]
    x_min = x[:, 0] - x[:, 2] / 2
    y_min = x[:, 1] - x[:, 3] / 2
    x_max = x[:, 0] + x[:, 2] / 2
    y_max = x[:, 1] + x[:, 3] / 2    
    # cos(angle) * (p.x - cx) - sin(angle) * (p.y - cy) + cx
    # sin(angle) * (p.x - cx) + cos(angle) * (p.y - cy) + cy
    x0 = cos_a * (x_min - x_c) - sin_a * (y_min - y_c) + x_c
    y0 = sin_a * (x_min - x_c) + cos_a * (y_min - y_c) + y_c
    x1 = cos_a * (x_min - x_c) - sin_a * (y_max - y_c) + x_c
    y1 = sin_a * (x_min - x_c) + cos_a * (y_max - y_c) + y_c
    x2 = cos_a * (x_max - x_c) - sin_a * (y_max - y_c) + x_c
    y2 = sin_a * (x_max - x_c) + cos_a * (y_max - y_c) + y_c
    x3 = cos_a * (x_max - x_c) - sin_a * (y_min - y_c) + x_c
    y3 = sin_a * (x_max - x_c) + cos_a * (y_min - y_c) + y_c
    y[:, 0] = torch.minimum(x0, x1) if isinstance(x, torch.Tensor) else np.minimum(x0, x1)
    y[:, 1] = torch.minimum(y0, y1) if isinstance(x, torch.Tensor) else np.minimum(y0, y1)
    y[:, 2] = torch.maximum(x2, x3) if isinstance(x, torch.Tensor) else np.maximum(x2, x3)
    y[:, 3] = torch.maximum(y2, y3) if isinstance(x, torch.Tensor) else np.maximum(y2, y3)
    return y


def xywha2xyxya(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxya2xywha(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def get_rotated_coors(box):
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-int(a*180/math.pi), center=(cx,cy), scale=1)
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 

    if isinstance(x0,torch.Tensor):
        r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
                         x1.unsqueeze(0),y1.unsqueeze(0),
                         x2.unsqueeze(0),y2.unsqueeze(0),
                         x3.unsqueeze(0),y3.unsqueeze(0)], 0)
    else:
        r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
    return r_box


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0]] -= pad[0]  # x padding
    coords[:, [1]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def nms_rotate(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, batches: int):
    """
        boxes: Tensor[N, 5] - x, y, w, h, theta
        scores: Tensor[N]
        iou_threshold: 0.0-1.0
    """
    # time_iou = time.time()
    # print(shape)
    with torch.no_grad():
        shape = boxes.shape[0]
        iou = torch.empty(shape, shape, device=boxes.device)
        if shape > 500:
            for i in range(batches):
                for j in range(batches):
                    sh_i_from = (shape//batches + 1)*i
                    sh_j_from = (shape//batches + 1)*j
                    sh_i_to = (shape//batches + 1)*(i + 1)
                    sh_j_to = (shape//batches + 1)*(j + 1)
                    if i == batches - 1:
                        sh_i_to = shape
                    if j == batches - 1:
                        sh_j_to = shape
                        #придумать что сделать с последним батчем
                    box1 = boxes[sh_i_from:sh_i_to].view([sh_i_to-sh_i_from, 1, 5]).expand([-1, sh_j_to-sh_j_from, -1])
                    box2 = boxes[sh_j_from:sh_j_to].view([1, sh_j_to-sh_j_from, 5]).expand([sh_i_to-sh_i_from, -1, -1])
                    iou[sh_i_from:sh_i_to, sh_j_from:sh_j_to] = cal_iou(box1, box2)[0]
        else:
            box1 = boxes.view([shape, 1, 5]).expand([-1, shape, -1])
            box2 = boxes.view([1, shape, 5]).expand([shape, -1, -1])
            iou = cal_iou(box1, box2)[0]
        mask_iou = iou > iou_threshold
        mask_scores = scores.unsqueeze(1) > scores
        # ind = np.diag_indices(mask_scores.shape[0])
        # mask_scores[ind[0], ind[1]] = torch.zeros(mask_scores.shape[0], dtype=bool, device=boxes.device)
        indices = torch.nonzero(~(mask_iou & mask_scores).sum(axis=0, dtype=bool)).squeeze()

    # iou = cal_iou(boxes.view([shape, 1, 5]).expand([-1, shape, -1]), 
    #               boxes.view([1, shape, 5]).expand([shape, -1, -1]))[0]

    # print('time for iou: ', (time.time() - time_iou), ' seconds')

    # time_nms = time.time()

    # indices = []
    # scores_cpu = scores.cpu()
    # iou_cpu = iou.cpu()
    # for i in range(boxes.shape[0]):
    #     is_add = True
    #     for j in range(boxes.shape[0]):
    #         if iou[i, j] > iou_threshold and scores[i] < scores[j]:
    #             is_add = False
    #             break
    #     if is_add:
    #         indices.append(i)
    # del mask_iou
    # del mask_scores
    # del iou
    # del scores_cpu
    # del iou_cpu
    # print('time for nms: ', (time.time() - time_nms), ' seconds')

    return indices #torch.tensor(indices, device=boxes.device)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 6  # number of classes
    xc = prediction[..., 5] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    agnostic=True

    t = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 6), device=x.device)
            v[:, :5] = l[:, 1:6]  # box
            v[:, 5] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 6] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 6:] *= x[:, 5:6]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # box = xywh2xyxy(x[:, :4])
        box = x[:, :5]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 6:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 6, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 6:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 5].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :5], x[:, 5]  # boxes (offset by class), scores
        boxes[:, :2] += c

        i = torchvision.ops.nms(xywh2xyxy(boxes), scores, iou_thres)
        
        # i = nms_rotate(boxes, scores, iou_thres, batches=2)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = cal_iou(boxes[i].view([boxes[i].shape[0], 1, 5]).repeat([1, boxes[i].shape[0], 1]), 
        #                   boxes.view([1, boxes.shape[0], 5]).repeat([boxes.shape[0], 1, 1]))[0] > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :5] = torch.mm(weights, x[:, :5]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    for key in 'optimizer', 'training_results', 'wandb_id':
        x[key] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
