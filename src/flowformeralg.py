import sys
sys.path.append('core')
from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from FlowFormer.configs.submission import get_cfg
from FlowFormer.core.utils.misc import process_cfg
from FlowFormer.core.utils import flow_viz
from FlowFormer.core.utils import frame_utils
import cv2
import math
import os.path as osp
from FlowFormer.core.FlowFormer import build_flowformer
from FlowFormer.core.utils.utils import InputPadder, forward_interpolate
import itertools

TRAIN_SIZE = [432, 960]

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()
    return model


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]


class Flowformeralg:
    def __init__(self):
        self.model = build_model()

    def compute_flow(self, image1, image2, weights=None):
        with torch.no_grad():
            print(f"computing flow...")
    
            image_size = image1.shape[1:]
    
            image1, image2 = image1[None].cuda(), image2[None].cuda()
    
            hws = compute_grid_indices(image_size)
            if weights is None:  # no tile
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
    
                flow_pre, _ = self.model(image1, image2)
    
                flow_pre = padder.unpad(flow_pre)
                flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
            else:  # tile
                flows = 0
                flow_count = 0
    
                for idx, (h, w) in enumerate(hws):
                    image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
                    image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
                    flow_pre, _ = self.model(image1_tile, image2_tile)
                    padding = (w, image_size[1] - w - TRAIN_SIZE[1], h, image_size[0] - h - TRAIN_SIZE[0], 0, 0)
                    flows += F.pad(flow_pre * weights[idx], padding)
                    flow_count += F.pad(weights[idx], padding)
    
                flow_pre = flows / flow_count
                flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

        return flow


if __name__ == '__main__':
    savepth = './FlowFormer/results/seg9'
    if not os.path.exists(savepth):
       os.mkdir(savepth)
    imgpath = './videos'
    video = 'P1050655-seg9.MP4'
    flowformer = Flowformeralg()
    cap = cv2.VideoCapture(os.path.join(imgpath, video))
    ret, img11 = cap.read()
    pre_img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2RGB)
    pre_img11 = torch.from_numpy(pre_img11).permute(2, 0, 1).float()
    i = 0
    while True:
        ret,img12 = cap.read()
        if ret == True:
            im12 = img12.copy()
            current_img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
            current_img12 = torch.from_numpy(current_img12).permute(2, 0, 1).float()

            flo1 = flowformer.compute_flow(pre_img11, current_img12)
            flow_img = flow_viz.flow_to_image(flo1)
            name = str(i) + '.jpg'
            viz_fn = os.path.join(savepth, name)
            cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])
            i += 1
            pre_img11 = current_img12
        else:
           break
