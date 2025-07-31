import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from .data import cfg_mnet, cfg_re50, cfg_mnetv3, cfg_gnet
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from .utils.box_utils import decode, decode_landm
from .models.retinaface_g import RetinaFace
from models.retinaface_m import RetinaFace as RetinaFace_m
from datetime import datetime
import os

def get_face_patch_indices(bbox, img_size=512, patch_size=16):
   
    num_patches = img_size // patch_size
    total_patches = num_patches ** 2

    x1, y1, x2, y2, _ = bbox
    start_col = max(0, int(x1 // patch_size))
    end_col = min(num_patches - 1, int(x2 // patch_size))
    start_row = max(0, int(y1 // patch_size))
    end_row = min(num_patches - 1, int(y2 // patch_size))
    

    patch_indices = []
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):

            idx = row * num_patches + col
            patch_indices.append(idx)

    roi_indices = sorted(list(set(patch_indices)))
    

    all_indices = list(range(total_patches))
    non_roi_indices = [idx for idx in all_indices if idx not in roi_indices]
    
    combined = roi_indices + non_roi_indices
    split_point = len(roi_indices)  
    
    return combined, split_point


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect_faces_tensor(input_tensor, net, device, cfg):
    """
    Detects faces in a batch of input tensors using RetinaFace model.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [B, 3, 112, 112] with pixel range [0, 255]
        net (torch.nn.Module): RetinaFace model for face detection
        device (torch.device): Device to perform computation on
        cfg (dict): Configuration dictionary containing model parameters
    
    Returns:
        List[List[List[int]]]: For each image in batch, returns a list of detected faces where each face is
        represented as [x1, y1, x2, y2, score]. Coordinates are in pixel values and score is confidence.
    
    Note:
        - Input tensor is assumed to be in BGR format (OpenCV default)
        - Performs mean subtraction with values [104, 117, 123] for BGR channels
        - Applies NMS and confidence thresholding (0.5) before returning results
    """
    """
    input_tensor: Tensor of shape [B, 3, 112, 112], pixel range [0, 255] (assumed)
    Returns: List of face bboxes [x1, y1, x2, y2, score] for each image in batch
    """
    input_tensor = input_tensor * 225
    batch_size, _, input_h, input_w = input_tensor.shape
    input_tensor = input_tensor.to(device).float()
    # print("input_tensor shape: {}".format(input_tensor.shape))
    mean = torch.tensor([123.0, 117.0, 104.0], device=device).view(1, 3, 1, 1)#[104.0, 117.0, 123.0] 
    input_tensor -= mean  # broadcasting to all pixels
    results = []
    combined_all = []
    split_point_all = []
    for i in range(batch_size):
        img = input_tensor[i].unsqueeze(0)  # shape [1, 3, H, W]
        im_height, im_width = input_h, input_w
        if im_height==112:
            patch_number=49
        else:
            patch_number=1024
        scale = torch.tensor([im_width, im_height, im_width, im_height], device=device)
        loc, conf, _ = net(img)
        
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']) * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]


        # 筛选
        inds = np.where(scores > 0.5)[0]#args.confidence_threshold
        # print(f"inds:{inds}")
        if len(inds) == 0:
            dets=[0,0,0,0,0]
            combined = list(range(patch_number))
            split_point = 0

        else:
            boxes = boxes[inds]
            scores = scores[inds]

            # top-k
            order = scores.argsort()[::-1][:1]#args.top_k
            boxes = boxes[order]
            scores = scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
            keep = py_cpu_nms(dets, 0.5 )#args.nms_threshold
            dets = dets[keep]

            dets = dets[:750]
            combined, split_point = get_face_patch_indices(dets[0],img_size=im_height)
        results.append(dets[0])
        combined_all.append(combined)
        split_point_all.append(split_point)


    return results, combined_all, split_point_all 




def Retina_detect(input_tensor):
    cfg = cfg_gnet
    pretrained_model="./model/retina_face/weights/ghostnet_Final.pth"
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, pretrained_model, False)
    net.eval()
    device = torch.device("cuda")
    net = net.to(device)
    results, combined_all, split_point_all = detect_faces_tensor(input_tensor, net, device, cfg)
    return results, combined_all, split_point_all

