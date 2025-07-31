
from retinaface import RetinaFace
import cv2
import numpy as np
import torch
import os
from datetime import datetime


def get_face_patch_indices(bbox, img_size=112, patch_size=16):

    num_patches = img_size // patch_size
    total_patches = num_patches ** 2

    if len(bbox) == 0:
        roi_indices = []
    else:
        roi_indices = []

        x1, y1, x2, y2 = bbox
        start_col = max(0, int(x1 // patch_size))
        end_col = min(num_patches - 1, int(x2 // patch_size))
        start_row = max(0, int(y1 // patch_size))
        end_row = min(num_patches - 1, int(y2 // patch_size))
        

        patch_indices = []
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):

                idx = row * num_patches + col
                roi_indices.append(idx)
        

        roi_indices = sorted(list(set(roi_indices)))
    
    all_indices = list(range(total_patches))
    non_roi_indices = [idx for idx in all_indices if idx not in roi_indices]
   
    combined = roi_indices + non_roi_indices

    split_point = len(roi_indices) 

        
    return combined, split_point


def process_with_retinaface_and_vit(image_path):

    img = cv2.imread(image_path) 
    if len(img.shape) == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (112, 112)) 

    faces = RetinaFace.detect_faces(img,threshold=0.5)

    if not isinstance(faces, dict):
        bboxes = []
    
    else:
        bboxes = faces['face_1']['facial_area']
        
    patch_indices, split_point = get_face_patch_indices(bboxes)
    return patch_indices, split_point 