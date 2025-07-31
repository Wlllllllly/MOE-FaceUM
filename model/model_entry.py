import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.geometry import denormalize_points
# import cv2
import numpy as np



class MOEFaceUM_TEST(nn.Module):
    def __init__(self, backbone_module, heads_module):
        super(MOEFaceUM_TEST, self).__init__()

        self.backbone_module = backbone_module
        self.heads_module = heads_module

    def forward(self, image,**kwargs):
        if self.mode == "evaluate":
            task_j=kwargs["task_j"]
            task_name=kwargs["task_name"]
            outputs = dict()
            outputs[task_name] = dict()
            outputs[task_name]["task_name"] = self.evaluation_task
            x = self.backbone_module(image,task_j=task_j)
            outputs[task_name]["backbone_output"] = x
            outputs = self.heads_module(outputs)

            return outputs[task_name]
