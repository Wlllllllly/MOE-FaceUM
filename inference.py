"""
Face Recognition - Task 0
Face Age Estimation - Task 1
Face Attributes Recognition - Task 2
Face Expression Recognition - Task 3
Face Parsing - Task 4
Face Landmarks - Task 5
"""

import os
import numpy as np
import cv2
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from facenet_pytorch import MTCNN
from model.backbone import backbone_entry
from model.heads import heads_holder_entry
import yaml
import numpy as np
from easydict import EasyDict as edict
import re
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch import distributed


try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )



loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class MOEFaceUM_TEST(nn.Module):
    def __init__(self, backbone_module, heads_module):
        super(MOEFaceUM_TEST, self).__init__()
        self.backbone_module = backbone_module
        self.heads_module = heads_module

    def forward(self, data, **kwargs):

        task_j=kwargs["task_j"]
        task_name=kwargs["task_name"]
        outputs = dict()
        outputs[task_name] = dict()
        outputs[task_name]["task_name"] = task_name
        x = self.backbone_module(data,task_j=task_j)
        outputs[task_name]["backbone_output"] = x
        outputs = self.heads_module(outputs)
        return outputs

def load_state_model(model, state):

    msg = model.load_state_dict(state, strict=False)

    state_keys = set(state.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        printlog(f'missing key: {k}')
    printlog(f'load msg: {msg}')

def visualize_mask(image_tensor, mask):
    image = image_tensor.numpy().transpose(1, 2, 0) * 255 
    image = image.astype(np.uint8)
    
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mapping = np.array([
        [0, 0, 0],
        [0, 153, 255],
        [102, 255, 153],
        [0, 204, 153],
        [255, 255, 102],
        [255, 255, 204],
        [255, 153, 0],
        [255, 102, 255],
        [102, 0, 51],
        [255, 204, 255],
        [255, 100, 102],
        [255, 200, 102],
        [255, 150, 102],
        [255, 250, 102],
        [255, 150, 152],
        [255, 250, 152],
        [155, 150, 152],
        [155, 250, 152]

    ])
    
    for index, color in enumerate(color_mapping):
        color_mask[mask == index] = color

    overlayed_image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    return overlayed_image, image, color_mask

def visualize_landmarks(im, landmarks, color, thickness=3, eye_radius=0):
    im = im.permute(1, 2, 0).numpy()
    im = (im * 255).astype(np.uint8)
    im = np.ascontiguousarray(im)
    landmarks = landmarks.squeeze().numpy().astype(np.int32)
    for (x, y) in landmarks:
        cv2.circle(im, (x,y), eye_radius, color, thickness)
    return im



def denorm_points(points, h, w):
    denorm_points = points * torch.tensor([w, h], dtype=torch.float32).to(points).view(1, 1, 2)
    return denorm_points

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean 
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def adjust_bbox(x_min, y_min, x_max, y_max, image_width, image_height, margin_percentage=50):
    width = x_max - x_min
    height = y_max - y_min
    
    increase_width = width * (margin_percentage / 100.0) / 2
    increase_height = height * (margin_percentage / 100.0) / 2
    
    x_min_adjusted = max(0, x_min - increase_width)
    y_min_adjusted = max(0, y_min - increase_height)
    x_max_adjusted = min(image_width, x_max + increase_width)
    y_max_adjusted = min(image_height, y_max + increase_height)
    
    return x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted


def test(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=loader)
    cfg = edict(config["common"])

    backbone = backbone_entry(cfg.backbone)
    heads_holder = heads_holder_entry(cfg.heads)

    model = MOEFaceUM_TEST(backbone, heads_holder)
    model = model.cuda() 
    model = torch.nn.parallel.DistributedDataParallel(module=model, broadcast_buffers=False, 
                                                          device_ids=[local_rank], bucket_cap_mb=16,
                                                          find_unused_parameters=True)
    model.register_comm_hook(None, fp16_compress_hook)
    # raise Exception(model.state_dict().keys())
    weights_path = args.model_path
    state_dict = torch.load(weights_path)
    msg = model.load_state_dict(state_dict['state_dict'], strict=False)
    state_keys = set(state_dict['state_dict'].keys())
    # raise Exception(state_keys)
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        print(f'missing key: {k}')

    print(f'load msg: {msg}')
    # raise Exception("stop")
    device = "cuda:" + str(args.gpu_num)
    model.eval()

    transforms_image_112 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(112,112), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    transforms_image_512 = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(512,512), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    mtcnn = MTCNN(keep_all=True)
    image = Image.open(args.image_path)
    width, height = image.size
    boxes, probs = mtcnn.detect(image)
    x_min, y_min, x_max, y_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height)
    image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    image_112 = transforms_image_112(image)
    image_512 = transforms_image_512(image)

   
    images_112 = image_112.unsqueeze(0).to(device=device)
    images_512 = image_512.unsqueeze(0).to(device=device)
    # raise Exception(f"image_112:{image_112.shape},image_512:{image_512.shape}")
    for i in range(6):
        task_j = i #kwargs["task_j"]
        if task_j==0:##Face Recognition
            feature_embedding = model(images_112,task_j=task_j,task_name="recog_ms1mv3")
            # print(feature_embedding['recog_ms1mv3']["head_output"].shape)
            save_face_feature=os.path.join(args.results_path, "Identity_recog.txt")
            with open(save_face_feature, "w") as f:
                for feature in feature_embedding['recog_ms1mv3']["head_output"].squeeze().tolist():
                    f.write(f"{feature}\n")

        elif task_j==1:##Age Estimation
            face_age = model(images_112,task_j=task_j,task_name="age_morph2")
            # print(face_age["age_morph2"]["head_output"].shape)
            bias_correction = 30.0
            logits = face_age["age_morph2"]["head_output"]  # shape: (B, 101)
            age_probs = F.softmax(logits, dim=1)            # 使用 softmax 而不是 sigmoid
            rank = torch.arange(101, dtype=torch.float32).cuda()
            age_preds = torch.sum(age_probs * rank, dim=1) + bias_correction  # shape: (B,)

            print(f"Age Estimation:{age_preds.item()}")

        elif task_j==2:##Attributes Recognition
            face_attributes = model(images_112,task_j=task_j,task_name="biattr_celeba")
            # print(face_attributes["biattr_celeba"]["head_output"].shape)
            celeba_attributes = [
                "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
                "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
                "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
                "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
                "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
                "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
            ]

            binary_output = (face_attributes["biattr_celeba"]["head_output"] > 0).int().squeeze().tolist() 
            celeb_attribute_save=os.path.join(args.results_path, "Attributes_binary.txt")
            with open(celeb_attribute_save, "w") as f:
                for attr, value in zip(celeba_attributes, binary_output):
                    f.write(f"{attr}: {value}\n")

        elif task_j==3:##Expression Recognition
            dist_empression={
                "surprise":3,
                "fear":4,
                "sad":2,
                "happy":1,
                "disgust":5,
                "neutral":0,
                "anger":6,
                "contempt":7#pass
                }
            reversed_dict = {v: k for k, v in dist_empression.items()}
            face_expression = model(images_112,task_j=task_j,task_name="affect_affectnet")
            # print(face_expression["affect_affectnet"]["head_output"].shape)
            expresion_preds = torch.argmax(face_expression["affect_affectnet"]["head_output"], dim=1)[0]
            expression_pre = reversed_dict.get(int(expresion_preds.item()))
            print(f"expression_pre:{expression_pre}")
        elif task_j==4:##Face Parsing
            face_parsing = model(images_512,task_j=task_j,task_name="parsing_celebam")
            # print(face_parsing["parsing_celebam"]["head_output"].shape)
            preds = face_parsing["parsing_celebam"]["head_output"].softmax(dim=-1)
            mask = torch.argmax(preds, dim=-1)
            pred_mask = mask[0].detach().cpu().numpy()
            save_path = os.path.join(args.results_path, "Dense_parsing.png")
            cv2.imwrite(f"{save_path}", pred_mask)
            mask, face, color_mask = visualize_mask(unnormalize(images_512[0].detach().cpu()), pred_mask)
            save_path = os.path.join(args.results_path, "Dense_parsing_visualization.png")
            cv2.imwrite(f"{save_path}", mask[:, :, ::-1])
        elif task_j==5:##Face landmark
            face_landmark = model(images_512,task_j=task_j,task_name="align_300w")
            # print(face_landmark["align_300w"]["head_output"]["landmark"].shape)
            image = unnormalize(images_512[0].detach().cpu())
            denorm_landmarks = denorm_points(face_landmark["align_300w"]["head_output"]["landmark"].view(-1,68,2)[0],512,512)
            im = visualize_landmarks(image, denorm_landmarks.detach().cpu(), (255, 255, 0))
            save_path_viz = os.path.join(args.results_path, "Dense_landmarks.png")
            save_path = os.path.join(args.results_path, "Dense_landmarks.txt")
            cv2.imwrite(f"{save_path_viz}", im[:, :, ::-1])
            with open(f'{save_path}', 'w') as file:
                for landmark in denorm_landmarks[0]:
                    x, y = landmark[0], landmark[1]
                    file.write(f"{x.item()} {y.item()}\n")
            file.close()

    


    # if tasks[0] == 0:
    #     preds = seg_output.softmax(dim=1)
    #     mask = torch.argmax(preds, dim=1)
    #     pred_mask = mask[0].detach().cpu().numpy()
    #     save_path = os.path.join(args.results_path, "parsing.png")
    #     cv2.imwrite(f"{save_path}", pred_mask)
    #     mask, face, color_mask = visualize_mask(unnormalize(images[0].detach().cpu()), pred_mask)
    #     save_path = os.path.join(args.results_path, "parsing_visualization.png")
    #     cv2.imwrite(f"{save_path}", mask[:, :, ::-1])
    # if tasks[0] == 1:


    # if tasks[0] == 3:
    #     probs = torch.sigmoid(attribute_output[0])
    #     preds = (probs >= 0.5).float()
    #     pred = preds.tolist()
    #     pred_str = [str(int(b)) for b in pred]
    #     joined_pred = " ".join(pred_str)
    #     save_path = os.path.join(args.results_path, "attribute.txt")
    #     with open(f'{save_path}', 'w') as file:
    #         file.write(joined_pred)
    #     file.close()
    # if tasks[0] == 4:
    #     age_preds = torch.argmax(age_output, dim=1)[0]
    #     gender_preds = torch.argmax(gender_output, dim=1)[0]
    #     race_preds = torch.argmax(race_output, dim=1)[0]
    #     save_path = os.path.join(args.results_path, "age_gender_race.txt")
    #     with open(f'{save_path}', 'w') as file:
    #         file.write(f"Age: {age_preds.item()} \n")
    #         file.write(f"Gender: {gender_preds.item()} \n")
    #         file.write(f"Race: {race_preds.item()}")
    #     file.close()
    # if tasks[0] == 5:
    #     probs = torch.sigmoid(visibility_output[0])
    #     preds = (probs >= 0.5).float()
    #     pred = preds.tolist()
    #     pred_str = [str(int(b)) for b in pred]
    #     joined_pred =  " ".join(pred_str)
    #     save_path = os.path.join(args.results_path, "visibility.txt")
    #     with open(f'{save_path}', 'w') as file:
    #         file.write(joined_pred)
    #     file.close()
    # image = unnormalize(images[0].detach().cpu())
    # image = image.permute(1, 2, 0).numpy()
    # image = (image * 255).astype(np.uint8)
    # save_path = os.path.join(args.results_path, "face.png")
    # cv2.imwrite(f"{save_path}", image[:, :, ::-1])


            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Provide absolute path to your config_path file")
    parser.add_argument("--model_path", type=str, help="Provide absolute path to your weights file")
    parser.add_argument("--image_path", type=str, help="Provide absolute path to the image you want to perform inference on")
    parser.add_argument("--results_path", type=str, help="Provide path to the folder where results need to be saved")
    parser.add_argument("--gpu_num", type=str, help="Provide the gpu number")
    args = parser.parse_args()  
    test(args)