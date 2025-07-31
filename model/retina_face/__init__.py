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
    """
    将人脸BBox转换为ViT的Patch Token索引列表
    
    参数:
        bbox: RetinaFace返回的边界框 [x1,y1,x2,y2]
        img_size: 输入图像尺寸 (默认112)
        patch_size: ViT的patch大小 (默认16)
        
    返回:
        list: 覆盖人脸的patch索引列表 (按行优先顺序)
    """
    # 计算patch网格数量
    num_patches = img_size // patch_size
    total_patches = num_patches ** 2
    # 将BBox坐标转换为patch索引范围
    x1, y1, x2, y2, _ = bbox
    start_col = max(0, int(x1 // patch_size))
    end_col = min(num_patches - 1, int(x2 // patch_size))
    start_row = max(0, int(y1 // patch_size))
    end_row = min(num_patches - 1, int(y2 // patch_size))
    
    # 生成所有patch索引
    patch_indices = []
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            # 计算线性索引 (行优先顺序)
            idx = row * num_patches + col
            patch_indices.append(idx)
    
    # 去重并排序
    roi_indices = sorted(list(set(patch_indices)))
    
    # 2. 收集non-ROI patch索引 (背景区域)
    all_indices = list(range(total_patches))
    non_roi_indices = [idx for idx in all_indices if idx not in roi_indices]
    
    # 3. 合并为 [ROI... | non-ROI...] 结构
    combined = roi_indices + non_roi_indices
    split_point = len(roi_indices)  # 记录分界点
    # combined_all.append(combined)
    # split_point_all.append(split_point)
    
    return combined, split_point

def visualize_patches(img_raw, patch_indices, bbox=None, patch_size=16, save_dir="output"):
    """ 
    可视化显示被选中的patches并保存结果 
    
    参数:
        image_path: 输入图像路径
        patch_indices: 需要高亮的patch索引列表
        bbox: 可选,原始人脸BBox坐标 [x1,y1,x2,y2]
        patch_size: ViT的patch大小 (默认16)
        save_dir: 结果保存目录 (默认"output")
    """
    # 创建输出目录
    # os.makedirs(save_dir, exist_ok=True)
    
    # 读取和调整图像
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (112, 112))
    img=img_raw
    debug_img = img_raw.copy()
    
    # 绘制所有patch网格 (浅灰色)
    for i in range(0, 112, patch_size):
        cv2.line(debug_img, (i, 0), (i, 112), (200,200,200), 1)
        cv2.line(debug_img, (0, i), (112, i), (200,200,200), 1)
    
    # 高亮选中的patches (半透明绿色)
    num_patches_per_side = 112 // patch_size
    overlay = debug_img.copy()
    for idx in patch_indices:
        row = idx // num_patches_per_side
        col = idx % num_patches_per_side
        pt1 = (col*patch_size, row*patch_size)
        pt2 = ((col+1)*patch_size, (row+1)*patch_size)
        cv2.rectangle(overlay, pt1, pt2, (0,255,0), -1)
    
    # 添加透明度
    cv2.addWeighted(overlay, 0.3, debug_img, 0.7, 0, debug_img)
    
    # 绘制原始BBox (红色)
    if bbox is not None:
        # x1, y1, x2, y2,_= bbox
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])
        cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,0,255), 2)
    
    # 添加文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_img, f"Patches: {len(patch_indices)}", (5,15), font, 0.4, (255,255,255), 1)
    if bbox is not None:
        cv2.putText(debug_img, f"BBox: {bbox}", (5,30), font, 0.4, (255,255,255), 1)
    
    # 生成保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"patches_{timestamp}.png")
    
    # 保存结果
    cv2.imwrite(save_path, debug_img)
    # print(f"可视化结果已保存到: {save_path}")

    
    return debug_img

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
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
    # 减去均值：注意顺序为 BGR → RGB，如果原图像是 RGB，请换成 (123,117,104)
    mean = torch.tensor([123.0, 117.0, 104.0], device=device).view(1, 3, 1, 1)#[104.0, 117.0, 123.0] 
    input_tensor -= mean  # broadcasting to all pixels
    results = []
    combined_all = []
    split_point_all = []
    for i in range(batch_size):
        img = input_tensor[i].unsqueeze(0)  # shape [1, 3, H, W]
        # img_raw = input_tensor[i].detach().cpu().numpy()  # shape: [3, H, W]
        # img_raw = img_raw.transpose(1, 2, 0).astype(np.uint8)  # shape: [H, W, 3]
        # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)     # convert to BGR for OpenCV
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
            # img_raw = input_tensor[i].detach().cpu().numpy()  # shape: [3, H, W]
            # img_raw = img_raw.transpose(1, 2, 0).astype(np.uint8)  # shape: [H, W, 3]
            # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)     # convert to BGR for OpenCV
            # name = f'/home/wly/HD/code/Face_MoE/core/model/retina_face/fig/onface_detection_{i}' + '.jpg'
            # cv2.imwrite(name, img_raw)

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

            # 保留top_k
            dets = dets[:750]#args.keep_top_k
            # print(f"dets:{dets}")

            # ##visual results
            combined, split_point = get_face_patch_indices(dets[0],img_size=im_height)
        results.append(dets[0])
        combined_all.append(combined)
        split_point_all.append(split_point)

        # return combined, split_point
        # print(f"patch_indices:{patch_indices}")
        # visualize_patches(
        #     img_raw = img_raw,
        #     patch_indices=patch_indices,
        #     bbox=dets[0],
        #     save_dir="/home/wly/HD/code/Face_MoE/core/model/retina_face/fig"
        #     )


        # raise Exception(f"stop")


        # 最终输出（按vis_thres过滤）
        # image_results = []
        
        
        # for b in dets:
        #     if b[4] < 0.5:#args.vis_thres
        #         continue
            # print(f"b:{b}")
            # text = "{:.4f}".format(b[4])
            # b = list(map(int, b))
            # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            # cx = b[0]
            # cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # x1, y1, x2, y2, score = map(int, b)
            # image_results.append([x1, y1, x2, y2, score])
        # print(f"img_raw:{img_raw.shape}")
        # name = f'/home/wly/HD/code/Face_MoE/core/model/retina_face/fig/face_detection' + '.jpg'
        # cv2.imwrite(name, img_raw)
        # raise Exception(f"stop")

        # results.append(dets[0])

    return results, combined_all, split_point_all # list of length B，每个元素是 list of [x1,y1,x2,y2,score]




def Retina_detect(input_tensor):
    cfg = cfg_gnet#cfg_re50#cfg_mnet#cfg_gnet#cfg_mnetv3
    pretrained_model="/home/wly/HD/code/Face_MoE/core/model/retina_face/weights/ghostnet_Final.pth"#Resnet50_Final.pth"#ghostnet_Final.pth"#mobilev3_Final.pth"
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, pretrained_model, False)
    net.eval()
    device = torch.device("cuda")
    net = net.to(device)
    results, combined_all, split_point_all = detect_faces_tensor(input_tensor, net, device, cfg)

    # all_face_patches.append({
    #     'face_id': face_id,
    #     'bbox': bbox,
    #     'patch_indices': patch_indices,
    #     'split_point': split_point,
    #     'num_patches': len(patch_indices)
    # })
    # print(face_det)
    # raise Exception("stop")
    return results, combined_all, split_point_all

