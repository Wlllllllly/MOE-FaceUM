
from retinaface import RetinaFace
import cv2
import numpy as np
import torch
import os
from datetime import datetime

# def get_face_patch_indices(bbox, img_size=112, patch_size=16):
#     """
#     将人脸BBox转换为ViT的Patch Token索引列表
    
#     参数:
#         bbox: RetinaFace返回的边界框 [x1,y1,x2,y2]
#         img_size: 输入图像尺寸 (默认112)
#         patch_size: ViT的patch大小 (默认16)
        
#     返回:
#         list: 覆盖人脸的patch索引列表 (按行优先顺序)
#     """
#     # 计算patch网格数量
#     num_patches = img_size // patch_size
    
#     # 将BBox坐标转换为patch索引范围
#     x1, y1, x2, y2 = bbox
#     start_col = max(0, int(x1 // patch_size))
#     end_col = min(num_patches - 1, int(x2 // patch_size))
#     start_row = max(0, int(y1 // patch_size))
#     end_row = min(num_patches - 1, int(y2 // patch_size))
    
#     # 生成所有patch索引
#     patch_indices = []
#     for row in range(start_row, end_row + 1):
#         for col in range(start_col, end_col + 1):
#             # 计算线性索引 (行优先顺序)
#             idx = row * num_patches + col
#             patch_indices.append(idx)
    
#     return sorted(list(set(patch_indices)))  # 去重并排序

def get_face_patch_indices(bbox, img_size=112, patch_size=16):
    """
    将人脸BBox转换为ViT的Patch Token索引列表，并标注ROI/non-ROI分界点
    
    参数:
        bboxes: RetinaFace返回的边界框列表 [[x1,y1,x2,y2], ...]
        img_size: 输入图像尺寸 (默认112)
        patch_size: ViT的patch大小 (默认16)
        
    返回:
        tuple: (all_indices, split_point)
        - all_indices: 所有token索引 (前段ROI + 后段non-ROI)
        - split_point: ROI结束的索引位置 (len(ROI_indices))
    """
    num_patches = img_size // patch_size
    total_patches = num_patches ** 2
    # 1. 收集所有ROI patch索引 (人脸区域)
    if len(bbox) == 0:
        roi_indices = []
    else:
        roi_indices = []
        # 将BBox坐标转换为patch索引范围
        x1, y1, x2, y2 = bbox
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
                roi_indices.append(idx)
        
        # 去重并排序
        roi_indices = sorted(list(set(roi_indices)))
    
    # 2. 收集non-ROI patch索引 (背景区域)
    all_indices = list(range(total_patches))
    non_roi_indices = [idx for idx in all_indices if idx not in roi_indices]
    
    # 3. 合并为 [ROI... | non-ROI...] 结构
    combined = roi_indices + non_roi_indices

    split_point = len(roi_indices)  # 记录分界点

        
    return combined, split_point


def process_with_retinaface_and_vit(image_path):
    # 使用RetinaFace检测人脸
    img = cv2.imread(image_path) 
    if len(img.shape) == 2:  # 如果是灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (112, 112))  # 强制调整尺寸

    faces = RetinaFace.detect_faces(img,threshold=0.5)

    if not isinstance(faces, dict):
        bboxes = [] # 无人脸情况
    
    else:
        bboxes = faces['face_1']['facial_area']
        
    patch_indices, split_point = get_face_patch_indices(bboxes)
    return patch_indices, split_point 

    
    # 处理每个人脸的BBox
    # all_face_patches = []
    # for face_id, face_info in faces.items():
    #     bbox = face_info['facial_area']  # [x1,y1,x2,y2]

        # print(f"bbox:{bbox}")
        
        # 获取patch索引

    
        # all_face_patches.append({
        #     'face_id': face_id,
        #     'bbox': bbox,
        #     'patch_indices': patch_indices,
        #     'split_point': split_point,
        #     'num_patches': len(patch_indices)
        # })
    



def visualize_patches(image_path, patch_indices, bbox=None, patch_size=16, save_dir="output"):
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
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取和调整图像
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112))
    debug_img = img.copy()
    
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
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,0,255), 2)
    
    # 添加文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_img, f"Patches: {len(patch_indices)}", (5,15), font, 0.4, (255,255,255), 1)
    if bbox is not None:
        cv2.putText(debug_img, f"BBox: {bbox}", (5,30), font, 0.4, (255,255,255), 1)
    
    # 生成保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"{filename}_patches_{timestamp}.png")
    
    # 保存结果
    cv2.imwrite(save_path, debug_img)
    print(f"可视化结果已保存到: {save_path}")

    
    return debug_img


if __name__ == "__main__":
    result = process_with_retinaface_and_vit("/mnt/nas/CVAI_WLY/Face_datasets/CelebA/img_align_celeba/img_align_celeba/202598.jpg")
    
    if result:
        print(f"检测到 {len(result)} 张人脸")
        for i, face in enumerate(result):
            print(f"\n人脸 {i+1}:")
            print(f"BBox位置: {face['bbox']}")
            print(f"覆盖的Patch索引: {face['patch_indices']}")
            print(f"覆盖Patch数量: {face['num_patches']}")
            
            # 可视化显示前5个patch
            print("示例索引:", face['patch_indices'][:5], "...")
            visualize_patches(
            image_path="/mnt/nas/CVAI_WLY/Face_datasets/CelebA/img_align_celeba/img_align_celeba/202598.jpg",
            patch_indices=face['patch_indices'],
            bbox=face['bbox'],
            save_dir="/home/wly/HD/code/SwinFace/pictures"
            )
    else:
        print("未检测到人脸")