# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\main.py
# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np

# 1. 首先加载运行环境：包括路径挂载和环境变量修复
from environment import setup_environment
config = setup_environment()

# 2. 导入核心重构模块
from core.matcher import ELoFTRMatcher
from core.geometry import GeoTransformer
from core.processor import ImageProcessor
from utils.logger import MatchLogger

def main():
    """
    定位系统主入口：执行基于 EfficientLoFTR 的卫星底图纠偏定位，并实时分析误差。
    """
    # 初始化核心业务模块
    matcher = ELoFTRMatcher(config)
    geo_tool = GeoTransformer(config['paths']['base_image'])
    processor = ImageProcessor()
    logger = MatchLogger(config)
    
    # 将全量卫星基准图加载至内存
    full_base = cv2.imread(config['paths']['base_image'])
    if full_base is None:
        print("[CRITICAL] 无法加载基准图，请核对 yml 路径。")
        return

    # 【初始阶段：锁定理论位姿】
    init = config['initial_state']
    curr_height = init['h_o'] - init['h0']
    corners_gps = geo_tool.get_view_corners_gps(init['lng_o'], init['lat_o'], curr_height, init['beta_o'], config['camera']['fov'])
    img_xy = geo_tool.lonlat_to_pixel_list(corners_gps)
    
    # 扫描实时图像目录
    input_dir = config['paths']['realtime_dir']
    img_list = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    
    prev_xy = img_xy # 匹配失败时的回退状态
    
    print(f"\n[SYSTEM] 启动序列处理 | 误差计算模式: {'ENABLED' if logger.calculate_error else 'DISABLED'}")

    for img_name in img_list:
        t_start = time.time()
        frame = cv2.imread(os.path.join(input_dir, img_name))
        if frame is None: continue

        # --- 流程一：基准图局部动态采样与航向旋转对齐 ---
        base_roi, roi_y, roi_x = processor.get_base_roi(full_base, img_xy)
        local_corners = [[p[0]-roi_x, p[1]-roi_y] for p in img_xy]
        base_crop, rot_m, crop_y, crop_x = processor.rotate_and_crop(base_roi, local_corners)
        
        # --- 流程二：EfficientLoFTR 深度特征匹配 ---
        dst_pts_crop, G, count, _, match_data = matcher.match(frame, base_crop)
        
        # --- 流程三：矩阵校验与坐标逆向还原 ---
        is_valid = processor.validate_homography(G, config['matcher']['valid_k'])
        
        if is_valid and dst_pts_crop is not None:
            # 执行核心逆变换：由匹配空间映射回全量基准图空间
            actual_xy = processor.map_points_back(dst_pts_crop, base_crop.shape, base_roi.shape, rot_m, crop_y, crop_x, roi_y, roi_x)
            img_xy = actual_xy 
            
            # 计算中心点的经纬度及投影坐标
            _, _, lon, lat, px, py = geo_tool.get_center_info(actual_xy)
            # 记录定位结果，并自动根据真值计算三向物理误差
            logger.log_success(img_name, lon, lat, px, py, time.time() - t_start)
            
            # 保存全局大图映射效果 (PNG)
            if config['matcher'].get('save_full_overlay', True):
                logger.save_full_overlay(full_base, frame, actual_xy, img_name, config['matcher'].get('full_overlay_scale', 0.5))

            # 保存特征连线分析图
            if config['matcher'].get('save_matching_fig', True) and match_data:
                logger.save_matching_plot(frame, base_crop, *match_data, img_name)
        else:
            # 匹配失效处理
            actual_xy = prev_xy
            _, _, lon, lat, px, py = geo_tool.get_center_info(actual_xy)
            logger.log_fail(img_name, lon, lat, px, py, time.time() - t_start, f"点数:{count}")

        # 状态更新
        prev_xy = actual_xy
        img_xy = processor.expand_rect(actual_xy, config['matcher'].get('mag_k', 1.1))

    print(f"\n[SYSTEM] 任务全部处理完成。")

if __name__ == "__main__":
    main()