# -*- coding: utf-8 -*-
"""
关键点可视化工具
支持可视化MediaPipe提取的身体、手部和面部关键点
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


class KeypointVisualizer:
    """关键点可视化器"""
    
    # 身体关键点连接（MediaPipe Pose的33个关键点）
    POSE_CONNECTIONS = [
        # 面部轮廓
        (0, 1), (1, 2), (2, 3), (3, 7),
        # 左眼
        (0, 4), (4, 5), (5, 6), (6, 8),
        # 右眼
        (2, 9), (9, 10), (10, 11), (11, 12),
        # 身体
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # 左臂
        (11, 23), (23, 24), (24, 26), (26, 28), (28, 30), (30, 32),
        # 右臂
        (12, 25), (25, 27), (27, 29), (29, 31), (31, 33),
        # 躯干
        (23, 24), (24, 12), (12, 11), (11, 23),
    ]
    
    # 手部关键点连接（每只手21个关键点）
    HAND_CONNECTIONS = [
        # 拇指
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 食指
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 中指
        (0, 9), (9, 10), (10, 11), (11, 12),
        # 无名指
        (0, 13), (13, 14), (14, 15), (15, 16),
        # 小指
        (0, 17), (17, 18), (18, 19), (19, 20),
        # 手掌
        (5, 9), (9, 13), (13, 17),
    ]
    
    # 颜色定义
    COLORS = {
        'body': (0, 255, 0),        # 绿色 - 身体
        'left_hand': (255, 0, 0),   # 蓝色 - 左手
        'right_hand': (0, 0, 255),  # 红色 - 右手
        'face': (255, 255, 0),      # 青色 - 面部
    }
    
    def __init__(self, 
                 show_face: bool = False,
                 show_face_mesh: bool = False,
                 point_radius: int = 3,
                 line_thickness: int = 2):
        """
        初始化可视化器
        
        Args:
            show_face: 是否显示面部关键点（468个点，默认False）
            show_face_mesh: 是否显示面部网格（仅显示轮廓，默认False）
            point_radius: 关键点半径
            line_thickness: 连接线粗细
        """
        self.show_face = show_face
        self.show_face_mesh = show_face_mesh
        self.point_radius = point_radius
        self.line_thickness = line_thickness
    
    def visualize(self, 
                 image: np.ndarray, 
                 keypoints: Dict,
                 output_path: Optional[str] = None,
                 show: bool = True) -> np.ndarray:
        """
        可视化关键点
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
            keypoints: 关键点字典，包含body, left_hand, right_hand, face
            output_path: 输出图像路径（可选）
            show: 是否显示图像
        
        Returns:
            可视化后的图像
        """
        # 复制图像，避免修改原图
        vis_image = image.copy()
        
        # 确保图像是BGR格式
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        elif vis_image.shape[2] == 3:
            # 如果是RGB，转换为BGR
            if vis_image.dtype == np.uint8 and vis_image.max() <= 255:
                pass  # 已经是BGR或需要检查
        else:
            raise ValueError(f"不支持的图像格式: {vis_image.shape}")
        
        h, w = vis_image.shape[:2]
        
        # 可视化身体关键点
        if keypoints.get('body') is not None:
            vis_image = self._draw_pose(vis_image, keypoints['body'], w, h)
        
        # 可视化左手关键点
        if keypoints.get('left_hand') is not None:
            vis_image = self._draw_hand(vis_image, keypoints['left_hand'], w, h, 'left_hand')
        
        # 可视化右手关键点
        if keypoints.get('right_hand') is not None:
            vis_image = self._draw_hand(vis_image, keypoints['right_hand'], w, h, 'right_hand')
        
        # 可视化面部关键点
        if keypoints.get('face') is not None and (self.show_face or self.show_face_mesh):
            vis_image = self._draw_face(vis_image, keypoints['face'], w, h)
        
        # 保存图像
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"可视化结果已保存到: {output_path}")
        
        # 显示图像
        if show:
            cv2.imshow('Keypoint Visualization', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return vis_image
    
    def _draw_pose(self, image: np.ndarray, keypoints: np.ndarray, w: int, h: int) -> np.ndarray:
        """绘制身体姿态关键点"""
        if keypoints is None or len(keypoints) == 0:
            return image
        
        # 将归一化坐标转换为像素坐标
        points = []
        for kp in keypoints:
            x = int(kp[0] * w)
            y = int(kp[1] * h)
            points.append((x, y))
        
        # 绘制连接线
        for connection in self.POSE_CONNECTIONS:
            if connection[0] < len(points) and connection[1] < len(points):
                pt1 = points[connection[0]]
                pt2 = points[connection[1]]
                cv2.line(image, pt1, pt2, self.COLORS['body'], self.line_thickness)
        
        # 绘制关键点
        for pt in points:
            cv2.circle(image, pt, self.point_radius, self.COLORS['body'], -1)
        
        return image
    
    def _draw_hand(self, image: np.ndarray, keypoints: np.ndarray, w: int, h: int, hand_type: str) -> np.ndarray:
        """绘制手部关键点"""
        if keypoints is None or len(keypoints) == 0:
            return image
        
        color = self.COLORS[hand_type]
        
        # 将归一化坐标转换为像素坐标
        points = []
        for kp in keypoints:
            x = int(kp[0] * w)
            y = int(kp[1] * h)
            points.append((x, y))
        
        # 绘制连接线
        for connection in self.HAND_CONNECTIONS:
            if connection[0] < len(points) and connection[1] < len(points):
                pt1 = points[connection[0]]
                pt2 = points[connection[1]]
                cv2.line(image, pt1, pt2, color, self.line_thickness)
        
        # 绘制关键点
        for pt in points:
            cv2.circle(image, pt, self.point_radius, color, -1)
        
        return image
    
    def _draw_face(self, image: np.ndarray, keypoints: np.ndarray, w: int, h: int) -> np.ndarray:
        """绘制面部关键点"""
        if keypoints is None or len(keypoints) == 0:
            return image
        
        color = self.COLORS['face']
        
        # 将归一化坐标转换为像素坐标
        points = []
        for kp in keypoints:
            x = int(kp[0] * w)
            y = int(kp[1] * h)
            points.append((x, y))
        
        if self.show_face:
            # 显示所有面部关键点
            for pt in points:
                cv2.circle(image, pt, 1, color, -1)
        elif self.show_face_mesh:
            # 仅显示面部轮廓（前68个点通常是轮廓点）
            # MediaPipe面部有468个点，前部分通常是轮廓
            contour_points = points[:68] if len(points) >= 68 else points
            for i in range(len(contour_points) - 1):
                cv2.line(image, contour_points[i], contour_points[i + 1], color, 1)
            if len(contour_points) > 0:
                cv2.line(image, contour_points[-1], contour_points[0], color, 1)
        
        return image


def visualize_keypoints_from_image(image_path: str,
                                   output_path: Optional[str] = None,
                                   cache_dir: str = ".cache",
                                   show_face: bool = False,
                                   show: bool = True):
    """
    从图像文件可视化关键点
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        cache_dir: 模型缓存目录
        show_face: 是否显示面部关键点
        show: 是否显示图像
    """
    from .keypoint_extractor import MediaPipeKeypointExtractor
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 转换为RGB（MediaPipe需要RGB格式）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 提取关键点
    print("正在提取关键点...")
    extractor = MediaPipeKeypointExtractor(cache_dir=cache_dir)
    keypoints = extractor.extract_keypoints(image_rgb)
    
    # 可视化
    print("正在可视化关键点...")
    visualizer = KeypointVisualizer(show_face=show_face)
    visualizer.visualize(image, keypoints, output_path=output_path, show=show)
    
    print("可视化完成！")


def visualize_keypoints_from_video(video_path: str,
                                    output_path: Optional[str] = None,
                                    cache_dir: str = ".cache",
                                    show_face: bool = False,
                                    frame_interval: int = 1,
                                    max_frames: Optional[int] = None):
    """
    从视频文件可视化关键点
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        cache_dir: 模型缓存目录
        show_face: 是否显示面部关键点
        frame_interval: 帧间隔（每隔N帧处理一次）
        max_frames: 最大处理帧数（可选）
    """
    from .keypoint_extractor import MediaPipeKeypointExtractor
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
    
    # 创建输出视频写入器
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"输出视频: {output_path}")
    
    # 初始化提取器和可视化器
    extractor = MediaPipeKeypointExtractor(cache_dir=cache_dir)
    visualizer = KeypointVisualizer(show_face=show_face)
    
    frame_count = 0
    processed_count = 0
    
    print("正在处理视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按帧间隔处理
        if frame_count % frame_interval == 0:
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 提取关键点
            keypoints = extractor.extract_keypoints(frame_rgb, fps=fps)
            
            # 可视化
            vis_frame = visualizer.visualize(frame, keypoints, show=False)
            
            # 写入输出视频
            if out:
                out.write(vis_frame)
            
            processed_count += 1
            print(f"已处理 {processed_count} 帧 (总帧数: {frame_count + 1}/{total_frames})")
            
            # 限制最大帧数
            if max_frames and processed_count >= max_frames:
                break
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    if out:
        out.release()
    
    print(f"处理完成！共处理 {processed_count} 帧")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="关键点可视化工具")
    parser.add_argument("input", type=str, help="输入图像或视频路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径（可选）")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="模型缓存目录")
    parser.add_argument("--show-face", action="store_true", help="显示面部关键点")
    parser.add_argument("--no-show", action="store_true", help="不显示图像（仅保存）")
    parser.add_argument("--frame-interval", type=int, default=1, help="视频帧间隔（默认1）")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数（仅视频）")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        exit(1)
    
    # 判断是图像还是视频
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    if input_path.suffix.lower() in image_extensions:
        # 处理图像
        visualize_keypoints_from_image(
            str(input_path),
            output_path=args.output,
            cache_dir=args.cache_dir,
            show_face=args.show_face,
            show=not args.no_show
        )
    elif input_path.suffix.lower() in video_extensions:
        # 处理视频
        visualize_keypoints_from_video(
            str(input_path),
            output_path=args.output,
            cache_dir=args.cache_dir,
            show_face=args.show_face,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
    else:
        print(f"错误: 不支持的文件格式: {input_path.suffix}")
        exit(1)

