"""
���������� - �Զ��ж���Ƶ�������ͣ���̬/��̬�������ṩ���Ŷȡ�
"""

import numpy as np


class SceneClassifier:
    """
    ���������� - �Զ��ж���Ƶ�������ͣ���̬/��̬�������ṩ���Ŷȡ�
    """
    
    def __init__(self,
                 static_threshold: float = 0.1,
                 low_dynamic_threshold: float = 0.3,
                 medium_dynamic_threshold: float = 0.6,
                 high_dynamic_threshold: float = 1.0,
                 motion_ratio_threshold: float = 1.5):
        """
        ��ʼ������������
        
        Args:
            static_threshold: ��̬������ֵ�����嶯̬�ȣ�
            low_dynamic_threshold: �Ͷ�̬������ֵ
            medium_dynamic_threshold: �еȶ�̬������ֵ
            high_dynamic_threshold: �߶�̬������ֵ
            motion_ratio_threshold: �˶�������ֵ������/��������������������˶��������˶�
        """
        self.static_threshold = static_threshold
        self.low_dynamic_threshold = low_dynamic_threshold
        self.medium_dynamic_threshold = medium_dynamic_threshold
        self.high_dynamic_threshold = high_dynamic_threshold
        self.motion_ratio_threshold = motion_ratio_threshold
    
    def classify_scene(self,
                     background_motion: float,
                     subject_motion: float,
                     pure_subject_motion: float,
                     motion_ratio: float) -> dict:
        """
        ���ೡ������
        
        Args:
            background_motion: ������̬��
            subject_motion: ���嶯̬��
            pure_subject_motion: �����嶯̬�ȣ������ȥ������
            motion_ratio: �˶����ʣ�����/������
            
        Returns:
            �����������͡�ǿ�ȵȼ������ŶȵĽ���ֵ�
        """
        # 1) �ж�����˶��������������˶�����
        motion_dominant_type = self._determine_motion_dominant(
            background_motion, subject_motion, motion_ratio
        )
        
        # 2) ���ڴ����嶯̬���ж�ǿ�ȵȼ�
        intensity_level = self._determine_intensity_level(pure_subject_motion)
        
        # 3) �ۺ��жϵõ���������������
        scene_type, scene_description = self._determine_scene_type(
            motion_dominant_type, intensity_level, 
            background_motion, subject_motion, motion_ratio
        )
        
        return {
            'scene_type': scene_type,
            'scene_description': scene_description,
            'motion_dominant': motion_dominant_type,
            'intensity_level': intensity_level,
            'background_motion': float(background_motion),
            'subject_motion': float(subject_motion),
            'pure_subject_motion': float(pure_subject_motion),
            'motion_ratio': float(motion_ratio),
            'confidence': self._calculate_confidence(
                background_motion, subject_motion, motion_ratio
            )
        }
    
    def _determine_motion_dominant(self,
                                   background_motion: float,
                                   subject_motion: float,
                                   motion_ratio: float) -> str:
        """�ж��˶���������"""
        
        # ����˶�����С����ֵ��˵�����岢������ǿ�ڱ���
        if motion_ratio < self.motion_ratio_threshold:
            return 'camera_motion'  # ����˶�����
        # ��������˶����Դ��ڱ����˶�
        elif motion_ratio >= self.motion_ratio_threshold:
            return 'object_motion'  # �����˶�����
        else:
            return 'mixed_motion'  # ����˶�
    
    def _determine_intensity_level(self, pure_subject_motion: float) -> str:
        """�ж��˶�ǿ�ȵȼ�"""
        
        if pure_subject_motion < self.static_threshold:
            return 'static'
        elif pure_subject_motion < self.low_dynamic_threshold:
            return 'low_dynamic'
        elif pure_subject_motion < self.medium_dynamic_threshold:
            return 'medium_dynamic'
        elif pure_subject_motion < self.high_dynamic_threshold:
            return 'high_dynamic'
        else:
            return 'extreme_dynamic'
    
    def _determine_scene_type(self,
                              motion_dominant: str,
                              intensity_level: str,
                              background_motion: float,
                              subject_motion: float,
                              motion_ratio: float) -> tuple:
        """�ۺ��жϳ������Ͳ�������������"""
        
        # ����˶������ĳ���
        if motion_dominant == 'camera_motion':
            if intensity_level == 'static':
                return 'static_camera', '��̬����˶����������ƽ�ơ����ŵȣ�'
            elif intensity_level == 'low_dynamic':
                return 'low_dynamic_camera', '�Ͷ�̬����˶�����'
            else:
                return 'dynamic_camera', f'{intensity_level}����̬����˶�����'
        
        # �����˶������ĳ���
        elif motion_dominant == 'object_motion':
            if intensity_level == 'static':
                return 'static_object', '��̬���峡�������������ֹ��'
            elif intensity_level == 'low_dynamic':
                return 'low_dynamic_object', '�Ͷ�̬�����˶�����'
            elif intensity_level == 'medium_dynamic':
                return 'medium_dynamic_object', '�еȶ�̬�����˶�����'
            elif intensity_level == 'high_dynamic':
                return 'high_dynamic_object', '�߶�̬�����˶�����'
            else:
                return 'extreme_dynamic_object', '���߶�̬�����˶�����'
        
        # ����˶�����
        else:
            return 'mixed_scene', f'����˶�����������˶���{background_motion:.4f}�������˶���{subject_motion:.4f}��'
    
    def _calculate_confidence(self,
                             background_motion: float,
                             subject_motion: float,
                             motion_ratio: float) -> float:
        """��������������Ŷȣ�0~1��"""
        
        # �����˶����ʵ���ȷ�̶ȼ������Ŷ�
        if motion_ratio < 0.5 or motion_ratio > 2.0:
            # �˶����ʼ��ˣ�����ƫ��һ���������Ŷȸ�
            ratio_confidence = 0.9
        elif 0.8 < motion_ratio < 1.2:
            # �˶����ʽӽ�1���������֣������Ŷȵ�
            ratio_confidence = 0.3
        else:
            # �˶�������һ�����죬���Ŷ��е�
            ratio_confidence = 0.6
        
        # �����˶�ǿ�ȼ������Ŷ�
        total_motion = background_motion + subject_motion
        if 0.01 < total_motion < 5.0:
            intensity_confidence = 0.8  # ���ں�����˶���Χ
        else:
            intensity_confidence = 0.4  # �˶���С�����
        
        # �ۺ����Ŷ�
        confidence = (ratio_confidence + intensity_confidence) / 2
        
        return float(np.clip(confidence, 0.0, 1.0))

