# �ɸ�֪�������� (Perceptible Amplitude Score)

## ����

��ģ��ʹ�� **Grounded-SAM** �� **Co-Tracker** ������Ƶ�������뱳�����˶����ȣ�������ɸ�֪�˶����ȷ�����

## �����ص�

- **����/��������**��ʹ�� Grounding DINO + SAM ��������ָ�
- **�˶�����**��ʹ�� Co-Tracker ���е���٣������˶�����
- **��������**���Զ��жϳ������ͣ���̬/��̬����˶��������˶��ȣ�
- **�ɸ�֪����**����������˶��������˶����������˶��ȶ���ָ��

## ����ģ��

��ȷ������ģ���ļ������ص� `.cache` Ŀ¼��

```
.cache/
������ groundingdino_swinb_cogcoor.pth  # GroundingDINOģ��
������ sam_vit_h_4b8939.pth             # SAMģ��
������ scaled_offline.pth               # Co-Trackerģ��
������ google-bert/                     # BERTģ��
    ������ bert-base-uncased/
```

### ģ������˵��

1. **GroundingDINOģ��**: `groundingdino_swinb_cogcoor.pth`
   - ���ص�ַ: https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
   - ���浽: `.cache/groundingdino_swinb_cogcoor.pth`

2. **SAMģ��**: `sam_vit_h_4b8939.pth`
   - ���ص�ַ: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   - ���浽: `.cache/sam_vit_h_4b8939.pth`

3. **Co-Trackerģ��**: `scaled_offline.pth`
   - ���ص�ַ: https://huggingface.co/facebook/cotracker/resolve/main/cotracker_scaled_offline.pth
   - ���浽: `.cache/scaled_offline.pth`

4. **BERTģ��**: `google-bert/bert-base-uncased/`
   - ����ʹ�� Hugging Face �� transformers ���Զ�����
   - ���ֶ����ص�: `.cache/google-bert/bert-base-uncased/`

## ʹ�÷���

### ����ʹ��

```python
from src.aux_motion_intensity_2 import PASAnalyzer

# ��ʼ��������
analyzer = PASAnalyzer(
    device="cuda",
    enable_scene_classification=True
)

# ����������Ƶ
result = analyzer.analyze_video(
    video_path="path/to/video.mp4",
    subject_noun="person"  # ��������
)

print(result)
```

### ��������

```python
from src.aux_motion_intensity_2.batch import batch_analyze_videos

# ׼��Ԫ��Ϣ�б�
meta_infos = [
    {
        'filepath': 'video1.mp4',
        'subject_noun': 'person',
        'prompt': 'A person walking',
        'index': 0
    },
    # ... ������Ƶ
]

# ��������
results = batch_analyze_videos(
    analyzer=analyzer,
    meta_info_list=meta_infos,
    output_path='pas_results.json'
)
```

### ʹ�ýű�

�μ� `scripts/aux_motion_intensity_2/run_pas.py`

## �����ʽ

### �ɹ�����

```json
{
  "status": "success",
  "background_motion": 0.0234,
  "subject_motion": 0.0567,
  "pure_subject_motion": 0.0333,
  "total_motion": 0.0801,
  "motion_ratio": 1.423,
  "video_resolution": {
    "width": 1920,
    "height": 1080,
    "diagonal": 2202.9
  },
  "scene_classification": {
    "scene_type": "low_dynamic_object",
    "scene_description": "�Ͷ�̬�����˶�����",
    "motion_dominant": "object_motion",
    "intensity_level": "low_dynamic",
    "confidence": 0.75
  }
}
```

### ������

```json
{
  "status": "error",
  "error_reason": "no_subject_detected",
  "background_motion": 0.0234
}
```

## ����˵��

### PASAnalyzer����

- `device`: �豸���� ("cuda" �� "cpu")
- `grid_size`: Co-Tracker�����С��Ĭ��30��
- `enable_scene_classification`: �Ƿ����ó�������
- `scene_classifier_params`: ���������������ֵ�

### analyze_video����

- `video_path`: ��Ƶ�ļ�·��
- `subject_noun`: �������ʣ�����GroundingDINO��⣩
- `box_threshold`: ������ֵ��Ĭ��0.3��
- `text_threshold`: �ı���ֵ��Ĭ��0.25��
- `normalize_by_subject_diag`: �Ƿ�����Խ��߹�һ��

## ��������

### ��������

- `static_camera`: ��̬����˶�����
- `low_dynamic_camera`: �Ͷ�̬����˶�����
- `dynamic_camera`: ��̬����˶�����
- `static_object`: ��̬���峡��
- `low_dynamic_object`: �Ͷ�̬�����˶�����
- `medium_dynamic_object`: �еȶ�̬�����˶�����
- `high_dynamic_object`: �߶�̬�����˶�����
- `extreme_dynamic_object`: ���߶�̬�����˶�����
- `mixed_scene`: ����˶�����

### �˶���������

- `camera_motion`: ����˶�����
- `object_motion`: �����˶�����
- `mixed_motion`: ����˶�

## ע������

1. �״�������Ҫ����ģ���ļ�
2. ����ʹ��GPU���٣�device="cuda"��
3. ��������Ӧ����Ƶ��ʵ������ƥ��
4. �˶������ѹ�һ�����ɿ�ֱ��ʱȽ�

## ����������

- Grounded-Segment-Anything (third_party/)
- Co-Tracker (third_party/)
- GroundingDINO (third_party/Grounded-Segment-Anything/GroundingDINO/)
- Segment-Anything (third_party/Grounded-Segment-Anything/segment_anything/)

