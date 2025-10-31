# �ɸ�֪�������ֽű�

## ʹ�÷���

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json \
    --output_path results/pas_results.json \
    --enable_scene_classification \
    --device cuda
```

## ����˵��

### �������

- `--meta_info_path`: Ԫ��ϢJSON�ļ�·�������裩

### ��ѡ����

#### �������
- `--output_path`: ���JSON�ļ�·����Ĭ�ϣ����������ļ���

#### ģ�Ͳ���
- `--device`: �豸���� (cuda/cpu��Ĭ��: cuda)
- `--grid_size`: Co-Tracker�����С��Ĭ��: 30��
- `--box_threshold`: GroundingDINO������ֵ��Ĭ��: 0.3��
- `--text_threshold`: GroundingDINO�ı���ֵ��Ĭ��: 0.25��

#### ��������
- `--enable_scene_classification`: ���ó�������
- `--static_threshold`: ��̬������ֵ��Ĭ��: 0.1��
- `--low_dynamic_threshold`: �Ͷ�̬������ֵ��Ĭ��: 0.3��
- `--medium_dynamic_threshold`: �еȶ�̬������ֵ��Ĭ��: 0.6��
- `--high_dynamic_threshold`: �߶�̬������ֵ��Ĭ��: 1.0��
- `--motion_ratio_threshold`: �˶�������ֵ��Ĭ��: 1.5��

#### ����ѡ��
- `--no_subject_diag_norm`: ��������Խ��߹�һ��

## �����ʽ

Ԫ��ϢJSON�ļ�Ӧ������Ƶ��Ϣ�б�

```json
[
  {
    "index": 0,
    "filepath": "data/video1.mp4",
    "subject_noun": "person",
    "prompt": "A person walking"
  },
  {
    "index": 1,
    "filepath": "data/video2.mp4",
    "subject_noun": "dog",
    "prompt": "A dog running"
  }
]
```

## �����ʽ

�ű�����ÿ����Ƶ��Ԫ��Ϣ����� `perceptible_amplitude_score` �ֶΡ�

### �ɹ�����

```json
{
  "index": 0,
  "filepath": "data/video1.mp4",
  "subject_noun": "person",
  "prompt": "A person walking",
  "perceptible_amplitude_score": {
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
}
```

### ������

```json
{
  "index": 1,
  "filepath": "data/video2.mp4",
  "subject_noun": "dog",
  "prompt": "A dog running",
  "perceptible_amplitude_score": {
    "status": "error",
    "error_reason": "no_subject_detected",
    "background_motion": 0.0234
  }
}
```

## ʾ��

### ����ʹ��

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json
```

### ���ó�������

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json \
    --enable_scene_classification \
    --static_threshold 0.1 \
    --low_dynamic_threshold 0.3
```

### ʹ��CPU

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json \
    --device cpu
```

## ע������

1. �״�������Ҫ����ģ���ļ��� `.cache` Ŀ¼
2. ����ʹ��GPU���٣�`--device cuda`��
3. Ԫ��Ϣ�ļ��е� `subject_noun` Ӧ����Ƶ��ʵ������ƥ��
4. ������Զ�д��Ԫ��Ϣ�ļ���Ĭ�ϸ���ԭ�ļ���

