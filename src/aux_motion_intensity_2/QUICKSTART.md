# ���ٿ�ʼ

## 1. ����׼��

ȷ���Ѱ�װ��Ҫ��������

```bash
pip install torch torchvision
pip install opencv-python pillow numpy
pip install tqdm
pip install transformers  # for BERT
```

## 2. ����ģ��

����Ŀ��Ŀ¼���� `.cache` Ŀ¼������������ģ�ͣ�

```bash
mkdir -p .cache

# ���� GroundingDINO
wget -O .cache/groundingdino_swinb_cogcoor.pth \
  https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth

# ���� SAM
wget -O .cache/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ���� Co-Tracker
wget -O .cache/scaled_offline.pth \
  https://huggingface.co/facebook/cotracker/resolve/main/cotracker_scaled_offline.pth
```

## 3. ׼����������

����Ԫ��ϢJSON�ļ� `meta_info.json`��

```json
[
  {
    "index": 0,
    "filepath": "data/videos/video1.mp4",
    "subject_noun": "person",
    "prompt": "A person walking in the park"
  },
  {
    "index": 1,
    "filepath": "data/videos/video2.mp4",
    "subject_noun": "dog",
    "prompt": "A dog running"
  }
]
```

## 4. ���з���

### ��ʽA��ʹ�ýű�

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path meta_info.json \
    --output_path results/pas_results.json \
    --enable_scene_classification \
    --device cuda
```

### ��ʽB��Python����

```python
import json
from src.aux_motion_intensity_2 import PASAnalyzer

# ��ʼ��������
analyzer = PASAnalyzer(
    device="cuda",
    enable_scene_classification=True
)

# ����Ԫ��Ϣ
with open("meta_info.json", "r") as f:
    meta_infos = json.load(f)

# ������Ƶ
for meta_info in meta_infos:
    result = analyzer.analyze_video(
        video_path=meta_info["filepath"],
        subject_noun=meta_info["subject_noun"]
    )
    
    # �������ӵ�Ԫ��Ϣ
    meta_info["perceptible_amplitude_score"] = result
    print(f"Video {meta_info['index']}: {result['status']}")
    
    if result["status"] == "success":
        print(f"  Background motion: {result['background_motion']:.4f}")
        print(f"  Subject motion: {result['subject_motion']:.4f}")
        print(f"  Scene type: {result.get('scene_classification', {}).get('scene_type', 'N/A')}")

# ������
with open("results/pas_results.json", "w") as f:
    json.dump(meta_infos, f, indent=2)
```

## 5. �鿴���

�����д��JSON�ļ�������������Ϣ��

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

## ��������

### Q1: ģ������ʧ��

**A**: �����ֶ�����ģ���ļ�����ʹ�ô���ȷ���ļ���������������ļ���С����

### Q2: CUDA out of memory

**A**: ���ԣ�
- ���� `grid_size` ������Ĭ��30��
- ʹ�ø�С�����δ���
- ��ʹ�� CPU ģʽ��`--device cpu`���ٶȽ�����

### Q3: �޷���⵽����

**A**: ��飺
- `subject_noun` �Ƿ�����Ƶ����ƥ��
- ���Ե��� `--box_threshold` �� `--text_threshold`
- ȷ����Ƶ���������ɼ�������

### Q4: �������

**A**: ȷ����
- ��ȷ��װ��������
- ��Ŀ·������ӵ� PYTHONPATH
- ������������ȷ������ `third_party/` Ŀ¼

## ��һ��

- �Ķ� [README.md](README.md) �˽���ϸ����
- �鿴 [INTEGRATION.md](INTEGRATION.md) �˽⼯��ϸ��
- �ο�ʾ�������������Ż����

