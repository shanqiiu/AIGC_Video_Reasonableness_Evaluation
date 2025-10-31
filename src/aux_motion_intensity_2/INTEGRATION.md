# ����˵��

## ����

��ģ��� `VMBench_diy/perceptible_amplitude_score.py` �ع��������Ѽ��ɵ� `AIGC_Video_Reasonableness_Evaluation` ��Ŀ�С�

## Ŀ¼�ṹ

```
src/aux_motion_intensity_2/
������ __init__.py                    # ģ�����
������ analyzer.py                    # ����������PASAnalyzer��
������ scene_classifier.py            # ����������
������ motion_calculator.py          # �˶����㹤��
������ batch.py                       # ��������ӿ�
������ README.md                      # ʹ���ĵ�
������ INTEGRATION.md                # ���ĵ�

scripts/aux_motion_intensity_2/
������ run_pas.py                     # �����ű�
������ README.md                      # �ű�ʹ��˵��
```

## ��Ҫ�Ķ�

### 1. ģ�黯�ع�

ԭ�ű� `perceptible_amplitude_score.py` (819��) �����Ϊ�����ģ�飺

- **analyzer.py**: ���ķ����߼������� `PASAnalyzer` ��
- **scene_classifier.py**: ���������߼���SceneClassifier��
- **motion_calculator.py**: �˶����ȼ��㺯��
- **batch.py**: ��������ӿ�

### 2. ·������

��������·���ѵ���Ϊָ����Ŀ�ڵ� `third_party` Ŀ¼��

```python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
gsa_path = os.path.join(project_root, "third_party", "Grounded-Segment-Anything")
```

**˵��**���� `src/aux_motion_intensity_2/analyzer.py` ����������`../..`��������Ŀ��Ŀ¼��

### 3. ���뷽ʽ����

ԭ�ű��еĵ��뷽ʽ��
```python
import GroundingDINO.groundingdino.datasets.transforms as T
```

����Ϊ��
```python
import groundingdino.datasets.transforms as T
```

### 4. ����·������

ģ�������ļ�·�������·������Ϊ����·����
```python
self.config_file = os.path.join(
    gsa_path,
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinB.py"
)
```

### 5. API��

ԭ�ű������������ӻ�������`visualize_detection`, `visualize_masks`, `visualize_tracks` �ȣ���
���°汾�б����򣬱������Ĺ��ܡ�������ӻ����ɲο�ԭ�ű���

### 6. �ӳټ���

�����ӳټ��ز��ԣ�ģ�����״ε��� `analyze_video` ʱ���أ����ٳ�ʼ��������

## ʹ�öԱ�

### ԭ�汾ʹ�÷�ʽ

```bash
python VMBench_diy/perceptible_amplitude_score.py \
    --meta_info_path meta_info.json \
    --device cuda \
    --enable_scene_classification
```

### �°汾ʹ�÷�ʽ

#### ��ʽ1��ʹ�ýű�

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path meta_info.json \
    --device cuda \
    --enable_scene_classification
```

#### ��ʽ2��ֱ�ӵ���

```python
from src.aux_motion_intensity_2 import PASAnalyzer

analyzer = PASAnalyzer(device="cuda", enable_scene_classification=True)
result = analyzer.analyze_video("video.mp4", subject_noun="person")
```

## ������ϵ

```
PASAnalyzer
������ GroundingDINO (third_party/Grounded-Segment-Anything/GroundingDINO)
������ SAM (third_party/Grounded-Segment-Anything/segment_anything)
������ Co-Tracker (third_party/co-tracker)
������ SceneClassifier (����)
������ motion_calculator (����)
```

## ģ������

����ģ���ļ�Ӧ���� `.cache/` Ŀ¼�£�

- `groundingdino_swinb_cogcoor.pth`
- `sam_vit_h_4b8939.pth`
- `scaled_offline.pth`
- `google-bert/bert-base-uncased/`

## ��ԭ��Ŀ�Ĺ�ϵ

��ģ����Ϊ `aux_motion_intensity_2` �����е� `aux_motion_intensity` ģ�鲢�д��ڣ�
���ṩ���� Grounded-SAM + Co-Tracker ����һ�ֶ�̬�̶ȷ���������

- **aux_motion_intensity**: ʹ�� RAFT ��������
- **aux_motion_intensity_2**: ʹ�� Grounded-SAM + Co-Tracker ����

���߿��Բ������У��ṩ�������˶����������

## ���Խ���

1. ��������Ԫ��Ϣ�ļ�
2. ���нű���֤�����ģ�ͼ���
3. ����������Ƿ����Ԥ��
4. �Ա���ԭ�汾�����һ����

## ��֪����

1. ���ӻ�����δ��ȫǨ�ƣ��ɰ�����ӣ�
2. ģ���ļ���Ҫ�ֶ�����
3. �״�������Ҫ�ϳ�ʱ�����ģ��

## �����Ż�

- [ ] ���ģ���Զ����ؽű�
- [ ] �ָ����ӻ����ܣ���ѡ��
- [ ] ��ӵ�Ԫ����
- [ ] �����Ż�����������GPU֧�ֵȣ�

