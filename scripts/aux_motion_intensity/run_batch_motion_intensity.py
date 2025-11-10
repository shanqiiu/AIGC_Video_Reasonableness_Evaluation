import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.aux_motion_intensity.analyzer import MotionIntensityAnalyzer
from src.aux_motion_intensity.batch import batch_analyze_videos


def main():
    parser = argparse.ArgumentParser(description='Batch motion intensity analysis')
    parser.add_argument('--input', '-i', required=True, help='Input videos directory')
    parser.add_argument('--output', '-o', default='reports', help='Output directory for reports')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--method', default='raft', choices=['farneback', 'tvl1', 'raft'], help='Optical flow method')
    parser.add_argument('--raft_model', default="D:\mycode\AIGC_Video_Reasonableness_Evaluation\.cache\\raft-things.pth", help='RAFT model path (optional)')
    parser.add_argument('--no-camera-comp', action='store_true', help='Disable camera compensation')
    parser.add_argument('--normalize', action='store_true', help='Normalize flow by resolution')
    parser.add_argument('--flow-threshold-ratio', type=float, default=0.002, help='Static threshold ratio when normalized')
    parser.add_argument('--fov', type=float, default=60.0, help='Camera field of view in degrees')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames per video')
    parser.add_argument('--frame-skip', type=int, default=1, help='Frame sampling stride')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    analyzer = MotionIntensityAnalyzer(
        raft_model_path=args.raft_model,
        device=args.device,
        method=args.method,
        enable_camera_compensation=not args.no_camera_comp,
        use_normalized_flow=args.normalize,
        flow_threshold_ratio=args.flow_threshold_ratio,
    )

    batch_analyze_videos(
        analyzer,
        input_dir=args.input,
        output_dir=args.output,
        camera_fov=args.fov,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )


if __name__ == '__main__':
    main()


