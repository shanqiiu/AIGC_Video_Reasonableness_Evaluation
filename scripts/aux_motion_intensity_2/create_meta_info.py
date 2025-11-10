import json
import os
import argparse

def create_new_json(json_data, video_folder):
    """根据已有视频文件补全元数据中的绝对路径。"""
    new_json_data = []
    for item in json_data:
        index = item['index']
        video_filename = f"{index}.mp4"
        video_path = os.path.join(video_folder, video_filename)
        print(video_path)
        if os.path.exists(video_path):
            print(video_path)
            new_item = item.copy()
            new_item['filepath'] = os.path.abspath(video_path)
            new_json_data.append(new_item)
    return new_json_data

def main():
    parser = argparse.ArgumentParser(description="为现有视频生成包含绝对路径的新 JSON 文件")
    parser.add_argument("-i", "--input_json", default="./prompts/prompts.json", help="输入 JSON 文件路径")
    parser.add_argument("-v", "--video_folder", required=True, help="视频文件所在文件夹路径")
    parser.add_argument("-o", "--output_json", required=True, help="输出 JSON 文件保存路径")
    
    args = parser.parse_args()

    video_folder = os.path.abspath(args.video_folder)

    # 读取入口 JSON 文件
    with open(args.input_json, 'r') as f:
        json_data = json.load(f)

    # 为存在的本地视频生成新的元数据
    new_json_data = create_new_json(json_data, video_folder)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # 将补全后的元数据写入目标文件
    with open(args.output_json, 'w') as f:
        json.dump(new_json_data, f, indent=2)

    print(f"New JSON with {len(new_json_data)} entries saved to {args.output_json}")

if __name__ == "__main__":
    main()