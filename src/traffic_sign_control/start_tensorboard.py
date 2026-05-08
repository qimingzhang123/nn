"""
启动TensorBoard可视化工具
监控YOLO模型训练过程
"""
import os
import subprocess
import argparse

def find_latest_logdir(base_dir='runs/detect'):
    """查找最新的训练日志目录"""
    if not os.path.exists(base_dir):
        return None

    latest_dir = None
    latest_time = 0

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith('events.'):
                mtime = os.path.getmtime(os.path.join(root, f))
                if mtime > latest_time:
                    latest_time = mtime
                    # 找到包含events文件的目录
                    latest_dir = root

    return latest_dir


def start_tensorboard(logdir='runs/detect'):
    """
    启动TensorBoard

    Args:
        logdir: TensorBoard日志目录
    """
    # 自动查找最新的日志目录
    auto_dir = find_latest_logdir(logdir)
    if auto_dir:
        logdir = auto_dir
        print(f"自动找到最新日志目录: {logdir}")
    elif not os.path.exists(logdir):
        print(f"错误: 日志目录不存在: {logdir}")
        print("\n请先运行训练:")
        print("  python train.py --epochs 50")
        return

    print(f"\n{'='*50}")
    print(f"启动TensorBoard...")
    print(f"日志目录: {logdir}")
    print(f"访问地址: http://localhost:6006")
    print(f"{'='*50}")
    print(f"按 Ctrl+C 停止\n")

    try:
        subprocess.run(['tensorboard', '--logdir', logdir, '--port', '6006'])
    except KeyboardInterrupt:
        print("\nTensorBoard已停止")
    except FileNotFoundError:
        print("错误: TensorBoard未安装")
        print("安装命令: pip install tensorboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动TensorBoard监控训练")
    parser.add_argument('--logdir', type=str, default='runs/detect',
                       help='TensorBoard日志目录 (默认: runs/detect)')

    args = parser.parse_args()
    start_tensorboard(args.logdir)
