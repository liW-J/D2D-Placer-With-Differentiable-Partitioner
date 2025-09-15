#!/usr/bin/env python3
"""
将 .gp.pl 格式的布局结果转换为 TritonPart 要求的 placement_file 格式

使用方法:
    python3 gp_pl_to_embedding.py input.gp.pl [output.embedding.dat]

如果不指定输出文件名，会自动生成同名的 .embedding.dat 文件
"""

import sys
import os
from typing import List, Tuple


def convert_gp_pl_to_embedding(input_file: str, output_file: str = None):
    """
    将 .gp.pl 文件转换为 .embedding.dat 文件
    
    Args:
        input_file: 输入的 .gp.pl 文件路径
        output_file: 输出的 .embedding.dat 文件路径（可选）
    """

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return False

    # 确定输出文件路径
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.embedding.dat"

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    coordinates = []

    # 读取并解析 .gp.pl 文件
    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # 跳过空行和头行
                if not line or line.startswith('UCLA'):
                    continue

                # 解析格式：节点名 x坐标 y坐标 : FS
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coordinates.append((x, y))
                    except ValueError:
                        print(f"警告：第 {line_num} 行解析失败: {line}")
                        continue
                else:
                    print(f"警告：第 {line_num} 行格式不正确: {line}")
                    continue
    except Exception as e:
        print(f"错误：读取文件时发生异常: {e}")
        return False

    print(f"成功解析 {len(coordinates)} 个节点")

    if not coordinates:
        print("错误：没有找到有效的坐标数据")
        return False

    # 找到坐标范围
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    print(f"坐标范围：")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")

    # 归一化坐标到 [0, 1] 范围
    normalized_coords = []
    for x, y in coordinates:
        norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
        norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        normalized_coords.append((norm_x, norm_y))

    # 写入输出文件
    try:
        with open(output_file, 'w') as f:
            for x, y in normalized_coords:
                f.write(f"{x:.15e}, {y:.15e}\n")

        print(f"成功写入 {len(normalized_coords)} 个归一化坐标到 {output_file}")
        return True

    except Exception as e:
        print(f"错误：写入文件时发生异常: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(
            "使用方法: python3 gp_pl_to_embedding.py input.gp.pl [output.embedding.dat]"
        )
        print("示例: python3 gp_pl_to_embedding.py flattened-2d.gp.pl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_gp_pl_to_embedding(input_file, output_file)

    if success:
        print("转换完成！")
    else:
        print("转换失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
