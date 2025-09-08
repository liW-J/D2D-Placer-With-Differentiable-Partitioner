# D2D Placement Result Analyzer

## 概述

D2D Placement Result Analyzer 是一个集成的分析工具，用于分析Die-to-Die布局结果。它提供了全面的网络分析、HPWL统计、终端影响分析和可视化功能。

## 功能特性

### 1. 网络分类分析
- **Top Die Only Nets**: 仅在上层die中的网络
- **Bottom Die Only Nets**: 仅在下层die中的网络  
- **Crossing Nets**: 跨层网络（包括终端）

### 2. HPWL统计分析
- 各类型网络的总HPWL和平均HPWL
- 跨层网络在上下两层的HPWL分布
- 终端对HPWL的影响分析

### 3. 详细报告生成
- 文本格式的详细分析报告
- JSON格式的结构化数据
- 可配置的分析选项

### 4. 可视化图表
- 网络分布饼图
- HPWL分布箱线图
- 跨层网络详细分析图
- 终端影响分析图

## 文件结构

```
placer/tools/
├── d2d_result_analyzer.py          # 主要分析接口类
├── d2d_result_analyzer_simple.py   # 简化的placer集成接口
├── d2d_result_analyzer_example.py  # 使用示例
├── analyze_d2d_nets.py             # 基础网络分析
├── detailed_net_analysis.py        # 详细网络分析
├── terminal_impact_analysis.py     # 终端影响分析
└── README_D2D_Analysis.md         # 本文档
```

## 使用方法

### 方法1: 在Placer中自动分析

修改后的D2D placer会在placement完成后自动运行结果分析：

```python
# 在placer的output()方法后会自动调用
d2d_placer.output()
# 自动分析结果
d2d_placer.analyze_results(include_visualizations=True, include_terminal_analysis=True)
```

### 方法2: 手动调用分析

```python
from placer.tools.d2d_result_analyzer_simple import analyze_placer_results

# 分析placer结果
results = analyze_placer_results(
    placer_instance,
    include_visualizations=True,
    include_terminal_analysis=True
)
```

### 方法3: 独立分析工具

```python
from placer.tools.d2d_result_analyzer import D2DResultAnalyzer

# 创建分析器
analyzer = D2DResultAnalyzer(
    benchmark_file="benchmarks/iccad2022/case2_hidden.txt",
    output_file="install/results/case2_hidden/output.txt",
    result_dir="analysis_results"
)

# 运行完整分析
results = analyzer.run_comprehensive_analysis()
```

### 方法4: 快速分析函数

```python
from placer.tools.d2d_result_analyzer import quick_analyze

# 快速分析
results = quick_analyze(
    "benchmarks/iccad2022/case2_hidden.txt",
    "install/results/case2_hidden/output.txt"
)
```

## 配置选项

### 分析选项

- `include_visualizations`: 是否生成可视化图表 (默认: True)
- `include_terminal_analysis`: 是否分析终端影响 (默认: True)
- `save_intermediate`: 是否保存中间结果 (默认: False)

### 输出目录结构

```
result_dir/
├── d2d_net_analysis_results.json           # 基础分析结果
├── detailed_analysis_report.txt             # 详细报告
├── d2d_net_analysis_visualization.png      # 主要可视化图表
├── crossing_net_analysis.png               # 跨层网络分析图
├── terminal_impact_analysis.png            # 终端影响分析图
└── analysis_summary.json                   # 分析摘要
```

## 输出结果说明

### 1. 基础统计信息

```json
{
  "summary": {
    "total_nets": 1000,
    "top_die_only_count": 400,
    "bottom_die_only_count": 350,
    "crossing_nets_count": 250
  }
}
```

### 2. HPWL统计

```json
{
  "hpwl_statistics": {
    "top_die_only": {
      "count": 400,
      "total_hpwl": 1500000,
      "nets": [...]
    },
    "bottom_die_only": {
      "count": 350,
      "total_hpwl": 1200000,
      "nets": [...]
    },
    "crossing": {
      "count": 250,
      "total_hpwl": 2000000,
      "nets": [...]
    }
  }
}
```

### 3. 终端影响分析

```json
{
  "terminal_impact": {
    "positive_impacts": [...],
    "negative_impacts": [...],
    "zero_impacts": [...],
    "total_with_terminal": 2000000,
    "total_without_terminal": 1800000,
    "absolute_impact": 200000,
    "relative_impact": 11.11
  }
}
```

## 错误处理

分析器包含完善的错误处理机制：

1. **文件不存在**: 自动跳过并记录警告
2. **数据格式错误**: 提供详细错误信息
3. **依赖缺失**: 优雅降级，跳过不可用的功能
4. **资源清理**: 自动清理临时文件

## 性能考虑

- **内存使用**: 大型网络可能需要较多内存
- **处理时间**: 可视化生成可能较慢，特别是大量网络
- **并行处理**: 支持异步分析（未来版本）

## 扩展性

系统设计为模块化架构，易于扩展：

1. **新的分析指标**: 在相应的分析模块中添加
2. **新的可视化**: 在visualization模块中添加
3. **新的输出格式**: 在output模块中添加

## 故障排除

### 常见问题

1. **ImportError**: 检查Python路径和依赖安装
2. **文件权限错误**: 确保有写入结果目录的权限
3. **内存不足**: 减少同时分析的网络数量
4. **可视化失败**: 检查matplotlib后端设置

### 调试模式

启用详细日志记录：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 示例代码

完整的使用示例请参考 `d2d_result_analyzer_example.py` 文件。

## 贡献指南

欢迎提交改进建议和bug报告：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用Apache/MIT许可证。
