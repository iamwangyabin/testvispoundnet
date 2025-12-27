# Arrow数据集批处理使用说明

## 概述

`batch_process_arrow.py` 脚本专门为处理与PoundNet `test.py`兼容的Arrow格式数据集而设计。它使用与测试脚本完全相同的数据加载方式和配置文件。

## Arrow数据集格式

### 数据集结构
```
data/
├── DiffusionForensics/
│   ├── dataset.arrow           # Arrow格式数据集文件
│   ├── train.json             # 训练集分割定义
│   ├── test.json              # 测试集分割定义  
│   └── mapping.json           # 路径到索引的映射
├── ForenSynths/
│   ├── dataset.arrow
│   ├── test.json
│   └── mapping.json
├── Ojha/
└── DIF/
```

### 配置文件兼容性
脚本直接使用PoundNet的配置文件（如`cfgs/poundnet.yaml`），支持所有在`datasets.source`中定义的数据集：

```yaml
datasets:
  source:
    - target: data.ArrowDatasets
      data_root: '${datasets.base_path}/DiffusionForensics'
      sub_sets: ['lsun_bedroom_adm', 'lsun_bedroom_ddpm', ...]
      split: 'test'
      benchmark_name: 'DiffusionForensics'
    
    - target: data.ArrowDatasets
      data_root: '${datasets.base_path}/ForenSynths'
      sub_sets: ['progan', 'stylegan', 'stylegan2', ...]
      split: 'test'
      benchmark_name: 'ForenSynths'
```

## 使用方法

### 1. 基本用法 - 处理所有数据集
```bash
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/full_analysis/
```

### 2. 处理特定数据集
```bash
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/diffusion_only/ \
    --dataset_filter DiffusionForensics
```

### 3. 处理特定子集
```bash
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/gan_analysis/ \
    --subset_filter progan stylegan stylegan2
```

### 4. 限制每个子集的图片数量
```bash
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/sample_analysis/ \
    --max_images_per_subset 50 \
    --save_cams
```

### 5. 使用自定义检查点
```bash
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --checkpoint /path/to/custom_model.ckpt \
    --output_dir results/custom_model/
```

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | PoundNet配置文件路径（必需） | `cfgs/poundnet.yaml` |
| `--checkpoint` | 自定义模型检查点路径 | `/path/to/model.ckpt` |
| `--output_dir` | 输出目录（必需） | `results/analysis/` |
| `--target_layers` | 指定分析的层 | `image_encoder.transformer.resblocks.23.ln_2` |
| `--max_images_per_subset` | 每个子集最大图片数 | `100` |
| `--save_cams` | 保存CAM数组文件 | - |
| `--dataset_filter` | 过滤特定数据集 | `DiffusionForensics ForenSynths` |
| `--subset_filter` | 过滤特定子集 | `progan stylegan` |

## 输出结构

```
results/
├── arrow_dataset_results.json      # 完整的JSON结果
├── arrow_dataset_summary.csv       # CSV格式摘要
├── arrow_dataset_statistics.txt    # 文本统计信息
├── DiffusionForensics/             # 按数据集组织的结果
│   ├── lsun_bedroom_adm/
│   │   ├── image_000001_cam.npy    # CAM数组（如果启用--save_cams）
│   │   ├── image_000002_cam.npy
│   │   └── ...
│   └── lsun_bedroom_ddpm/
│       └── ...
└── ForenSynths/
    ├── progan/
    └── stylegan/
```

## 输出文件说明

### 1. arrow_dataset_results.json
包含所有图片的详细分析结果：
```json
{
  "results": [
    {
      "image_index": 0,
      "true_label": 1,
      "subset_name": "DiffusionForensics_lsun_bedroom_adm",
      "predicted_class_name": "Fake",
      "confidence": 0.9234,
      "real_prob": 0.0766,
      "fake_prob": 0.9234,
      "predicted_correct": true
    }
  ],
  "dataset_summaries": {...},
  "overall_statistics": {...}
}
```

### 2. arrow_dataset_summary.csv
便于Excel分析的表格格式：
```csv
subset_name,image_index,true_label,predicted_class,confidence,real_prob,fake_prob,correct
DiffusionForensics_lsun_bedroom_adm,0,1,Fake,0.9234,0.0766,0.9234,True
```

### 3. arrow_dataset_statistics.txt
统计摘要：
```
ARROW DATASET GRADCAM ANALYSIS SUMMARY
======================================

Overall Statistics:
  Total processed: 1000
  Successful: 995
  Overall accuracy: 0.8945
  Number of datasets: 4

Class Distribution:
  Real: 450 (45.2%)
  Fake: 545 (54.8%)

Per-Dataset Accuracy:
  DiffusionForensics_lsun_bedroom_adm: 0.9200
  ForenSynths_progan: 0.8800
  ...
```

## 与test.py的兼容性

### 相同点
- 使用相同的配置文件格式
- 使用相同的数据加载器（`ArrowDatasets`）
- 使用相同的数据预处理管道
- 支持相同的数据集和子集

### 不同点
- 批处理脚本逐张处理图片以生成GradCAM
- 提供额外的可视化和分析功能
- 支持过滤特定数据集或子集
- 生成详细的统计报告

## 性能考虑

### 处理速度
- 单张图片：约1-3秒（取决于GPU）
- 建议使用`--max_images_per_subset`限制处理数量进行快速测试

### 内存使用
- 每张图片的CAM约200KB
- 大数据集建议不保存CAM数组（不使用`--save_cams`）

### GPU要求
- 推荐：8GB+ VRAM
- 最低：4GB VRAM
- 支持CPU处理（较慢）

## 示例工作流

### 1. 快速测试
```bash
# 每个子集只处理10张图片进行快速测试
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/quick_test/ \
    --max_images_per_subset 10
```

### 2. 特定数据集深度分析
```bash
# 只分析DiffusionForensics数据集并保存CAM
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/diffusion_deep/ \
    --dataset_filter DiffusionForensics \
    --save_cams
```

### 3. GAN生成图片分析
```bash
# 只分析GAN生成的图片
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/gan_analysis/ \
    --subset_filter progan stylegan stylegan2 biggan
```

## 故障排除

### 常见问题

1. **"数据集路径不存在"**
   - 检查配置文件中的`base_path`设置
   - 确保Arrow数据集文件存在

2. **"CUDA内存不足"**
   - 减少`max_images_per_subset`
   - 不使用`--save_cams`选项

3. **"找不到检查点文件"**
   - 检查配置文件中的`resume.path`
   - 或使用`--checkpoint`指定正确路径

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=.
python -u visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/debug/ \
    --max_images_per_subset 5
```

## 总结

Arrow数据集批处理功能提供了与PoundNet测试管道完全兼容的GradCAM分析能力，支持：

- ✅ 与`test.py`相同的数据集格式和配置
- ✅ 灵活的数据集和子集过滤
- ✅ 详细的统计分析和可视化
- ✅ 高效的批处理和内存管理
- ✅ 多种输出格式支持

这使得研究人员可以轻松地对PoundNet在各种数据集上的注意力模式进行深入分析。