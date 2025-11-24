

### 前置要求
- Anaconda/Miniconda
- Python 3.10+ (推荐使用 conda 环境)
- 8GB内存

### 步骤0: 激活 conda 环境

```bash
# 示例：激活您的 conda 环境
conda activate <your_env_name>
```

### 步骤1: 安装依赖（在已激活的 conda 环境 中）

```bash
pip install pandas numpy scikit-learn lightfm matplotlib
```

### 步骤2: 准备数据

确保数据文件在正确位置：
- `data/item.csv` - 图书元数据
- `data/user.csv` - 用户信息
- `data/inter_reevaluation.csv` - 训练数据
此外测试数据位于 `data/inter_final_选手可见.csv`。

### 步骤3: 训练模型（确保已激活目标 conda 环境）

```bash
# 确保在已激活的 conda 环境
conda activate <your_env_name>
python code/lightfm_pipeline.py
```

预期耗时: 3-5分钟

### 步骤4: 生成推荐（确保已激活目标 conda 环境）

```bash
# 确保在已激活的 conda 环境
conda activate <your_env_name>
python code/juesai.py
```

预期耗时: 1-2分钟

### 步骤5: 查看结果

```bash
head output/submission_人民当家作组.csv
```

输出格式:
```csv
user_id,book_id
53,38062
270,38062
...
```

## 📊 预期性能

- **推荐用户数**: 1,451
- **唯一书籍数**: 235
- **覆盖率**: 16.20%
- **处理速度**: ~1分钟

## 📝 详细文档

请查看 `docs/技术报告.md` 了解：
- 完整系统架构
- 注意力机制详解
- 院系协同过滤算法
- 参数调优指南
- 实验结果分析

## ⚙️ 核心参数

在 `code/juesai.py` 中可调整：

```python
LIGHTFM_WEIGHT = 0.45           # LightFM权重
ATTENTION_TEMPERATURE = 0.05    # 注意力温度
DEPT_COLLABORATIVE_BOOST = 0.3  # 院系协同加成
DEPT_AFFINITY_BOOST = 0.25      # 院系亲和度加成
```

## 🐛 常见问题

**Q: 找不到数据文件**
```bash
# 检查数据文件是否存在
ls -lh data/
```

**Q: 模型加载失败**
```bash
# 重新训练模型
python code/lightfm_pipeline.py
```

**Q: 推荐结果为空**
```bash
# 检查测试数据路径
# 确认 `data/inter_final_选手可见.csv` 存在
```

## 📧 技术支持

详细文档请查看 `docs/技术报告.md`

---

**版本**: v2.0 (注意力机制+院系协同优化版)  
**更新**: 2025-11-21
