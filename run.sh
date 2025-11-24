#!/bin/bash

# 图书推荐系统 - 一键运行脚本
# 版本: v2.0
# 使用当前激活的conda环境

set -e  # 遇到错误立即退出

echo "======================================"
echo "  图书推荐系统 - 完整运行流程"
echo "======================================"

# 检查conda环境
echo ""
echo "[1/6] 检查Python环境..."
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到激活的conda环境"
    echo "请先激活conda环境: conda activate <your_env_name>"
    exit 1
fi
echo "✓ 当前环境: $CONDA_DEFAULT_ENV"

# 检查Python环境
echo ""
echo "[2/6] 检查Python环境..."
python --version || { echo "错误: 未找到Python"; exit 1; }

# 检查依赖
echo ""
echo "[3/6] 检查依赖包..."
python -c "import pandas, numpy, sklearn, lightfm" 2>/dev/null || {
    echo "依赖缺失，正在安装到Aigc环境..."
    pip install pandas numpy scikit-learn lightfm matplotlib
}
echo "✓ 依赖检查完成"

# 检查数据文件
echo ""
echo "[4/6] 检查数据文件..."
if [ ! -f "data/item.csv" ] || [ ! -f "data/user.csv" ] || [ ! -f "data/inter_final_选手可见.csv" ]; then
    echo "错误: 数据文件缺失，请确保以下文件存在："
    echo "  - data/item.csv"
    echo "  - data/user.csv"
    echo "  - data/inter_final_选手可见.csv"
    echo "  - data/inter_reevaluation.csv (可选)"
    exit 1
fi
echo "✓ 数据文件检查完成"

# 训练LightFM模型
echo ""
echo "[5/6] 训练LightFM模型..."
echo "预计耗时: 3-5分钟"
python code/lightfm_pipeline.py
echo "✓ 模型训练完成"

# 生成推荐结果
echo ""
echo "[6/6] 生成推荐结果..."
echo "预计耗时: 1-2分钟"
python code/juesai.py
echo "✓ 推荐生成完成"

# 显示结果摘要
echo ""
echo "======================================"
echo "  运行完成！"
echo "======================================"
echo ""
echo "结果文件: output/submission_人民当家作组.csv"
echo ""
echo "结果统计:"
python -c "
import pandas as pd
df = pd.read_csv('output/submission_人民当家作组.csv')
print(f'  推荐用户数: {len(df)}')
print(f'  唯一书籍数: {df[\"book_id\"].nunique()}')
print(f'  覆盖率: {df[\"book_id\"].nunique()/len(df)*100:.2f}%')
print(f'  Top-1书籍: {df[\"book_id\"].value_counts().iloc[0]} 次 ({df[\"book_id\"].value_counts().iloc[0]/len(df)*100:.1f}%)')
"

echo ""
echo "查看前10条结果:"
head -11 output/submission_人民当家作组.csv

echo ""
echo "详细技术报告: docs/技术报告.md"
echo "======================================"
