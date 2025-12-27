#!/bin/bash
# Docker 环境变量配置示例脚本
# 用于设置 navsim 导入所需的环境变量

# 设置 DiffusionDrive 根目录路径
export DIFFUSIONDRIVE_ROOT=/data2/file_swap/sh_space/DiffusionDrive

# 设置 PYTHONPATH，将 DiffusionDrive 目录添加到 Python 搜索路径
export PYTHONPATH=${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH}

# 设置 NAVSIM_DEVKIT_ROOT（脚本中使用的环境变量）
export NAVSIM_DEVKIT_ROOT=${DIFFUSIONDRIVE_ROOT}

# 验证环境变量设置
echo "PYTHONPATH: $PYTHONPATH"
echo "NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"

# 测试 navsim 导入
python -c "import navsim; print('✓ navsim 导入成功！')" 2>&1 || echo "✗ navsim 导入失败，请检查路径设置"


