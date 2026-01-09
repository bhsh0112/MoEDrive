#!/bin/bash
# 清理损坏的缓存文件

source scripts/SetPath.sh

CACHE_DIR="${NAVSIM_EXP_ROOT}/training_cache"

if [ ! -d "$CACHE_DIR" ]; then
    echo "缓存目录不存在: $CACHE_DIR"
    exit 0
fi

echo "检查缓存目录: $CACHE_DIR"
echo "找到的缓存文件数量: $(find "$CACHE_DIR" -name "*.pkl.gz" 2>/dev/null | wc -l)"

read -p "是否删除所有缓存文件并重新生成? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在删除缓存文件..."
    find "$CACHE_DIR" -name "*.pkl.gz" -type f -delete 2>/dev/null
    echo "缓存文件已删除。下次训练时会重新生成。"
    echo "建议在 train.sh 中设置 force_cache_computation=True"
else
    echo "已取消操作。"
fi
