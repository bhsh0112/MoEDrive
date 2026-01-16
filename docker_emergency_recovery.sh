#!/bin/bash
# Docker 紧急恢复脚本
# 用于快速诊断和尝试恢复丢失的 Docker 资源

set -e

echo "========================================="
echo "Docker 紧急恢复脚本"
echo "========================================="
echo ""

BACKUP_DIR="${HOME}/docker_recovery_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"

echo "1. 检查当前 Docker 状态..."
echo "------------------------"
docker system df 2>&1 | tee "${BACKUP_DIR}/docker_status.txt"
echo ""

echo "2. 导出当前镜像列表..."
echo "------------------------"
docker images -a > "${BACKUP_DIR}/current_images.txt" 2>&1 || true
cat "${BACKUP_DIR}/current_images.txt"
echo ""

echo "3. 导出当前容器列表..."
echo "------------------------"
docker ps -a > "${BACKUP_DIR}/current_containers.txt" 2>&1 || true
cat "${BACKUP_DIR}/current_containers.txt"
echo ""

echo "4. 检查悬空镜像（可能包含未标记的镜像层）..."
echo "------------------------"
DANGLING=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING" ]; then
    echo "发现悬空镜像："
    docker images -f "dangling=true"
    echo "这些可能是已删除容器的残留层，可以尝试恢复"
else
    echo "未发现悬空镜像"
fi
echo ""

echo "5. 检查 Docker 数据目录大小..."
echo "------------------------"
if [ -d "/var/lib/docker" ]; then
    sudo du -sh /var/lib/docker 2>/dev/null || du -sh /var/lib/docker 2>/dev/null || echo "无法检查"
    echo ""
    echo "Docker 子目录大小："
    sudo du -sh /var/lib/docker/* 2>/dev/null | sort -h | tail -10 || echo "无法检查"
else
    echo "警告：/var/lib/docker 目录不存在！"
fi
echo ""

echo "6. 检查磁盘空间..."
echo "------------------------"
df -h | grep -E "(Filesystem|/var|/var/lib)"
echo ""

echo "7. 检查最近的 Docker 操作日志（需要 sudo 权限）..."
echo "------------------------"
echo "尝试从 journalctl 读取日志..."
sudo journalctl -u docker.service --since "7 days ago" --no-pager 2>/dev/null | \
    grep -iE "(rm|delete|remove|prune|clean|error|fail|kill)" | \
    tail -50 > "${BACKUP_DIR}/docker_operations.log" || \
    echo "无法访问 journalctl（可能需要 sudo 权限）"

if [ -f "${BACKUP_DIR}/docker_operations.log" ]; then
    cat "${BACKUP_DIR}/docker_operations.log"
else
    echo "日志文件未生成"
fi
echo ""

echo "8. 检查定时任务..."
echo "------------------------"
echo "用户 crontab:"
crontab -l 2>/dev/null | grep -i docker || echo "无 Docker 相关定时任务"
echo ""

echo "9. 检查命令历史..."
echo "------------------------"
echo "从 bash_history 查找 Docker 删除命令："
if [ -f ~/.bash_history ]; then
    grep -iE "docker.*(rm|prune|clean|system.*prune)" ~/.bash_history | tail -20 || echo "未找到相关命令"
else
    echo "bash_history 文件不存在"
fi
echo ""

echo "10. 尝试查找可恢复的镜像层..."
echo "------------------------"
echo "检查 /var/lib/docker/image/ 目录（需要 sudo）..."
if [ -d "/var/lib/docker/image" ]; then
    sudo ls -lah /var/lib/docker/image/*/imagedb/content/sha256/ 2>/dev/null | \
        head -20 || echo "无法访问（需要 sudo 权限）"
else
    echo "镜像数据库目录不存在"
fi
echo ""

echo "========================================="
echo "诊断信息已保存到: ${BACKUP_DIR}"
echo "========================================="
echo ""
echo "下一步建议："
echo "1. 查看 ${BACKUP_DIR}/docker_operations.log 找出删除操作的时间点"
echo "2. 检查是否有备份可以恢复"
echo "3. 如果有 docker-compose.yml，尝试重新创建容器"
echo "4. 从 Docker Hub 或其他镜像仓库重新拉取需要的镜像"
echo ""
echo "恢复方案请参考: docker_recovery_guide.md"



