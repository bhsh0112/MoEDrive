#!/bin/bash
# Docker 容器和镜像丢失分析脚本

echo "========================================="
echo "Docker 容器和镜像丢失分析报告"
echo "========================================="
echo ""

echo "1. 当前 Docker 系统状态："
echo "------------------------"
docker system df
echo ""

echo "2. 当前所有镜像："
echo "------------------------"
docker images -a
echo ""

echo "3. 当前所有容器："
echo "------------------------"
docker ps -a
echo ""

echo "4. 检查 Docker 日志中的删除操作（最近7天）："
echo "------------------------"
journalctl -u docker.service --since "7 days ago" --no-pager | grep -iE "(rm|delete|remove|prune|clean)" | tail -50 || echo "无法访问 journalctl"
echo ""

echo "5. 检查系统日志中的 Docker 删除操作："
echo "------------------------"
grep -i "docker" /var/log/syslog 2>/dev/null | grep -E "(rm|delete|remove|prune|clean)" | tail -50 || echo "无法访问系统日志"
echo ""

echo "6. 检查 Docker 数据目录使用情况："
echo "------------------------"
df -h /var/lib/docker 2>/dev/null || df -h /
echo ""

echo "7. 检查是否有定时清理任务："
echo "------------------------"
crontab -l 2>/dev/null | grep -i docker || echo "未找到相关定时任务"
systemctl list-timers 2>/dev/null | grep -i docker || echo "未找到相关 systemd 定时器"
echo ""

echo "8. 检查 Docker 服务状态："
echo "------------------------"
systemctl status docker.service --no-pager -l | head -20 || echo "无法检查服务状态"
echo ""

echo "9. 检查最近的 shell 历史记录（可能包含删除命令）："
echo "------------------------"
if [ -f ~/.bash_history ]; then
    grep -iE "docker.*(rm|prune|clean|system)" ~/.bash_history | tail -20
else
    echo "无法访问 bash 历史记录"
fi
echo ""

echo "========================================="
echo "分析完成"
echo "========================================="

