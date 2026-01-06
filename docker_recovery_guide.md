# Docker 容器和镜像丢失分析与恢复方案

## 一、问题分析

根据您提供的日志信息，我们观察到以下情况：

### 1. 日志分析
从系统日志中看到的 `/tasks/delete` 事件实际上是正常的容器任务删除事件。这些 "ignoring event" 消息表示 Docker 守护进程正在忽略某些任务删除事件，这通常发生在：
- 容器正常停止时
- 容器被手动删除时
- Docker 守护进程重启后清理残留任务时

### 2. 可能的原因

#### 原因1：手动删除操作
- 执行了 `docker rm` 或 `docker rmi` 命令
- 执行了 `docker system prune` 或 `docker system prune -a` 命令
- 执行了 `docker image prune` 或 `docker container prune` 命令

#### 原因2：磁盘空间不足自动清理
- Docker 可能在磁盘空间不足时自动清理未使用的资源
- 检查磁盘使用情况：`df -h /var/lib/docker`

#### 原因3：定时清理任务
- 可能配置了 crontab 或其他定时任务自动清理 Docker 资源
- 检查：`crontab -l` 和 `systemctl list-timers`

#### 原因4：Docker 守护进程问题
- Docker 守护进程崩溃或重启可能导致状态不一致
- 检查 Docker 服务日志：`journalctl -u docker.service -n 100`

#### 原因5：文件系统问题
- `/var/lib/docker` 目录可能被意外删除或损坏
- 磁盘 I/O 错误可能导致数据丢失

## 二、诊断步骤

执行以下命令进行诊断：

```bash
# 1. 检查当前 Docker 资源
docker system df
docker images -a
docker ps -a

# 2. 检查 Docker 日志
sudo journalctl -u docker.service --since "7 days ago" | grep -iE "(rm|delete|remove|prune|clean|error|fail)"

# 3. 检查系统日志
sudo grep -i "docker" /var/log/syslog | grep -E "(rm|delete|remove|prune|clean)" | tail -100

# 4. 检查磁盘空间
df -h /var/lib/docker
du -sh /var/lib/docker/*

# 5. 检查定时任务
crontab -l
sudo systemctl list-timers

# 6. 检查命令历史
history | grep -iE "docker.*(rm|prune|clean)"
cat ~/.bash_history | grep -iE "docker.*(rm|prune|clean)"
```

## 三、恢复方案

### 方案1：从备份恢复（推荐）

如果您有 Docker 数据的备份：

```bash
# 停止 Docker 服务
sudo systemctl stop docker

# 恢复备份的 /var/lib/docker 目录
sudo cp -r /path/to/backup/docker/* /var/lib/docker/

# 启动 Docker 服务
sudo systemctl start docker
```

### 方案2：从镜像仓库重新拉取

对于已知的镜像，可以从 Docker Hub 或其他仓库重新拉取：

```bash
# 列出需要恢复的镜像
# 根据您的需求，重新拉取镜像
docker pull <image_name>:<tag>

# 例如：
# docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

### 方案3：从容器配置重建（如果容器有 docker-compose.yml）

```bash
# 如果有 docker-compose.yml 文件
cd /path/to/docker-compose/
docker-compose up -d

# 或者从 Dockerfile 重新构建
docker build -t <image_name> .
```

### 方案4：检查未清理的容器层

有时容器已删除但层仍然存在：

```bash
# 查看所有层
docker image ls -a --digests

# 尝试恢复悬空镜像
docker images -f "dangling=true" -q | xargs -r docker tag <source>:<tag> <target>:<tag>
```

### 方案5：从 Docker 日志重建容器配置

```bash
# 查看容器创建历史
sudo journalctl -u docker.service | grep -i "create" | grep -i "container"

# 查找容器配置信息
sudo grep -r "docker run" /var/log/
```

## 四、预防措施

### 1. 定期备份

创建定期备份脚本：

```bash
#!/bin/bash
# docker_backup.sh

BACKUP_DIR="/path/to/backup/docker"
DATE=$(date +%Y%m%d_%H%M%S)

sudo systemctl stop docker
sudo tar -czf "${BACKUP_DIR}/docker_backup_${DATE}.tar.gz" -C /var/lib docker
sudo systemctl start docker

# 保留最近7天的备份
find ${BACKUP_DIR} -name "docker_backup_*.tar.gz" -mtime +7 -delete
```

### 2. 禁用自动清理

```bash
# 在 /etc/docker/daemon.json 中配置
{
  "storage-driver": "overlay2",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

### 3. 使用标签保护重要镜像

```bash
# 给重要镜像添加保护标签
docker tag <image> <image>:protected

# 清理时排除保护标签
docker image prune -a --filter "label!=protected"
```

### 4. 监控磁盘空间

```bash
# 设置磁盘空间监控
# 当 /var/lib/docker 使用率超过 80% 时告警
```

### 5. 记录重要容器命令

保存创建容器的完整命令到文件中：

```bash
# 保存容器启动命令
docker inspect <container_id> | jq '.[0].Config.Cmd' > container_config.json
```

## 五、紧急恢复检查清单

- [ ] 检查是否有备份
- [ ] 查看 Docker 日志找出删除时间点
- [ ] 检查是否有定时清理任务
- [ ] 确认磁盘空间是否充足
- [ ] 检查 Docker 服务是否正常
- [ ] 查看命令历史记录
- [ ] 检查其他用户的操作历史
- [ ] 尝试从镜像仓库恢复
- [ ] 检查是否有 docker-compose 配置文件

## 六、联系支持

如果上述方法都无法恢复，可以考虑：
1. 使用数据恢复工具（如 testdisk, photorec）尝试恢复 /var/lib/docker
2. 检查是否有其他备份位置
3. 联系系统管理员查看更详细的日志

## 注意事项

⚠️ **重要提示：**
- 在尝试恢复前，先停止 Docker 服务避免数据被覆盖
- 恢复操作前请先备份当前状态
- 如果是生产环境，请联系系统管理员协助处理

