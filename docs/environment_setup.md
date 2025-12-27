# 环境变量配置指南

## 问题描述
即使有 `navsim` 目录，Python 也无法导入 `navsim` 模块，因为该目录不在 Python 的搜索路径中。

## 解决方案

### 方法 1: 设置 PYTHONPATH 环境变量（推荐）

将包含 `navsim` 的目录添加到 `PYTHONPATH` 环境变量中：

```bash
# 临时设置（当前终端会话有效）
export PYTHONPATH=/data2/file_swap/sh_space/DiffusionDrive:$PYTHONPATH

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export PYTHONPATH=/data2/file_swap/sh_space/DiffusionDrive:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### 方法 2: 设置 NAVSIM_DEVKIT_ROOT 环境变量

脚本中使用了 `$NAVSIM_DEVKIT_ROOT` 环境变量，也需要设置：

```bash
# 临时设置
export NAVSIM_DEVKIT_ROOT=/data2/file_swap/sh_space/DiffusionDrive

# 永久设置
echo 'export NAVSIM_DEVKIT_ROOT=/data2/file_swap/sh_space/DiffusionDrive' >> ~/.bashrc
source ~/.bashrc
```

### 方法 3: 在 Docker 中设置环境变量

如果使用 Docker，需要在 Dockerfile 或 docker-compose.yml 中设置：

**在 Dockerfile 中：**
```dockerfile
# 设置 PYTHONPATH
ENV PYTHONPATH=/data2/file_swap/sh_space/DiffusionDrive:$PYTHONPATH

# 设置 NAVSIM_DEVKIT_ROOT
ENV NAVSIM_DEVKIT_ROOT=/data2/file_swap/sh_space/DiffusionDrive
```

**在 docker-compose.yml 中：**
```yaml
services:
  your_service:
    environment:
      - PYTHONPATH=/data2/file_swap/sh_space/DiffusionDrive:${PYTHONPATH}
      - NAVSIM_DEVKIT_ROOT=/data2/file_swap/sh_space/DiffusionDrive
```

**在 docker run 命令中：**
```bash
docker run -e PYTHONPATH=/data2/file_swap/sh_space/DiffusionDrive:$PYTHONPATH \
           -e NAVSIM_DEVKIT_ROOT=/data2/file_swap/sh_space/DiffusionDrive \
           your_image
```

## 验证设置

设置完成后，可以通过以下方式验证：

```bash
# 检查环境变量
echo $PYTHONPATH
echo $NAVSIM_DEVKIT_ROOT

# 在 Python 中测试导入
python -c "import navsim; print('navsim 导入成功！')"
```

## 注意事项

1. 确保 `navsim` 目录位于 `/data2/file_swap/sh_space/DiffusionDrive/navsim`
2. 如果路径不同，请相应调整环境变量的值
3. 在 Docker 容器中，确保路径映射正确（使用 `-v` 参数挂载目录）
4. 建议同时设置 `PYTHONPATH` 和 `NAVSIM_DEVKIT_ROOT` 两个环境变量


