# 准备环境
1. 使用最新的Docker环境

本项目使用Docker Desktop,还可以直接使用Docker Enginee

- 保证gnome-terminal
```
sudo apt install gnome-terminal
```
- 安装Docker的其他依赖库,注意下面的安装可能会有网络问题，如果出现网络问题，可以给apt增加一个代理。如：
```
sudo apt -o Acquire::http::proxy="http://127.0.0.1:12333" update
```

```
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

- 下载安装包
```
wget https://desktop.docker.com/linux/main/amd64/149282/docker-desktop-4.30.0-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64&_gl=1*1s8tvji*_ga*ODMxMjAyNS4xNzE3ODU3MTgw*_ga_XJWPQMJYHQ*MTcxODExNjQxOC42LjEuMTcxODExNjUyMi42MC4wLjA.
```

- 安装
```
sudo apt-get install ./docker-desktop-<version>-<arch>.deb
```

- 打开
```
systemctl --user start docker-desktop
```

2. 使用最新的算能镜像
```
docker run --privileged --name bm1688 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

## 注意：一下操作在docker里面
3. 下载算能SDK

```
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/24/05/27/11/SDK-23.09_LTS_SP2.zip
unzip SDK-23.09_LTS_SP2.zip
```

4. 配置TPU-Mlir环境

```
cd SDK-23.09_LTS_SP2/SDK-23.09_LTS_SP2/
cd tpu-mlir_20231116_054500/
tar -xvf tpu-mlir_v1.3.140-g3180ff37-20231116.tar.gz
cd tpu-mlir_v1.3.140-g3180ff37-20231116.tar.gz
source envsetup.sh
```

还可以通过如下的方式进行配置：
```
pip install tpu_mlir
pip install tpu_mlir[all]
```

如果无法进行在线安装，可通过如下方式安装：
```
pip install tpu_mlir-*-py3-none-any.whl
pip install tpu_mlir-*-py3-none-any.whl[all]
```

5. 安装libsophon

```
cd SDK-23.09_LTS_SP2/SDK-23.09_LTS_SP2/libsophon_20240523_205217
sudo apt install ./sophon-libsophon_0.4.9-LTS_amd64.deb
sudo apt install ./sophon-libsophon-dev_0.4.9-LTS_amd64.deb
```

6. 下载模型MobleVLM
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download mtgv/MobileVLM_V2-1.7B --local-dir MobileVLM_V2-1.7B
```

7. 修改MobileVLM配置

将config.json中的max_sequence_length改为512，以满足内存需求

8. 将模型导出为ONNX
```
python compile/export_mobilevlm_onnx.py
```

9. 将ONNX模型，转成bmodel模型
```
./complie.sh
```

