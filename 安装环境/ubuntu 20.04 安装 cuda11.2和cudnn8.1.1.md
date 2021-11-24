# ubuntu 20.04 安装 cuda11.2 和cudnn8.2.1

根据[TensorFlow 官网](https://tensorflow.google.cn/install/source_windows)，我们选取CUDA11.2搭配cuDNN8.1安装，NVIDIA驱动版本尽量460

![Snipaste_2021-11-24_10-01-27](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111241712367.png)

## 一、安装cuda11.2

### 1.打开官网

https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal

![img](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111241705783.png)

```
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run
```

到下面这个命令输入accept：

![image-20211124171614698](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111241716868.png)

这里一定要去掉选中的driver，然后install：

![image-20211124171004412](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111241710590.png)

### 2.设置环境变量

```
 vim ~/.bashrc
```

 #复制下面内容到最后一行保存退出

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
export PATH=$PATH:/usr/local/cuda-11.2/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.2
```

```
source ~/.bashrc
```

到此安装完成，输入 nvcc -V 验证 显示：



## 二、安装cudnn8.1

### 1.打开官网

https://developer.nvidia.com/rdp/cudnn-archive

```
uname -a #先查看下系统版本 我的是 20.04
```

下载这三个，根据cuda和服务器版本下载

![image-20211124172018021](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111241720082.png)

### 2.安装

```
#依次安装
sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb
```

### 3.验证cuDNN能不能用

#官方说法：To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v8 directory in the debian file.

```
# 将cuDNN例子复制到可写路径中
cp -r /usr/src/cudnn_samples_v8/ $HOME

# 转到可写路径
cd  ~/cudnn_samples_v8/mnistCUDNN

# 编译文件。
sudo make clean 
sudo make

# 运行样例程序。
sudo ./mnistCUDNN
```

如果成功运行，会显示下列信息：

```
bai@ubuntu:~/cudnn_samples_v8/mnistCUDNN$ sudo ./mnistCUDNN
Executing: mnistCUDNN
cudnnGetVersion() : 8101 , CUDNN_VERSION from cudnn.h : 8101 (8.1.1)
Host compiler version : GCC 9.3.0

There are 2 CUDA capable devices on your machine :
device 0 : sms 30  Capabilities 6.1, SmClock 1582.0 Mhz, MemSize (Mb) 12196, MemClock 5705.0 Mhz, Ecc=0, boardGroupID=0
device 1 : sms 30  Capabilities 6.1, SmClock 1582.0 Mhz, MemSize (Mb) 12192, MemClock 5705.0 Mhz, Ecc=0, boardGroupID=1
Using device 0

Testing single precision
Loading binary file data/conv1.bin
Loading binary file data/conv1.bias.bin
Loading binary file data/conv2.bin
Loading binary file data/conv2.bias.bin
Loading binary file data/ip1.bin
Loading binary file data/ip1.bias.bin
Loading binary file data/ip2.bin
Loading binary file data/ip2.bias.bin
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 184784 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.014336 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.014336 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.044928 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.049088 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.086016 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.488384 time requiring 184784 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 2000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 128000 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.046080 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.057344 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.069632 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.097152 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.110592 time requiring 128000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.242336 time requiring 2000 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 184784 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.014304 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.014336 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.037664 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.041984 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.058368 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.067584 time requiring 2057744 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 2000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 128000 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.046816 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.052192 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.059392 time requiring 2000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.069632 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.096256 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.109568 time requiring 128000 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!
```

#查看cudnn版本：

```
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

