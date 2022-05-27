# v2fly！VLESS+Ws+Tls 一键安装脚本！VLESS协议！V2ray最新、无状态、轻量级的传输协议。V2Ray 客户端和服务器之间的桥梁协议

VMESS相信大家都不陌生了，他是V2RAY的传输协议。但是在5月V2RAY爆出新闻之后，很多大神们都在弄一个新的协议，最具代表的就是VLESS协议。

VLESS 是一个无状态的轻量传输协议，它分为入站和出站两部分，可以作为 V2Ray 客户端和服务器之间的桥梁。

与 VMess 不同，VLESS 不依赖于系统时间，认证方式同样为 UUID，但不需要 alterId。

> 今天为大家带来的是一键安装 VLESS+Ws+Tls 的脚本，很简单，只是告知大家，这个东西出来了

**本期内容视频：[点击播放](https://www.v2rayssr.com/go?url=https://youtu.be/qv-IlD2yKIY)**

## 准备工作

1、VPS一台，重置主流版本的操作系统

2、域名一个，做好解析 （没有域名或是不会 [点击这里](https://www.v2rayssr.com/yumingreg.html)）

## 开始搭建

![img](https://gitee.com/linchang98/document/raw/markdown-picture/2022/222.png)

今天使用是还是来自 wulabing 大神的脚本，项目地址为：[点击访问](https://www.v2rayssr.com/go?url=https://github.com/wulabing/V2Ray_ws-tls_bash_onekey)

本脚本位于 wulabing DEV 的分支，

此脚本 git 环境需要自己安装，不然伪装网站无法正常拉取！脚本并未集成该命令，安装命令如下

```
yum install -y git  #CentOS安装命令apt install -y git  #Debian安装命令
```

一键脚本如下：

```
wget -N --no-check-certificate -q -O install.sh "https://raw.githubusercontent.com/wulabing/V2Ray_ws-tls_bash_onekey/dev/install.sh" && chmod +x install.sh && bash install.sh
```

### 安装/更新方式（h2 和 ws 版本已合并）

Vmess+websocket+TLS+Nginx+Website

```
wget -N --no-check-certificate -q -O install.sh "https://raw.githubusercontent.com/wulabing/V2Ray_ws-tls_bash_onekey/master/install.sh" && chmod +x install.sh && bash install.sh
```

VLESS+websocket+TLS+Nginx+Website

```
wget -N --no-check-certificate -q -O install.sh "https://raw.githubusercontent.com/wulabing/V2Ray_ws-tls_bash_
```

熟悉的3D界面

![img](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E6%88%AA%E5%B1%8F2020-08-15-19.01.36.jpg)

## VLESS支持的客户端

（截止到2020年8月15日）

#### [V2RayNG](https://www.v2rayssr.com/go?url=https://github.com/2dust/v2rayNG) ![img](https://img.shields.io/github/commit-activity/m/2dust/v2rayNG?color=informational&label=commits&style=social) ![img](https://img.shields.io/github/stars/2dust/v2rayNG?style=social)

V2RayNG 是一个基于 V2Ray 内核的 Android 应用，它可以创建基于 VMess 的 VPN 连接。

#### [V2rayN](https://www.v2rayssr.com/go?url=https://github.com/2dust/v2rayN) ![img](https://img.shields.io/github/commit-activity/m/2dust/v2rayN?color=informational&label=commits&style=social) ![img](https://gitee.com/linchang98/document/raw/markdown-picture/2022/v2rayN)

V2RayN 是一个基于 V2Ray 内核的 Windows 客户端。

#### [V2rayU](https://www.v2rayssr.com/go?url=https://github.com/yanue/V2rayU) ![img](https://img.shields.io/github/commit-activity/m/yanue/V2rayU?color=informational&label=commits&style=social) ![img](https://gitee.com/linchang98/document/raw/markdown-picture/2022/V2rayU)

V2rayU，基于 V2Ray 核心的 macOS 客户端，使用 Swift 4.2 编写，支持 VMess、Shadowsocks、SOCKS5 等服务协议，支持订阅，支持二维码、剪贴板导入、手动配置、二维码分享等。

#### [Qv2ray](https://www.v2rayssr.com/go?url=https://github.com/Qv2ray/Qv2ray) ![img](https://img.shields.io/github/commit-activity/m/Qv2ray/Qv2ray?color=informational&label=commits&style=social) ![img](https://img.shields.io/github/stars/Qv2ray/Qv2ray?style=social)

跨平台 V2Ray 客户端，支持 Linux、Windows、macOS，可通过插件系统支持 SSR / Trojan / Trojan-Go / NaiveProxy 等协议，不支持批量测速，不支持自动更新，不建议小白使用

### 支持VLESS软路由插件

（截止到2020年8月15日）

#### [PassWall](https://www.v2rayssr.com/go?url=https://github.com/xiaorouji/openwrt-package)

PassWall（v3.9.35+），支持 OpenWRT



# 免费CND

```
免费CDN:cloudflare.com
传送门:https://dash.cloudflare.com/
按照要求注册即可
```

录入要加速的域名

![图片](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87.png)

使用免费版

![图片-1](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87-1.png)

因为cloudflare设置为dns服务器所以无法使用，单击继续进入下一步

![图片-2](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87-2.png)

在域名服务商那里将nameserver更改为cloudflare的nameserver

![图片-3](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87-3.png)

修改dns服务器为cloudflare的nameserver

等大约10分钟之后回到cloudflare，点击完成，检查nameservers

![图片-6](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87-6.png)

点击完成检查，成功后就可以进行域名解析及加速了

![图片-7-1024x616](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87-7-1024x616.png)

做一条A记录解析，并开启CDN，等大约10分钟后即可使用

这里需要注意一下，因为我们开启了tls，所以cloudflare到我们的服务器之间是需要进行全链路ssl，否则会导致访问失败的情况

![图片-8-1024x728](https://gitee.com/linchang98/document/raw/markdown-picture/2022/%E5%9B%BE%E7%89%87-8-1024x728.png)





