## Typora+PicGo-Core搭建个人在线笔记

### 1、本文档解决的问题

Typora 是一款 Markdown 文档编写工具，也可以用来编写日常的工作记录。但是这个软件有个小问题，就是如果文档中插入了图片，在跨机器查看文档的时候图片就没法正常显示。

好在 Typora 提供了一些插件可以解决这个问题，我们可以在编写文档时把图片上传到网络上，这样无论在哪里，只要有网络文档就可以正常显示。

可以存储 Typora 的服务有很多，比如说 阿里云、腾讯云、SM.MS、七牛云、GitHub等，但是上述的像阿里云、腾讯云都需要收费、而七牛云有域名限制、GitHub网络又不好。所以再三思索下决定使用 Gitee，这样既可以存图片，又可以存 markdown 文档一举两得。

### 2、软件安装

我们需要下载两个软件：

- Typora 最新版本的安装包

- Node 软件安装包（用来安装插件）


#### 2.1、Typora 安装

下载完 Typora 之后，点击安装就可以了。如果想完成图片自动上传，则需要 Typora 支持上传服务设定。也就是在 偏好设置 的 图像 设置中要有下面这个选项才可以。比较低的版本是没有这个的，所以最好下载最新版。



#### 2.2、插件安装

安装插件前需要先安装 Node，安装完之后执行如下命令可以检验是否安装成功

```
C:\Users\denggh\Desktop>node -v
v12.16.3
```

需要用到的插件有两个，他们是 PicGo-Core 和 gitee-uploader 。安装的相关命令如下：

```
// 设置仓库为淘宝的，这样下载速度就会比较快
C:\Users\denggh\Desktop>npm config set registry https://registry.npm.taobao.org

// 安装 picgo-core
C:\Users\denggh\Desktop>npm install picgo -g

// 查看 picgo-core 安装到哪里了
C:\Users\denggh\Desktop>where picgo
C:\Users\denggh\AppData\Roaming\npm\picgo
C:\Users\denggh\AppData\Roaming\npm\picgo.cmd

// 安装 gitee-uploader 插件
C:\Users\denggh\Desktop>picgo install gitee-uploader
```


picgo 默认安装到 `~/AppData/Roaming/npm` 目录下。gitee-uploader 插件安装到 `~/.picgo/node_modules` 目录下，如果不想放到这个目录其实可以修改的。修改文件 `~/AppData/Roaming/npm/node_modules/picgo/dist/src/core/PicGo.js`文件找到以下内容，修改后面这个 `config.json` 文件位置即可。

```
this.configPath = os_1.homedir() + '/.picgo/config.json';
```

#### 2.3、Gitee 仓库配置

我们登录 Gitee 创建一个仓库，仓库需要公开的，不然图片无法访问。同时我们还需要创建一个仓库的私人令牌。在个人中心 安全设置 私人令牌 哪里就可以创建。

仓库名称就叫做 document 他有两个分支：master markdown-picture 。master 分支用来存放 md 文档，markdown-picture 用来存放图片。里面以年份作为文件夹存放当年的编写 md 文件的图片。

参考：https://gitee.com/readiay/document

#### 2.4、插件相关配置

我们找到`config.json` 文件，并修改他为如下内容

```java
{
  "picBed": {
    "current": "gitee",
    "uploader": "gitee",
    "gitee": {
      "branch": "markdown-picture", // 图片上传到这个分支
      "customPath": "",
      "customUrl": "",
      "path": "2020/", // 路径
      "repo": "readiay/document", // <用户名>/<仓库名称>
      "token": "34097jy0142b767hy64569baadeac247" // token 令牌
    }
  },
  "picgoPlugins": {
    "picgo-plugin-gitee-uploader": true
  },
  "picgo-plugin-gitee-uploader": {
    "lastSync": "2020-05-07 11:43:43"
  }
}
```

然后进入 Typora 软件，进行如下配置

![aHR0cHM6Ly9naXRlZS5jb20vcmVhZGlheS9kb2N1bWVudC9yYXcvbWFya2Rvd24tcGljdHVyZS8yMDIwL2ltYWdlLTIwMjAwNTA4MDAzMzMwNzY3LnBuZw](https://gitee.com/linchang98/document/raw/markdown-picture/2021/aHR0cHM6Ly9naXRlZS5jb20vcmVhZGlheS9kb2N1bWVudC9yYXcvbWFya2Rvd24tcGljdHVyZS8yMDIwL2ltYWdlLTIwMjAwNTA4MDAzMzMwNzY3LnBuZw.png)

### 4、测试运行

配置完之后，我们就可以进行 md 文档编写了，然后截图之后直接拷贝到 md 文档中选择上传图片即可。他会自动把图片的URL替换成对应的网络图片URL。

————————————————

版权声明：本文为CSDN博主「Readiay」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Readiay/article/details/105985859