# Win11应用商店打不开怎么办？

#### 1、首先在Win11系统桌面，单机搜索按钮，在弹出菜单中选择搜索`WIndows Powershell`，然后右键以管理员身份启动。

![image-20211117112457305](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171124477.png)

#### 2、接着在开启的的窗口中输入指令：

```
Get-AppxPackage -allusers | Select Name, PackageFullName
```

![image-20211117112844868](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171128905.png)

#### 3、然后在出现的窗口中找到`Microsoft.WindowsStore`相对应的数值，然后复制它后面的具体数据。

![image-20211117113027836](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171130873.png)

#### 4、接下来在下面的窗口中输入以下指令：

```
Add-appxpackage -register "C:\Program Files\WindowsApps\Microsoft.WindowsStore_22110.1402.13.0_x64__8wekyb3d8bbwe\appxmanifest.xml"-disabledevelopmentmode
```

注意我们需要将中间对应的值换成我们上面复制的详细数值，我这里修改的是`Microsoft.WindowsStore_12107.1001.15.0_x64__8wekyb3d8bbwe`

![image-20211117113156278](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171131304.png)

#### 5、设置完成后win11系统就会自动开始重新部署应用商店。

![IMG_260](C:/Users/bai/AppData/Local/Temp/msohtmlclip1/01/clip_image005.jpg)

#### 6、这时我们开启应用商店即可正常使用啦。

# Win11应用商店无法加载页面怎么解决？

#### 1、首先右击左下角的微软徽标，选择设置。

![image-20211117113327427](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171133477.png)

#### 2、接着在设置中搜索`Internet 选项`

#### ![image-20211117113436807](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171134838.png)

#### 5、最后单击选项中的高级选项卡，勾选“使用TLS1.1和1.2”即可。

![image-20211117113525043](https://gitee.com/linchang98/document/raw/markdown-picture/2021/202111171135079.png)