# 你必须要知道CNN模型：ResNet

**欢迎交流与转载，文章会同步发布在公众号：机器学习算法全栈工程师(Jeemy110)**

## 引言

深度残差网络（Deep residual network, ResNet）的提出是CNN图像史上的一件里程碑事件，让我们先看一下ResNet在ILSVRC和COCO 2015上的战绩：

![img](https://gitee.com/linchang98/document/raw/markdown-picture/2021/v2-5e98ec97def099a4e6fb6b7ce3b1d460_720w.jpg)

​												图1 ResNet在ILSVRC和COCO 2015上的战绩

ResNet取得了5项第一，并又一次刷新了CNN模型在ImageNet上的历史：

![img](https://gitee.com/linchang98/document/raw/markdown-picture/2021/v2-606573bdaaa97de6b8b10fb00f76d29a_720w.jpg)

​																	图2 ImageNet分类Top-5误差

ResNet的作者[何凯明](https://link.zhihu.com/?target=http%3A//kaiminghe.com/)也因此摘得CVPR2016最佳论文奖，当然何博士的成就远不止于此，感兴趣的可以去搜一下他后来的辉煌战绩。那么ResNet为什么会有如此优异的表现呢？其实ResNet是解决了深度CNN模型难训练的问题，从图2中可以看到14年的VGG才19层，而15年的ResNet多达152层，这在网络深度完全不是一个量级上，所以如果是第一眼看这个图的话，肯定会觉得ResNet是靠深度取胜。事实当然是这样，但是ResNet还有架构上的trick，这才使得网络的深度发挥出作用，这个trick就是残差学习（Residual learning）。下面详细讲述ResNet的理论及实现。

## 深度网络的退化问题

从经验来看，网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时理论上可以取得更好的结果，从图2中也可以看出网络越深而效果越好的一个实践证据。但是更深的网络其性能一定会更好吗？实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。这个现象可以在图3中直观看出来：56层的网络比20层网络效果还要差。这不会是过拟合问题，因为56层网络的训练误差同样高。我们知道深层网络存在着梯度消失或者爆炸的问题，这使得深度学习模型很难训练。但是现在已经存在一些技术手段如BatchNorm来缓解这个问题。因此，出现深度网络的退化问题是非常令人诧异的。

![img](https://i.loli.net/2021/11/08/7LbNGAop4Iu5S8r.png)

​															图3 20层与56层网络在CIFAR-10上的误差

## 残差学习

深度网络的退化问题至少说明深度网络不容易训练。但是我们考虑这样一个事实：现在你有一个浅层网络，你想通过向上堆积新层来建立深层网络，一个极端情况是这些增加的层什么也不学习，仅仅复制浅层网络的特征，即这样新层是恒等映射（Identity mapping）。在这种情况下，深层网络应该至少和浅层网络性能一样，也不应该出现退化现象。好吧，你不得不承认肯定是目前的训练方法有问题，才使得深层网络很难去找到一个好的参数。

这个有趣的假设让何博士灵感爆发，他提出了**残差学习来解决退化问题**。对于一个堆积层结构（几层堆积而成）当输入为 ![[公式]](https://www.zhihu.com/equation?tex=x) 时其学习到的特征记为 ![[公式]](https://www.zhihu.com/equation?tex=H(x)) ，现在我们希望其可以学习到残差 ![[公式]](https://www.zhihu.com/equation?tex=F(x)%3DH(x)-x) ，这样其实原始的学习特征是 ![[公式]](https://www.zhihu.com/equation?tex=F(x)%2Bx) 。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。残差学习的结构如图4所示。这有点类似与电路中的“短路”，所以是一种短路连接（shortcut connection)。

![img](https://pic4.zhimg.com/80/v2-252e6d9979a2a91c2d3033b9b73eb69f_720w.jpg)

​																				图4 残差学习单元

为什么残差学习相对更容易，从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。不过我们可以从数学的角度来分析这个问题，首先残差单元可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=\begin{align}+%26+{{y}_{l}}%3Dh({{x}_{l}})%2BF({{x}_{l}}%2C{{W}_{l}})+\\+%26+{{x}_{l%2B1}}%3Df({{y}_{l}})+\\+\end{align}+)

其中 ![[公式]](https://www.zhihu.com/equation?tex=x_{l}) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_{l%2B1}) 分别表示的是第 ![[公式]](https://www.zhihu.com/equation?tex=l) 个残差单元的输入和输出，注意每个残差单元一般包含多层结构。 ![[公式]](https://www.zhihu.com/equation?tex=F) 是残差函数，表示学习到的残差，而 ![[公式]](https://www.zhihu.com/equation?tex=h(x_{l})%3Dx_{l}) 表示恒等映射， ![[公式]](https://www.zhihu.com/equation?tex=f) 是ReLU激活函数。基于上式，我们求得从浅层 ![[公式]](https://www.zhihu.com/equation?tex=l) 到深层 ![[公式]](https://www.zhihu.com/equation?tex=L) 的学习特征为：

![[公式]](https://www.zhihu.com/equation?tex={{x}_{L}}%3D{{x}_{l}}%2B\sum\limits_{i%3Dl}^{L-1}{F({{x}_{i}}}%2C{{W}_{i}}))

利用链式规则，可以求得反向过程的梯度：

![image-20211108170022282](https://i.loli.net/2021/11/08/PQw1S2zGiv9REfd.png)

式子的第一个因子 ![[公式]](https://www.zhihu.com/equation?tex=\frac{\partial+loss}{\partial+{{x}_{L}}}) 表示的损失函数到达 ![[公式]](https://www.zhihu.com/equation?tex=L) 的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。要注意上面的推导并不是严格的证明。

## ResNet的网络结构

ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如图5所示。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。从图5中可以看到，ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习，其中虚线表示feature map数量发生了改变。图5展示的34-layer的ResNet，还可以构建更深的网络如表1所示。从表中可以看到，对于18-layer和34-layer的ResNet，其进行的两层间的残差学习，当网络更深时，其进行的是三层间的残差学习，三层卷积核分别是1x1，3x3和1x1，一个值得注意的是隐含层的feature map数量是比较小的，并且是输出feature map数量的1/4。

![img](https://i.loli.net/2021/11/08/87owDE3aIYhUeNB.jpg)图5 ResNet网络结构图

![img](https://gitee.com/linchang98/document/raw/markdown-picture/2021/v2-1dfd4022d4be28392ff44c49d6b4ed94_720w.jpg)表1 不同深度的ResNet

下面我们再分析一下残差单元，ResNet使用两种残差单元，如图6所示。左图对应的是浅层网络，而右图对应的是深层网络。对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：（1）采用zero-padding增加维度，此时一般要先做一个downsamp，可以采用strde=2的pooling，这样不会增加参数；（2）采用新的映射（projection shortcut），一般采用1x1的卷积，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用projection shortcut。

![img](https://pic1.zhimg.com/80/v2-0892e5423616c30f69ded61111b111c0_720w.jpg)图6 不同的残差单元

作者对比18-layer和34-layer的网络效果，如图7所示。可以看到普通的网络出现退化现象，但是ResNet很好的解决了退化问题。

![img](https://pic2.zhimg.com/80/v2-ac88d9e118e3a85922188daba84f7efd_720w.jpg)

​																图7 18-layer和34-layer的网络效果

最后展示一下ResNet网络与其他网络在ImageNet上的对比结果，如表2所示。可以看到ResNet-152其误差降到了4.49%，当采用集成模型后，误差可以降到3.57%。

![img](https://pic4.zhimg.com/80/v2-0a2c8a209a221817f91c1f1728327beb_720w.jpg)表2 ResNet与其他网络的对比结果

说一点关于残差单元题外话，上面我们说到了短路连接的几种处理方式，其实作者在[文献[2\]](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1603.05027)中又对不同的残差单元做了细致的分析与实验，这里我们直接抛出最优的残差结构，如图8所示。改进前后一个明显的变化是采用pre-activation，BN和ReLU都提前了。而且作者推荐短路连接采用恒等变换，这样保证短路连接不会有阻碍。感兴趣的可以去读读这篇文章。

![img](https://pic1.zhimg.com/80/v2-4e0bf37ecad2f306fe09d32a2d37d908_720w.jpg)

​																	图8 改进后的残差单元及效果

## ResNet的TensorFlow实现

这里给出ResNet50的TensorFlow实现，模型的实现参考了[Caffe版本](https://link.zhihu.com/?target=https%3A//github.com/KaimingHe/deep-residual-networks)的实现，核心代码如下：

```python
class ResNet50(object):
    def __init__(self, inputs, num_classes=1000, is_training=True,
                 scope="resnet50"):
        self.inputs =inputs
        self.is_training = is_training
        self.num_classes = num_classes

        with tf.variable_scope(scope):
            # construct the model
            net = conv2d(inputs, 64, 7, 2, scope="conv1") # -> [batch, 112, 112, 64]
            net = tf.nn.relu(batch_norm(net, is_training=self.is_training, scope="bn1"))
            net = max_pool(net, 3, 2, scope="maxpool1")  # -> [batch, 56, 56, 64]
            net = self._block(net, 256, 3, init_stride=1, is_training=self.is_training,
                              scope="block2")           # -> [batch, 56, 56, 256]
            net = self._block(net, 512, 4, is_training=self.is_training, scope="block3")
                                                        # -> [batch, 28, 28, 512]
            net = self._block(net, 1024, 6, is_training=self.is_training, scope="block4")
                                                        # -> [batch, 14, 14, 1024]
            net = self._block(net, 2048, 3, is_training=self.is_training, scope="block5")
                                                        # -> [batch, 7, 7, 2048]
            net = avg_pool(net, 7, scope="avgpool5")    # -> [batch, 1, 1, 2048]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze") # -> [batch, 2048]
            self.logits = fc(net, self.num_classes, "fc6")       # -> [batch, num_classes]
            self.predictions = tf.nn.softmax(self.logits)


    def _block(self, x, n_out, n, init_stride=2, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            h_out = n_out // 4
            out = self._bottleneck(x, h_out, n_out, stride=init_stride,
                                   is_training=is_training, scope="bottlencek1")
            for i in range(1, n):
                out = self._bottleneck(out, h_out, n_out, is_training=is_training,
                                       scope=("bottlencek%s" % (i + 1)))
            return out

    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]
        if stride is None:
            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):
            h = conv2d(x, h_out, 1, stride=stride, scope="conv_1")
            h = batch_norm(h, is_training=is_training, scope="bn_1")
            h = tf.nn.relu(h)
            h = conv2d(h, h_out, 3, stride=1, scope="conv_2")
            h = batch_norm(h, is_training=is_training, scope="bn_2")
            h = tf.nn.relu(h)
            h = conv2d(h, n_out, 1, stride=1, scope="conv_3")
            h = batch_norm(h, is_training=is_training, scope="bn_3")

            if n_in != n_out:
                shortcut = conv2d(x, n_out, 1, stride=stride, scope="conv_4")
                shortcut = batch_norm(shortcut, is_training=is_training, scope="bn_4")
            else:
                shortcut = x
            return tf.nn.relu(shortcut + h)
```

完整实现可以参见[GitHub](https://link.zhihu.com/?target=https%3A//github.com/xiaohu2015/DeepLearning_tutorials/)。

## 总结

ResNet通过残差学习解决了深度网络的退化问题，让我们可以训练出更深的网络，这称得上是深度网络的一个历史大突破吧。也许不久会有更好的方式来训练更深的网络，让我们一起期待吧！

## 参考资料

1. [Deep Residual Learning for Image Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.03385).
2. [Identity Mappings in Deep Residual Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1603.05027).
3. [去膜拜一下大神](https://link.zhihu.com/?target=http%3A//kaiminghe.com/).
