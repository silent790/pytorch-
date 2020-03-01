
# CNN实现识别手写数字
# =================================================================
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
# 用到 PyTorch 自带的数据加载器包括 dataset，sampler，以及 data loader 这三个对象组成的套件。
# 自动进行数据的分布式加载
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np

# 定义超参
image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 20  #训练的总循环周期
batch_size = 64  #一个撮（批次）的大小，64张图片

# 加载 MNIST 数据，如果没有下载过，就会在当前路径下新建 /data 子目录，并把文件存放其中
# MNIST 数据是属于 torchvision 包自带的数据，所以可以直接调用。
train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                            train=True,   #提取训练集
                            #将图像转化为 Tensor，在加载数据的时候，就可以对图像做预处理
                            transform=transforms.ToTensor(),  
                            download=True) #当找不到文件的时候，自动下载

# 加载测试数据集
test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
'''
如果想要调用非 PyTorch 的自带数据，比如自己准备的数据集，
可以用 torchvision.datasets.ImageFolder 或者 torch.utils.data.TensorDataset 来加载
'''
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
                                    
# 首先创建 test_dataset 中所有数据的索引下标
indices = range(len(test_dataset))
# 利用数据下标，将 test_dataset 中的前 5000 条数据作为 校验数据
indices_val = indices[:5000]
# 剩下的就作为测试数据了
indices_test = indices[5000:]



# 根据分好的下标，构造两个数据集的 SubsetRandomSampler 采样器，它会对下标进行采样
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

# 校验数据集的加载器
validation_loader = torch.utils.data.DataLoader(dataset =test_dataset,
                                                batch_size = batch_size,
                                                sampler = sampler_val
                                               )

# 验证数据集的加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          sampler = sampler_test
                                         )

# 随便指定一个数据下标
idx = 100

# dataset 支持下标索引
# 其中提取出来的每一个元素为 features，target格式，即属性和标签
# [0] 表示索引 features
muteimg = train_dataset[idx][0].numpy()

# 由于一般的图像包含rgb三个通道，而MINST数据集的图像都是灰度的，只有一个通道
# 因此，我们忽略通道，把图像看作一个灰度矩阵
plt.imshow(muteimg[0,...], cmap ='gray')
print('标签是：',train_dataset[idx][1])

# 定义卷积神经网络：4 和 8 为人为指定的两个卷积层的厚度（feature map的数量）
depth = [4, 8]

class ConvNet(nn.Module):
    def __init__(self):
        # 该函数在创建一个 ConvNet 对象的时候，即调用如下语句：net=ConvNet()，就会被调用
        # 首先调用父类相应的构造函数
        super(ConvNet, self).__init__()

        # 其次构造ConvNet需要用到的各个神经模块。
        '''注意，定义组件并没有真正搭建这些组件，只是把基本建筑砖块先找好'''
        self.conv1 = nn.Conv2d(1, 4, 5, padding = 2) #定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.pool = nn.MaxPool2d(2, 2) #定义一个Pooling层，一个窗口为2*2的pooling运算
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2) #第二层卷积，输入通道为depth[0], 
                                                                   #输出通道为depth[1]，窗口为5，padding为2
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1] , 512) 
                                                            #一个线性连接层，输入尺寸为最后一层立方体的平铺，输出层512个节点
        self.fc2 = nn.Linear(512, num_classes) #最后一层线性分类单元，输入为512，输出为要做分类的类别数

    def forward(self, x):
        #该函数完成神经网络真正的前向运算，我们会在这里把各个组件进行实际的拼装
        #x的尺寸：(batch_size, image_channels, image_width, image_height)
        x = F.relu(self.conv1(x))  #第一层卷积，激活函数用ReLu，为了防止过拟合
        #x的尺寸：(batch_size, num_filters, image_width, image_height)
        x = self.pool(x) #第二层pooling，将图片变小
        #x的尺寸：(batch_size, depth[0], image_width/2, image_height/2)
        x = F.relu(self.conv2(x)) #第三层又是卷积，窗口为5，输入输出通道分别为depth[0]=4, depth[1]=8
        #x的尺寸：(batch_size, depth[1], image_width/2, image_height/2)
        x = self.pool(x) #第四层pooling，将图片缩小到原大小的1/4
        #x的尺寸：(batch_size, depth[1], image_width/4, image_height/4)

        # 将立体的特征图Tensor，压成一个一维的向量
        # view这个函数可以将一个tensor按指定的方式重新排布。
        # 下面这个命令就是要让x按照batch_size * (image_size//4)^2*depth[1]的方式来排布向量
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        #x的尺寸：(batch_size, depth[1]*image_width/4*image_height/4)

        x = F.relu(self.fc1(x)) #第五层为全链接，ReLu激活函数
        #x的尺寸：(batch_size, 512)

        x = F.dropout(x, training=self.training) #以默认为0.5的概率对这一层进行dropout操作，为了防止过拟合
        x = self.fc2(x) #全链接
        #x的尺寸：(batch_size, num_classes)

        #输出层为log_softmax，即概率对数值log(p(x))。采用log_softmax可以使得后面的交叉熵计算更快
        x = F.log_softmax(x, dim = 1) 
        return x

    def retrieve_features(self, x):
        #该函数专门用于提取卷积神经网络的特征图的功能，返回feature_map1, feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x)) #完成第一层卷积
        x = self.pool(feature_map1)  # 完成第一层pooling
        feature_map2 = F.relu(self.conv2(x)) #第二层卷积，两层特征图都存储到了feature_map1, feature_map2中
        return (feature_map1, feature_map2)


# 定义准确率
def rightness(predictions, labels):
    # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    pred = torch.max(predictions.data, 1)[1] 
    # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    # 返回正确的数量和这一次一共比较了多少元素
    return rights, len(labels) 

net = ConvNet() #新建一个卷积神经网络的实例，此时ConvNet的__init__函数就会被自动调用

criterion = nn.CrossEntropyLoss() #Loss函数的定义，交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #定义优化器，普通的随机梯度下降算法

# 参数:
# data : torch.Variable
# target: torch.Variable
def train_model(data, target):

    # 给网络模型做标记，标志说模型正在训练集上训练
    # 这种区分主要是为了打开 net 的 training 标志
    # 从而决定是否运行 dropout 与 batchNorm
    net.train() 

    output = net(data) #神经网络完成一次前馈的计算过程，得到预测输出output
    loss = criterion(output, target) #将output与标签target比较，计算误差
    optimizer.zero_grad() #清空梯度
    loss.backward() #反向传播
    optimizer.step() #一步随机梯度下降算法
    right = rightness(output, target) #计算准确率所需数值，返回数值为（正确样例数，总样本数）
    return right, loss


# Evaluation Mode
def evaluation_model():
    # net.eval() 给网络模型做标记，标志说模型现在是验证模式
    # 此方法将模型 net 的 training 标志设置为 False
    # 模型中将不会运行 dropout 与 batchNorm
    net.eval() 

    #记录校验数据集准确率的容器
    val_rights = [] 

    '''开始在校验数据集上做循环，计算校验集上面的准确度'''
    for (data, target) in validation_loader:
        data, target = Variable(data), Variable(target)

        # 完成一次模型的 forward 计算过程，得到模型预测的分类概率
        output = net(data) 

        # 统计正确数据次数，得到：（正确样例数，batch总样本数）
        right = rightness(output, target) 

        # 加入到容器中，以供后面计算正确率使用
        val_rights.append(right)

    return val_rights

# ----------------------------------------------------------------------------------------
record = [] #记录准确率等数值的容器
weights = [] #每若干步就记录一次卷积核

#开始训练循环
for epoch in range(num_epochs):
    train_rights = [] #记录训练数据集准确率的容器
    ''' 
    下面的enumerate是构造一个枚举器的作用。就是在对train_loader做循环迭代的时候，enumerate会自动吐出一个数字指示循环了几次
    这个数字就被记录在了batch_idx之中，它就等于0，1，2，……
    train_loader每迭代一次，就会吐出来一对数据data和target，分别对应着一个batch中的手写数字图，以及对应的标签。
    '''
    for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
        # 将 Tensor 转化为 Variable，data 为一批图像，target 为一批标签
        data, target = Variable(data), Variable(target) 

        # 调用模型训练函数
        right, loss = train_model(data, target)

        #将计算结果装到列表容器train_rights中
        train_rights.append(right) 

        if batch_idx % 100 == 0: #每间隔100个batch执行一次打印等操作

            # 调用模型验证函数
            val_rights = evaluation_model()

            # 统计验证模型时的正确率
            # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            # 统计上面训练模型时的正确率
            # train_r为一个二元组，分别记录目前已经经历过的所有训练集中分类正确的数量和该集合中总的样本数，
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

            # 计算并打印出模型在训练时和在验证时的准确率
            # train_r[0]/train_r[1]就是训练集的分类准确度，同样，val_r[0]/val_r[1]就是校验集上的分类准确度
            print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, 
                100. * train_r[0].numpy() / train_r[1], 
                100. * val_r[0].numpy() / val_r[1]))

            # 将准确率和权重等数值加载到容器中，以方便后面将模型训练曲线绘制出来
            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

            # 在这里将模型中的权重参数保存起来，以供后面解剖分析神经网络时使用
            # weights 录了训练周期中所有卷积核的演化过程，net.conv1.weight就提取出了第一层卷积核的权重
            # clone的意思就是将 weight.data 中的数据做一个拷贝放到列表中，
            # 否则当 weight.data 变化的时候，列表中的每一项数值也会联动
            '''这里使用clone这个函数很重要'''
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(), 
                            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

#绘制训练过程的误差曲线，校验集和测试集上的错误率。
plt.figure(figsize = (10, 7))
plt.plot(record) #record记载了每一个打印周期记录的训练和校验数据集上的准确度
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()


# ================================================================================================
#在测试集上分批运行，并计算总的正确率
net.eval() #标志模型当前为运行阶段
vals = [] #记录准确率所用列表

#对测试数据集进行循环
for data, target in test_loader:
    data, target = Variable(data, requires_grad=True), Variable(target)
    output = net(data) #将特征数据喂入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
    vals.append(val) #记录结果

#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0].data.numpy() / rights[1]
print(right_rate)

idx = 4
muteimg = test_dataset[idx][0].numpy()
plt.imshow(muteimg[0,...], cmap='gray')
print('正确标签是：', test_dataset[idx][1])

# 使用torch.view()将输入变换形状，神经网络的输入为：（batch, 1, 28, 28）
# 测试的时候只有一个数据，所以 batch 为 1
test_input = torch.Tensor(muteimg).view(1, 1, 28, 28)
out = net(Variable(test_input))
print('模型预测结果是：', torch.max(out, 1)[1].data.numpy())


# ==============================================================
# 解构该卷积神经网络：
print(net.parameters)

plt.figure(figsize = (10, 7))
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.axis('off')
    #提取第一层卷积核中的权重值，注意conv1是net的属性
    plt.imshow(net.conv1.weight.data.numpy()[i,0,...]) 

idx = 4
# 首先定义读入的图片
# 它是从 test_dataset 中提取第 idx 个批次的第 0 个图，其次 unsqueeze 的作用是在最前面添加一维
# 目的是为了让这个 input_x 的 tensor 是四维的，这样才能输入给 net，补充的那一维表示 batch
input_x = test_dataset[idx][0].unsqueeze(0) 

# 调用 net 的 retrieve_features 方法可以抽取出喂入当前数据后吐出来的所有特征图（第一个卷积和第二个卷积层）
feature_maps = net.retrieve_features(Variable(input_x))
# feature_maps 是有两个元素的列表，分别表示第一层和第二层卷积的所有特征图
# 所以 feature_maps[0] 就是第一层卷积的特征图

plt.figure(figsize = (10, 7))

#有四个特征图，循环把它们打印出来
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.axis('off')
    plt.imshow(feature_maps[0][0, i,...].data.numpy())
plt.show()
# ---------------------------------------------------------------------------------------
# 第二层卷积核：

plt.figure(figsize = (15, 10))
for i in range(4):
    for j in range(8):
        plt.subplot(4, 8, i * 8 + j + 1)
        plt.axis('off')
        plt.imshow(net.conv2.weight.data.numpy()[j, i,...])
plt.show()

# =========================================================================================
# 鲁棒性测试，平移图像
# 提取中test_dataset中的第idx个批次的第0个图的第0个通道对应的图像，定义为a。
a = test_dataset[idx][0][0]

# 平移后的新图像将放到b中。根据a给b赋值。
b = torch.zeros(a.size()) #全0的28*28的矩阵
w = 3 #平移的长度为3个像素

# 对于b中的任意像素i,j，它等于a中的i,j+w这个位置的像素
for i in range(a.size()[0]):
    for j in range(0, a.size()[1] - w):
        b[i, j] = a[i, j + w]

# 将b画出来
muteimg = b.numpy()
plt.axis('off')
plt.imshow(muteimg)

# 把b喂给神经网络，得到分类结果pred（prediction是预测的每一个类别的概率的对数值），并把结果打印出来
prediction = net(Variable(b.unsqueeze(0).unsqueeze(0)))
pred = torch.max(prediction.data, 1)[1]
print('预测结果：', pred)

#提取b对应的featuremap结果
feature_maps = net.retrieve_features(Variable(b.unsqueeze(0).unsqueeze(0)))

plt.figure(figsize = (10, 7))
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.axis('off')
    plt.imshow(feature_maps[0][0, i,...].data.numpy())

plt.figure(figsize = (10, 7))
for i in range(8):
    plt.subplot(2,4,i + 1)
    plt.axis('off')

    plt.imshow(feature_maps[1][0, i,...].data.numpy())
plt.show()