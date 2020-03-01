if __name__ == '__main__': 
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
    import torch.nn.functional as F
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import copy
    import os

    # 数据存储总路径
    data_dir = 'transfer-data'
    # 图像的大小为224*224
    image_size = 224
    # 从data_dir/train加载文件
    # 加载的过程将会对图像自动作如下的图像增强操作：
    # 1. 随机从原始图像中切下来一块224*224大小的区域
    # 2. 随机水平翻转图像
    # 3. 将图像的色彩数值标准化
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                        transforms.Compose([
                                            transforms.RandomResizedCrop(image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
                                        )

    # 加载校验数据集，对每个加载的数据进行如下处理：
    # 1. 放大到256*256像素
    # 2. 从中心区域切割下224*224大小的图像区域
    # 3. 将图像的色彩数值标准化
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                        transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
                                        )

    # 创建相应的数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle = True, num_workers=4)

    # 读取得出数据中的分类类别数
    # 如果只有蜜蜂和蚂蚁，那么是2
    num_classes = len(train_dataset.classes)
    print(num_classes)

    # ===============================================================================================================
    # 检测本机器是否安装GPU，将检测结果记录在布尔变量use_cuda中
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    # 当可用GPU的时候，将新建立的张量自动加载到GPU中
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    # 定义imshow函数，可以将数据集中的某张图片打印出来
    def imshow(inp, title=None):
        # 将一张图打印显示出来，inp为一个张量，title为显示在图像上的文字

        # 一般的张量格式为：channels * image_width * image_height
        # 而一般的图像为 image_width * image_height * channels 
        # 所以，需要将张量中的 channels 转换到最后一个维度
        inp = inp.cpu().numpy().transpose((1, 2, 0)) 

        #由于在读入图像的时候所有图像的色彩都标准化了，因此我们需要先调回去
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1) 

        #将图像绘制出来
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # 暂停一会是为了能够将图像显示出来。

    #获取第一个图像batch和标签
    images, labels = next(iter(train_loader))

    # 将这个batch中的图像制成表格绘制出来
    out = torchvision.utils.make_grid(images)

    imshow(out, title=[train_dataset.classes[x] for x in labels])


    # ==========================================================================================
    # 用于手写数字识别的卷积神经网络
    depth = [4, 8]

    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 4, 5, padding = 2) #输入通道为1，输出通道为4，窗口大小为5，padding为2
            self.pool = nn.MaxPool2d(2, 2) #一个窗口为2*2的pooling运算
            self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2) #第二层卷积，输入通道为depth[0], 输出通道为depth[1]，窗口wei15，padding为2
            self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1] , 512) #一个线性连接层，输入尺寸为最后一层立方体的平铺，输出层512个节点
            self.fc2 = nn.Linear(512, num_classes) #最后一层线性分类单元，输入为

        def forward(self, x):
            #神经网络完成一步前馈运算的过程，从输入到输出
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            # 将立体的Tensor全部转换成一维的Tensor。两次pooling操作，所以图像维度减少了1/4
            x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
            x = F.relu(self.fc1(x)) #全链接，激活函数
            x = F.dropout(x, training=self.training) #以默认为0.5的概率对这一层进行dropout操作
            x = self.fc2(x) #全链接，激活函数
            x = F.log_softmax(x, dim=1) #log_softmax可以理解为概率对数值
            return x
    
    def rightness(predictions, labels):
        # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
        pred = torch.max(predictions.data, 1)[1] 
        # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
        rights = pred.eq(labels.data.view_as(pred)).sum() 
        # 返回正确的数量和这一次一共比较了多少元素
        return rights, len(labels)

    # 加载网络
    net = ConvNet()
    # 如果有GPU就把网络加载到GPU中
    net = net.cuda() if use_cuda else net
    criterion = nn.CrossEntropyLoss() #Loss函数的定义
    optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9)

    # 参数:
    # data : Variable
    # target: Variable
    def train_model(data, target):

        # 给网络模型做标记，标志说模型正在训练集上训练
        # 这种区分主要是为了打开 net 的 training 标志
        # 从而决定是否运行 dropout 与 batchNorm
        net.train() 

        output = net(data) # 神经网络完成一次前馈的计算过程，得到预测输出output
        loss = criterion(output, target) # 将output与标签target比较，计算误差
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 一步随机梯度下降算法

        # 计算准确率所需数值，返回数值为（正确样例数，总样本数）
        right = rightness(output, target) 

        # 如果计算在 GPU 中，打印的数据再加载到CPU中
        loss = loss.cpu() if use_cuda else loss
        return right, loss

    # Evaluation Mode
    def evaluation_model():
        # net.eval() 给网络模型做标记，标志说模型现在是验证模式
        # 此方法将模型 net 的 training 标志设置为 False
        # 模型中将不会运行 dropout 与 batchNorm
        net.eval() 

        vals = []
        #对测试数据集进行循环
        for data, target in val_loader:
            data, target = Variable(data, requires_grad=True), Variable(target)
            # 如果GPU可用，就把数据加载到GPU中
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data) #将特征数据喂入网络，得到分类的输出
            val = rightness(output, target) #获得正确样本数以及总样本数
            vals.append(val) #记录结果

        return vals  
    record = [] #记录准确率等数值的容器

    #开始训练循环
    num_epochs = 20
    best_model = net
    best_r = 0.0

    for epoch in range(num_epochs):
        #optimizer = exp_lr_scheduler(optimizer, epoch)
        train_rights = [] #记录训练数据集准确率的容器
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
            data, target = Variable(data), Variable(target) #将Tensor转化为Variable，data为图像，target为标签
            # 如果有GPU就把数据加载到GPU上
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # 调用训练函数
            right, loss = train_model(data, target)

            train_rights.append(right) #将计算结果装到列表容器中

            train_losses.append(loss.data.cpu().numpy())


        # train_r 为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

        #在测试集上分批运行，并计算总的正确率
        vals = evaluation_model()

        #计算准确率
        val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
        val_ratio = 1.0*val_r[0].cpu().numpy()/val_r[1]

        if val_ratio > best_r:
            best_r = val_ratio
            best_model = copy.deepcopy(net)
        #打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
        print('训练周期: {} \tLoss: {:.6f}\t训练正确率: {:.2f}%, 校验正确率: {:.2f}%'.format(
            epoch, np.mean(train_losses), 100. * train_r[0].cpu().numpy() / train_r[1], 100. * val_r[0].cpu().numpy()/val_r[1]))       
        record.append([np.mean(train_losses), 1. * train_r[0].data.cpu().numpy() / train_r[1], 1. * val_r[0].data.cpu().numpy() / val_r[1]])

    #在测试集上分批运行，并计算总的正确率
    net.eval() #标志模型当前为运行阶段
    test_loss = 0
    correct = 0
    vals = []

    #对测试数据集进行循环
    for data, target in val_loader:
        data, target = Variable(data, requires_grad=True), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data) #将特征数据喂入网络，得到分类的输出
        val = rightness(output, target) #获得正确样本数以及总样本数
        vals.append(val) #记录结果

    #计算准确率
    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    right_rate = 1.0 * rights[0].data.cpu().numpy() / rights[1]
    print(right_rate)

    # 绘制误差率曲线
    x = [x[0] for x in record]
    y = [1 - x[1] for x in record]
    z = [1 - x[2] for x in record]
    #plt.plot(x)
    plt.figure(figsize = (10, 7))
    plt.plot(y)
    plt.plot(z)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.show()

