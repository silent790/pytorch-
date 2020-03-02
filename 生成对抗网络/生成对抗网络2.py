if __name__ == "__main__":

    import os

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.optim as optim
    import torch.nn.functional as F

    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    import torchvision.utils as vutil


    import matplotlib.pyplot as plt
    import numpy as np
    nn.Module.dump_patches = True
    # 定义超参数
    image_size = 28 #图像尺寸大小
    input_dim = 100 #输入给生成器的向量维度，维度越大可以增加生成器输出样本的多样性
    num_channels = 1# 图像的通道数
    num_features = 64 #生成器中间的卷积核数量
    batch_size = 64 #批次大小

    use_cuda = torch.cuda.is_available() #定义一个布尔型变量，标志当前的GPU是否可用

    # 如果当前GPU可用，则将优先在GPU上进行张量计算
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ===============================================================================================

    # 加载MNIST数据，如果没有下载过，就会在当前路径下新建/data子目录，并把文件存放其中 
    # MNIST数据是属于torchvision包自带的数据，所以可以直接调用。
    # 在调用自己的数据的时候，我们可以用torchvision.datasets.ImageFolder或者torch.utils.data.TensorDataset来加载
    train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                                train=True,   #提取训练集
                                #将图像转化为Tensor，在加载数据的时候，就可以对图像做预处理
                                transform=transforms.ToTensor(),  
                                download=True) #当找不到文件的时候，自动下载

    # 加载测试数据集
    test_dataset = dsets.MNIST(root='./data', 
                            train=False, 
                            transform=transforms.ToTensor())

    # 首先创建 test_dataset 中所有数据的索引下标
    indices = range(len(test_dataset))
    # 利用数据下标，将 test_dataset 中的前 5000 条数据作为 校验数据
    indices_val = indices[:5000]
    # 剩下的就作为测试数据了
    indices_test = indices[5000:]


    # 根据这些下标，构造两个数据集的SubsetRandomSampler采样器，它会对下标进行采样
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    # 根据两个采样器来定义加载器，注意将sampler_val和sampler_test分别赋值给了validation_loader和test_loader
    validation_loader = torch.utils.data.DataLoader(dataset =test_dataset,
                                                    batch_size = batch_size,
                                                    sampler = sampler_val
                                                )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            sampler = sampler_test
                                            )
    # 训练数据集的加载器，自动将数据分割成batch，顺序随机打乱
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True # shuffle 代表打乱数据
                                            )
    def make_show(img):
        # 将张量变成可以显示的图像
        img = img.data.expand(batch_size, 3, image_size, image_size)
        return img
    def imshow(inp, title=None, ax=None):
        # 在屏幕上绘制图像
        """Imshow for Tensor."""
        if inp.size()[0] > 1:
            inp = inp.cpu().numpy().transpose((1, 2, 0))
        else:
            inp = inp[0].cpu().numpy()
        mvalue = np.amin(inp)
        maxvalue = np.amax(inp)
        if maxvalue > mvalue:
            inp = (inp - mvalue)/(maxvalue - mvalue)
        ax.imshow(inp)
        if title is not None:
            ax.set_title(title)
# 生成器建立类：
    #生成器模型定义
    class ModelG(nn.Module):
        def __init__(self):
            super(ModelG,self).__init__()
            self.model=nn.Sequential() #model为一个内嵌的序列化的神经网络模型

            # 利用add_module增加一个反卷积层，输入为input_dim维，输出为2*num_features维，窗口大小为5，padding是0
            # 输入图像大小为1，输出图像大小为W'=(W-1)S-2P+K+P'=(1-1)*2-2*0+5+0=3, 5*5
            self.model.add_module('deconv1',nn.ConvTranspose2d(input_dim, num_features*2, 5, 2, 0, bias=False))
            # 增加一个batchnorm层
            self.model.add_module('bnorm1',nn.BatchNorm2d(num_features*2))
            # 增加非线性层
            self.model.add_module('relu1',nn.ReLU(True))
            # 增加第二层反卷积层，输入2*num_features维，输出num_features维，窗口5，padding=0
            # 输入图像大小为5，输出图像大小为W'=(W-1)S-2P+K+P'=(5-1)*2-2*0+5+0=13, 13*13
            self.model.add_module('deconv2',nn.ConvTranspose2d(num_features*2, num_features, 5, 2, 0, bias=False))
            # 增加一个batchnorm层
            self.model.add_module('bnorm2',nn.BatchNorm2d(num_features))
            # 增加非线性层
            self.model.add_module('relu2',nn.ReLU(True))

            # 增加第二层反卷积层，输入2*num_features维，输出num_features维，窗口4，padding=0
            # 输入图像大小为13，输出图像大小为W'=(W-1)S-2P+K+P'=(13-1)*2-2*0+4+0=28, 28*28
            self.model.add_module('deconv3',nn.ConvTranspose2d(num_features, num_channels, 4, 2, 0,bias=False))
            #self.model.add_module('tanh',nn.Tanh())
            self.model.add_module('sigmoid',nn.Sigmoid())
        def forward(self,input):
            output = input

            #遍历网络的所有层，一层层输出信息
            for name, module in self.model.named_children():
                output = module(output)
            #输出一张28*28的图像
            return(output)

    def weight_init(m):
        class_name=m.__class__.__name__
        if class_name.find('conv')!=-1:
            m.weight.data.normal_(0,0.02)
        if class_name.find('norm')!=-1:
            m.weight.data.normal_(1.0,0.02)
# =========================================================================================================
    #定义生成器模型
    net = ModelG()
    net = net.cuda() if use_cuda else net #转到GPU上

    #目标函数采用最小均方误差
    criterion = nn.MSELoss()
    #定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # 随机选择生成0-9的数字，用于每个周期打印查看结果用
    samples = np.random.choice(10, batch_size) 
    samples = Variable(torch.from_numpy(samples).type(dtype))

    # 改变输入数字的尺寸，适应于生成器网络
    with torch.no_grad():
        samples.resize_(batch_size,1,1,1)
    samples = Variable(samples.data.expand(batch_size, input_dim, 1, 1))
    samples = samples.cuda() if use_cuda else samples #加载到GPU

    def save_evaluation_samples(netModel, save_path='gan'):
        # 去除首位空格
        save_path = save_path.strip()
        if not os.path.exists(save_path):
            os.makedirs(save_path) 

        # 产生一组图像保存到指定文件夹下，检测生成器当前的效果
        fake_u = netModel(samples) #用原始网络作为输入，得到伪造的图像数据
        fake_u = fake_u.cpu() if use_cuda else fake_u
        img = make_show(fake_u) #将张量转化成可绘制的图像
        vutil.save_image(img, save_path + '/fake %s.png'% (epoch)) #保存生成的图像

    def train_ModelG(target, data):
        # 将数据加载到GPU中
        if use_cuda:
            target, data = target.cuda(), data.cuda()
        #将输入的数字标签转化为生成器net能够接受的(batch_size, input_dim, 1, 1)维张量
        data = data.type(dtype)
        data = data.reshape(data.size()[0], 1, 1, 1)
        data = data.expand(data.size()[0], input_dim, 1, 1)

        net.train() # 给网络模型做标记，标志说模型正在训练集上训练，
                    #这种区分主要是为了打开关闭net的training标志
        output = net(data) #神经网络完成一次前馈的计算过程，得到预测输出output
        loss = criterion(output, target) #将output与标签target比较，计算误差
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #一步随机梯度下降算法

        if use_cuda:
            loss = loss.cpu()
        return loss

    def evaluation_ModelG():
        net.eval() # 给网络模型做标记，标志说模型在校验集上运行
        val_loss = [] #记录校验数据集准确率的容器

        '''开始在校验数据集上做循环，计算校验集上面的准确度'''
        idx = 0
        for (data, target) in validation_loader:
            target, data = Variable(data), Variable(target)
            idx += 1
            if use_cuda:
                target, data = target.cuda(), data.cuda()
            data = data.type(dtype)
            data = data.reshape(data.size()[0], 1, 1, 1)
            data = data.expand(data.size()[0], input_dim, 1, 1)
            output = net(data) #完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
            loss = criterion(output, target) #将output与标签target比较，计算误差
            if use_cuda:
                loss = loss.cpu()
            val_loss.append(loss.data.numpy())
        return val_loss
# ========================================================================================================
    # 定义待迁移的网络框架，所有的神经网络模块包括：Conv2d、MaxPool2d，Linear等模块都不需要重新定义，会自动加载
    # 但是网络的forward功能没有办法自动实现，需要重写。
    # 一般的，加载网络只加载网络的属性，不加载方法
    depth = [4, 8]
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
        def forward(self, x):
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
        def retrieve_features(self, x):
            #该函数专门用于提取卷积神经网络的特征图的功能，返回feature_map1, feature_map2为前两层卷积层的特征图
            feature_map1 = F.relu(self.conv1(x)) #完成第一层卷积
            x = self.pool(feature_map1)  # 完成第一层pooling
            feature_map2 = F.relu(self.conv2(x)) #第二层卷积，两层特征图都存储到了feature_map1, feature_map2中
            return (feature_map1, feature_map2)
    def rightness(predictions, labels):
        # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
        pred = torch.max(predictions.data, 1)[1] 
        # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
        rights = pred.eq(labels.data.view_as(pred)).sum() 
        return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
    netR = torch.load('minst_conv_checkpoint') #读取硬盘上的minst_conv_checkpoint文件
    netR = netR.cuda() if use_cuda else netR #加载到GPU中
    for para in netR.parameters():
        para.requires_grad = False #将识别器的权重设置为固定值
    #===================================================================================================== 
    netG = ModelG() #新建一个生成器
    netG = netG.cuda() if use_cuda else netG #加载到GPU上
    netG.apply(weight_init) #初始化参数

    criterion = nn.CrossEntropyLoss() #用交叉熵作为损失函数
    optimizer = optim.SGD(netG.parameters(), lr=0.0001, momentum=0.9) #定义优化器


    def train_ConvNet(target, data):
        if use_cuda:
            target, data = target.cuda(), data.cuda()

        # 复制标签变量放到了label中
        label = data.clone()

        data = data.type(dtype)
        # 改变张量形状以适用于生成器网络
        data = data.reshape(data.size()[0], 1, 1, 1)
        data = data.expand(data.size()[0], input_dim, 1, 1)

        netG.train() # 给网络模型做标记，标志说模型正在训练集上训练，
        netR.train() #这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout
#   连接处
        output1 = netG(data) #神经网络完成一次前馈的计算过程，得到预测输出output
        output = netR(output1) #用识别器网络来做分类

        loss = criterion(output, label) #将output与标签target比较，计算误差
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #一步随机梯度下降算法

        right = rightness(output, label) #计算准确率所需数值，返回数值为（正确样例数，总样本数）

        if use_cuda:
            loss = loss.cpu()
        return right, loss

    def evaluation_ConvNet():
        netG.eval() # 给网络模型做标记，标志说模型正在校验集上运行，
        netR.eval() #这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout
        val_loss = [] #记录校验数据集准确率的容器
        val_rights = []

        '''开始在校验数据集上做循环，计算校验集上面的准确度'''
        for (data, target) in validation_loader:
            # 注意target是图像，data是标签
            target, data = Variable(data), Variable(target)
            if use_cuda:
                target, data = target.cuda(), data.cuda()
            label = data.clone()
            data = data.type(dtype)
            #改变Tensor大小以适应生成网络
            data = data.reshape(data.size()[0], 1, 1, 1)
            data = data.expand(data.size()[0], input_dim, 1, 1)

            output1 = netG(data) #神经网络完成一次前馈的计算过程，得到预测输出output
            output = netR(output1) #利用识别器来识别
            loss = criterion(output, label) #将output与标签target比较，计算误差
            if use_cuda:
                loss = loss.cpu()
            val_loss.append(loss.data.numpy())
            right = rightness(output, label) #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
            val_rights.append(right)

        return val_loss, val_rights

    # 随机选择batch_size个数字，用他们来生成数字图像
    samples = np.random.choice(10, batch_size)
    samples = Variable(torch.from_numpy(samples).type(dtype))

    # 产生一组图像保存到temp1文件夹下（需要事先建立好该文件夹），检测生成器当前的效果
    with torch.no_grad():
        samples.resize_(batch_size,1,1,1)
    samples = Variable(samples.data.expand(batch_size, input_dim, 1, 1))
    samples = samples.cuda() if use_cuda else samples
# =============================================================================================
    #开始训练
    step = 0 #计数经历了多少时间步
    # num_epochs = 100 #总的训练周期
    num_epochs = 1 # 因训练模型的时间过长，建议在自己的环境中完成完整训练
    record = []
    print('Initialized!')

    # num_epochs = 100 #总训练周期
    num_epochs = 1 # 建议在自己的环境中完成完整的训练
    statistics = [] #数据记载器
    for epoch in range(num_epochs):
        train_loss = []
        train_rights = []

        # 加载数据
        for batch_idx, (data, target) in enumerate(train_loader):
            # !!!!注意图像和标签互换了!!!!
            target, data = Variable(data), Variable(target) #将Tensor转化为Variable，data为一批标签，target为一批图像     
            # 调用训练函数
            right, loss = train_ConvNet(target, data)       
            train_loss.append(loss.data.numpy())   
            train_rights.append(right) #将计算结果装到列表容器train_rights中
            step += 1

            if step % 100 == 0: #每间隔100个batch执行一次打印等操作
                # 调用验证函数
                val_loss, val_rights = evaluation_ConvNet()
                # 统计验证模型时的正确率
                # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                # 统计上面训练模型时的正确率
                # train_r为一个二元组，分别记录目前已经经历过的所有训练集中分类正确的数量和该集合中总的样本数，
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                # 计算并打印出模型在训练时和在验证时的准确率
                # train_r[0]/train_r[1]就是训练集的分类准确度，同样，val_r[0]/val_r[1]就是校验集上的分类准确度
                print(('训练周期: {} [{}/{} ({:.0f}%)]\t训练数据Loss: {:.6f},正确率: {:.2f}%\t校验数据Loss:' +
                    '{:.6f},正确率:{:.2f}%').format(epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(train_loss), 
                                                100. * train_r[0] / train_r[1], 
                                                np.mean(val_loss), 
                                                100. * val_r[0] / val_r[1]))
                #记录中间的数据
                statistics.append({'loss':np.mean(train_loss),'train': 100. * train_r[0] / train_r[1],
                                'valid':100. * val_r[0] / val_r[1]})

        # 产生一组图像保存到 ConvNet 文件夹下（需要事先建立好该文件夹），检测生成器当前的效果
        save_evaluation_samples(netG, 'ConvNet')

    # 训练曲线
    result1 = [100 - i['train'] for i in statistics]
    result2 = [100 - i['valid'] for i in statistics]
    plt.figure(figsize = (10, 7))
    plt.plot(result1, label = 'Training')
    plt.plot(result2, label = 'Validation')
    plt.xlabel('Step')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.show()
# ===============================================================================================
    netG = torch.load('ConvNetG_CPU.mdl')
    netR = torch.load('ConvNetR_CPU.mdl')

    if use_cuda:
        netG = netG.cuda()
        netR = netR.cuda()
    
    #绘制一批样本
    samples = torch.Tensor([0,1,2,3,4,5,6,7,8,9])
    samples = Variable(samples.type(dtype))

    sample_size = 10
    with torch.no_grad():
        samples.resize_(sample_size,1,1,1)
    samples = Variable(samples.data.expand(sample_size, input_dim, 1, 1))
    samples = samples.cuda() if use_cuda else samples
    fake_u = netG(samples)
    fake_u = fake_u.cpu() if use_cuda else fake_u
    samples = samples.cpu() if use_cuda else samples
    img = fake_u #.expand(sample_size, 3, image_size, image_size) #将张量转化成可绘制的图像
    #fig = plt.figure(figsize = (15, 6))
    f, axarr = plt.subplots(2,5, sharex=True, figsize=(15,6))

    for i in range(sample_size):
        axarr[i // 5, i % 5].axis('off')
        imshow(img[i].data, samples.data.numpy()[i][0,0,0].astype(int), axarr[i // 5, i % 5])
    plt.show()

    batch = next(iter(test_loader))
    indx = torch.nonzero(batch[1] == 6)
    data = batch[0][indx[0]]

    img = data.expand(1, 1, image_size, image_size)
    print(img.size())
    plt.axis('off')
    imshow(img[0], 6, plt.gca())

    input_x_real = Variable(data)
    input_x_real = input_x_real.cuda() if use_cuda else input_x_real
    output = netR(input_x_real)
    _, prediction = torch.max(output, 1)
    print('识别器对真实图片的识别结果：', prediction)

    #首先定义读入的图片
    idx = 6
    ax = plt.gca()
    ax.axis('off')
    imshow(fake_u[idx].data, 6, plt.gca())
    print(samples[idx][0])

    #它是从test_dataset中提取第idx个批次的第0个图，其次unsqueeze的作用是在最前面添加一维，
    #目的是为了让这个input_x的tensor是四维的，这样才能输入给net。补充的那一维表示batch。
    input_fake = fake_u[idx]
    input_fake = input_fake.unsqueeze(0)
    input_fake = input_fake.cuda() if use_cuda else input_fake
    output = netR(input_fake)
    _, prediction = torch.max(output, 1)
    print('识别器对生成图片的识别结果：', prediction)
