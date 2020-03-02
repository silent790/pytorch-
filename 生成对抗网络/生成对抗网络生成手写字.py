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
#  ==================================================================================================
# 训练模型:

    print('Initialized!')

    #开始训练
    step = 0 #计数经历了多少时间步
    # num_epochs = 100 #总的训练周期
    num_epochs = 1 # 因训练模型的时间过长，建议在自己的环境中完成完整训练
    record = []
    for epoch in range(num_epochs):
        train_loss = []

        # 加载数据批次
        for batch_idx, (data, target) in enumerate(train_loader):
            # 注意数据中的data转化为了要预测的target，数据中的target则转化成了输入给网络的标签
            target, data = Variable(data), Variable(target) #将Tensor转化为Variable，data为一批图像，target为一批标签
            # 调用模型训练函数，返回损失函数值
            loss = train_ModelG(target, data)
            # 记录损失函数值
            train_loss.append(loss.data.numpy())
            step += 1

            if step % 100 == 0: #每间隔100个batch执行一次打印等操作   
                # 调用模型验证函数，达到验证误差值
                val_loss = evaluation_ModelG()
                # 打印误差等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
                print('训练周期: {} [{}/{} ({:.0f}%)]\t训练数据Loss: {:.6f}\t校验数据Loss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(train_loss), np.mean(val_loss)))
                record.append([np.mean(train_loss), np.mean(val_loss)])

        # 随机选择生成0-9的数字，验证模型的生成结果并保存
        save_evaluation_samples(net, 'MSE')

    plt.figure(figsize = (10, 7))
    plt.plot([i[0] for i in record], label='Training')
    plt.plot([i[1] for i in record], label='Validation')
    plt.xlabel('Batchs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ================================================================================================
    # 绘制一批图像样本
    fake_u = net(samples) #用原始网络作为输入，得到伪造的图像数据
    fake_u = fake_u.cpu() if use_cuda else fake_u
    samples = samples.cpu() if use_cuda else samples
    img = fake_u.data #将张量转化成可绘制的图像
    fig = plt.figure(figsize = (15, 15))
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(8,8, sharex=True, figsize=(15,15))
    for i in range(batch_size):
        axarr[i // 8, i % 8].axis('off')
        imshow(img[i], samples.data.numpy()[i][0,0,0].astype(int),axarr[i // 8, i % 8])
    
    plt.show()
