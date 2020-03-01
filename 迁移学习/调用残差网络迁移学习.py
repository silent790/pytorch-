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

    # 创建相应的数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle = True, num_workers=4)

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
        inp = inp.cpu().cpu().numpy().transpose((1, 2, 0)) 

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

    def rightness(predictions, labels):
        # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
        pred = torch.max(predictions.data, 1)[1] 
        # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
        rights = pred.eq(labels.data.view_as(pred)).sum() 
        # 返回正确的数量和这一次一共比较了多少元素
        return rights, len(labels)
    # ===============================================================================

    # 加载残差网络
    torch.utils.model_zoo.load_url('http://labfile.oss.aliyuncs.com/courses/1073/resnet18-5c106cde.pth')
        # 加载模型库中的residual network，并设置pretrained为true，这样便可加载相应的权重
    net = models.resnet18(pretrained=False)
        #如果存在GPU，就将网络加载到GPU上
    net = net.cuda() if use_cuda else net
        # 将网络的架构打印出来
    print(net)
    # ===================================================================================================
    # 读取最后线性层的输入单元数，这是前面各层卷积提取到的特征数量
    num_ftrs = net.fc.in_features

    # 重新定义一个全新的线性层，它的输出为2，原本是1000
    net.fc = nn.Linear(num_ftrs, 2)

    #如果存在GPU则将网络加载到GPU中
    net.fc = net.fc.cuda() if use_cuda else net.fc

    criterion = nn.CrossEntropyLoss() #Loss函数的定义
    # 将网络的所有参数放入优化器中
    optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9)
    # 预训练方法
    record = [] #记录准确率等数值的容器

    #开始训练循环
    num_epochs = 20
    net.train(True) # 给网络模型做标记，标志说模型在训练集上训练
    best_model = net
    best_r = 0.0
    for epoch in range(num_epochs):
        #optimizer = exp_lr_scheduler(optimizer, epoch)
        train_rights = [] #记录训练数据集准确率的容器
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
            data, target = Variable(data), Variable(target) #将Tensor转化为Variable，data为图像，target为标签
            #如果存在GPU则将变量加载到GPU中
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data) #完成一次预测
            loss = criterion(output, target) #计算误差
            optimizer.zero_grad() #清空梯度
            loss.backward() #反向传播
            optimizer.step() #一步随机梯度下降
            right = rightness(output, target) #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
            train_rights.append(right) #将计算结果装到列表容器中
            loss = loss.cpu() if use_cuda else loss
            train_losses.append(loss.data.cpu().numpy())


            #if batch_idx % 20 == 0: #每间隔100个batch执行一次
        #train_r为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

        #在测试集上分批运行，并计算总的正确率
        net.eval() #标志模型当前为运行阶段
        test_loss = 0
        correct = 0
        vals = []
        #对测试数据集进行循环
        for data, target in val_loader:
            #如果存在GPU则将变量加载到GPU中
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=True), Variable(target)
            output = net(data) #将特征数据喂入网络，得到分类的输出
            val = rightness(output, target) #获得正确样本数以及总样本数
            vals.append(val) #记录结果

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


    x = [x[0] for x in record]
    y = [1 - x[1] for x in record]
    z = [1 - x[2] for x in record]
    #plt.plot(x)
    plt.figure(figsize = (10, 7))
    plt.plot(y)
    plt.plot(z)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')

    def visualize_model(model, num_images=6):
        images_so_far = 0
        fig = plt.figure(figsize=(15,10))

        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            preds = preds.cpu().cpu().numpy() if use_cuda else preds.cpu().numpy()
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot( 2,num_images//2, images_so_far)
                ax.axis('off')

                ax.set_title('predicted: {}'.format(val_dataset.classes[preds[j]]))
                imshow(data[0][j])

                if images_so_far == num_images:
                    return
    visualize_model(net)

    plt.ioff()
    plt.show()

