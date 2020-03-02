
net = torch.load('ModelG_CPU.mdl')

if use_cuda:
    net = net.cuda()

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