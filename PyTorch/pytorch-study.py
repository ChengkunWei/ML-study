import torch
import numpy as np
import torch.nn.functional as F #激励函数
from torch.autograd import Variable #变量

import torch.utils.data as Data
import torch.nn as nn
import torchvision


import pdb



# ############# Data convert:  Numpy <=> Tensor ##############################

# # Pytorch 使用的数据是张量 tensor
# # tensor 数据类型和 Numpy 数据类型可以相互转换
# # Variable 是构建图的
# # Variable 是篮子，tensor 是鸡蛋，需要把鸡蛋放在篮子里，才能使用Pytorch的反向传播等功能
# # Variable 包含数据和梯度

# np_data = np.arange(6).reshape(2,3)
# torch_data = torch.from_numpy(np_data) # conver data to tensor in torch
# tensor2array = torch_data.numpy() # convert tersor to numpy

# print(
#     "\n numpy", np_data,
#     "\n torch", torch_data,
#     "\n tensor2array", tensor2array,
# )

# ############ function in torch 

# data = [-1,-2,1,2]
# tensor = torch.FloatTensor(data)

# print(
#     "\n abs",
#     "\n numpy", np.abs(data), 
#     "\n torch", torch.abs(tensor)
# )

# ### matrix multiply
# data = [[1,2],[3,4]]
# tensor = torch.FloatTensor(data)

# print(
#     '\n numpy:', np.matmul(data,data),
#     '\n torch:', torch.mm(tensor,tensor)
# )


# ##########  

# data = [[1,2],[3,4]]
# tensor = torch.FloatTensor(data) # tensor put in to variable
# variable = Variable(tensor, requires_grad =True) # 


# t_out = torch.mean(tensor*tensor)
# v_out = torch.mean(variable*variable)

# print(tensor)
# print('\n variable:', variable,variable.type())

# print(t_out)

# print(v_out)


# v_out.backward()

# print(variable.grad) # variable 含有两类信息，数据（data）和 梯度（grad）

# print(variable.data) 

# print(variable.data.numpy())


# ################# Activate function  ################################

# # 激励函数 确保可微分，掰弯函数曲线，神经网络出来的函数都是线性的关系
# # 卷积神经网络 CNN 推荐使用：relu 
# # 循环神经网络 RNN 推荐使用: relu 和 tanh 
# # 层数少激励函数比较随便，层数多需要好好选择激励函数

# # 一般的激活函数 relu,s sigmod ,tanh, softplus
# # softmax 分类的激励函数


# ################## Regession 回归 ##############################


# x = torch.unsqueeze(torch.linspace(-1,1,100),dim =1)  # x data (tensor)
# y = x.pow(2) + 0.2 * torch.rand(x.size())


# print("this is tensor")
# print(x)
# print(y)

# x , y = Variable(x), Variable(y)

# print("This is variable")
# print(x)
# print(y)

# # torch.nn.Modul 集成net 主模块
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net,self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden,n_output)
    
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x

# net = Net(1, 10, 1)
# print(net)

# optimizer = torch.optim.SGD(net.parameters(), lr=0.5) # net.paramaters() 优化器输入的整个网络的参数
# loss_func = torch.nn.MSELoss()


# for i in range(100):

#     predict = net(x)
#     loss = loss_func(predict,y) #predict 在前，真实数据在后

#     optimizer.zero_grad() # 将梯度全部将为0
#     loss.backward() # 反向传播
#     optimizer.step() # 以学习效率lr 优化梯度

#     print(loss.data.numpy())



# ################## Classification 分类 ##############################



# #pdb.set_trace()
# n_data = torch.ones(100,2)
# x0 = torch.normal(2*n_data,1)          # class0 x data (tensor), shape=(100,2)
# y0 = torch.zeros(100)                  # class0 y data (tensor), shape=(100,1)
# x1 = torch.normal(-2*n_data,1)         # class1 x data (tensor), shape=(100,2)
# y1 = torch.ones(100)                   # class1 y data (tensor), shape=(100,1)

# # 数据合并
# x = torch.cat((x0,x1),0).type(torch.FloatTensor) # FloatTensor = 32-bit floating 
# #标签一定要是LongTensor
# y = torch.cat((y0,y1),).type(torch.LongTensor)   # LongTensor = 64-bit integer

# x,y = Variable(x),Variable(y)

# # torch.nn.Modul 集成net 主模块
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net,self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden,n_output)
    
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x

# net = Net(2, 10, 2)
# print(net)

# optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()

# for t in range(100):
#     prediction = net(x)
#     loss = loss_func(prediction,y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print('Loss；   ', loss.data.numpy())

#     out = torch.max(F.softmax(prediction),1)[1]  # 返回概率最大值的位置[1],返回最大值[0]
#     pred_y = out.data.numpy().squeeze()

#     #pdb.set_trace()
#     target_y = y.data.numpy()
#     accuracy = sum(pred_y==target_y)/200
#     print("Accuracy:   ", accuracy)


############################  快速搭建  ######################################


# net2 = torch.nn.Sequential( # 直接垒积木
#     torch.nn.Linear(2,10),
#     torch.nn.ReLU(),  # 注意与torch.nn.Function.relu()的区别
#     torch.nn.Linear(10,2),
# )
# print(net2)

############################## 提取与保存   ###################################

# x = torch.unsqueeze(torch.linspace(-1,1,100), dim =1)    # x data (tensor),shape = （100，1）
# y = x.pow(2) + 2*torch.rand(x.size())                      # noisy y data (tensor),shape=(100,1)
# x,y = Variable(x,requires_grad = False), Variable(y,requires_grad = False)

# def save():
#     net1= torch.nn.Sequential(
#         torch.nn.Linear(1,10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10,1),
#     )

#     optimizer = torch.optim.SGD(net1.parameters(),lr=0.5)
#     loss_func = torch.nn.MSELoss()

#     for t in range(100):
#         prediction = net1(x)
#         loss = loss_func(prediction,y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


#     torch.save(net1,"net.pkl")  # 保存整个神经网络
#     torch.save(net1.state_dict(), "net_parameters.pkl") # 保存整个 parameters


# def restore_net():
#     net2 = torch.load("net.pkl")


# def restore_params():
#     # 建立一个一模一样的网络

#     net3 = torch.nn.Sequential(
#         torch.nn.Linear(1,10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10,1),
#     )
#     net3.load_state_dict(torch.load("net_parameters.pkl"))   # 快一点


# save()
# restore_net()
# restore_params()

############################## 批训练   ###################################

# # 一小批一小批的训练 minibatch training

# BATCH_SIZE = 8  # 每一批多少个


# x = torch.linspace(1,10,10)
# y = torch.linspace(10,1,10)

# #torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y) 
# torch_dataset = Data.TensorDataset(x,y) 
# loader = Data.DataLoader(  #定义loader
#     dataset = torch_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle =False, # 是否要打乱
#     num_workers = 2, # 用几个线程或进程提取数据
# )

# for epoch in range(3): # 训练3次
#     for step, (batch_x, batch_y) in enumerate(loader):
#         # training 
#         print('Epoch: ', epoch, '| Step: ',step,'|batch x: ',
#         batch_x.numpy(), '| batch_y:', batch_y.numpy())

# ############################## 优化器  ###################################


# # hyper parameters
# LR = 0.01
# BATCH_SIZE = 32
# EPOCH = 12

# x = torch.unsqueeze(torch.linspace(-1,1,100),dim =1)  # x data (tensor)
# y = x.pow(2) + 0.2 * torch.rand(x.size())


# torch_dataset = Data.TensorDataset(x,y)
# loader = Data.DataLoader(
#     dataset = torch_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle = True,
#     num_workers =2,
# )

# class Net(torch.nn.Module):
#     def __init__(self, n_feature=1, n_hidden=20, n_output=1):
#         super(Net,self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden,n_output)
    
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x


# net_SGD = Net()
# net_Momentum = Net()
# net_RMSprop = Net()
# net_Adam = Net()

# nets = [net_SGD, net_Momentum,net_RMSprop,net_Adam]

# opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
# opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
# opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR, alpha=0.9)
# opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))

# optimizers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]


# loss_func = torch.nn.MSELoss()
# losses_his = [[],[],[],[]]

# for epoch in range(EPOCH):
#     print(epoch)
#     for step,(batch_x,batch_y) in enumerate(loader):
#         b_x = Variable(batch_x)
#         b_y = Variable(batch_y)

#         for net, opt, l_his in zip(nets, optimizers,losses_his):
#             output = net(b_x)
#             loss = loss_func(output,b_y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             l_his.append(loss.data.numpy())

        
# print('SGD_loss   ', '   Momentum_loss   ','   RMSprop_loss   ','   Adam_loss   ')
# for SGD_loss, Momentum_loss, RMSprop_loss, Adam_loss in zip(losses_his[0],losses_his[1],losses_his[2],losses_his[3]):
#     print(SGD_loss,'   ' ,Momentum_loss,'   ',RMSprop_loss,'   ','   ',Adam_loss)


# ############################## 卷积神经网络 ###################################
# 由神经层组成 
# 卷积 不是对每一个像素做处理，而是对每一个小块图像做处理，加强了图片信息的连续性，同时加深了对图片的理解
# 每次卷积搜集小的像素块 -> 生成一个,高度更高，长和宽压缩的图片，对图片更深的理解
# 每层卷积，会对更大的区域进行识别
# 
# 池化：可以提高精度
# 池化的作用:每次卷积会无意的丢掉一些信息，pooling 主要解决这个问题
# 也就是说，在卷积的时候不压缩长和宽，压缩的工作交给池化 pooling
#
#
######常用神经网络结构#########
#
# image -> convolution -> max pooling -> convolution -> max pooling -> Fully connected -> Fully connected -> Classification


# hyper parameters

# EPOCH = 5
# BATCH_SIZE = 50
# LR = 0.01
# DOWNLODA_MNIST = True

# # train data
# train_data = torchvision.datasets.MNIST(
#     root = './mnist',
#     train = True,
#     transform = torchvision.transforms.ToTensor(), # (0,1) (0,255)
#     download = DOWNLODA_MNIST,
# )

# print(train_data.train_data.size())
# print(train_data.train_labels.size())


# train_loader = Data.DataLoader(dataset=train_data, batch_size = BATCH_SIZE,shuffle=True, num_workers=2)

# # test data
# test_data = torchvision.datasets.MNIST(
#     root = './mnist',
#     train = False,
# )

# test_x = Variable(torch.unsqueeze(test_data.test_data,dim =1), volatile = True).type(torch.FloatTensor)[:2000].cuda()/255
# test_y = test_data.test_labels[:2000].cuda()

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(   # (1,28,28) 的图片
#                 in_channels = 1,   # 输入的图片高度
#                 out_channels = 16, # 输出的高度，16个filter,16个特征，图片把高度变为16
#                 kernel_size = 5,   # 5*5的区域
#                 stride = 1,  # 每隔多少步，跳一下
#                 padding = 2, # 图片的补全 if stride=1, pad = (kernel_size -1) /2 = (5-1)/2
#             ),    # ->(16,28,28) 的图片
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2), # 筛选出重要的信息 变成 -> （16，14，14）
#         )
#         self.conv2 = nn.Sequential( # 输入 ->（16，14，14）
#             nn.Conv2d(
#                 in_channels = 16,   # 输入的图片高度
#                 out_channels = 32, # 输出的高度，16个filter,16个特征，图片把高度变为16
#                 kernel_size = 5,   # 5*5的区域
#                 stride = 1,  # 每隔多少步，跳一下
#                 padding = 2, # 图片的补全 if stride=1, pad = (kernel_size -1) /2 = (5-1)/2
#             ), # 输出 ->（32，14，14）
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2), # 筛选出重要的信息 # 输出 ->（32，7，7）
#         )

#         self.out = nn.Linear(32*7*7, 10) # 10分类
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.conv2(x) # (batch, 32,7,7)
#         x = x.view(x.size(0),-1)  # (batch, 32*7*7) 数据展平
#         output = self.out(x)
#         return output

# cnn = CNN()
# cnn.cuda()
# print(cnn)


# optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) #optimize all CNN parameters
# loss_func = nn.CrossEntropyLoss()                    # the target label is not one-hot


# for epoch in range(EPOCH):
#     for step,(x,y) in enumerate(train_loader):
#         # GPU in cuda data to gpu
#         b_x = Variable(x).cuda() # batch x
#         b_y = Variable(y).cuda() # batch y

#         output = cnn(b_x) # cnn output
#         loss = loss_func(output,b_y) # cross entropy loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if step % 50  == 0:
#             test_output = cnn(test_x)
#             pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
#             accuracy = sum(pred_y.cpu().data.numpy()==test_y.cpu().data.numpy())/test_y.size(0)
#             #accuracy = sum(pred_y.data ==test_y.data) /test_y.size(0)
#             print('Epoch:  ',epoch,'|  train loss: %.4f' % loss.cpu().data.numpy(),'|  test accuracy: ', accuracy)

# # print 10 predictions from test data
# test_output = cnn (test_x[:10])
# pred_y = torch.max(test_output,1)[1].data.cpu().numpy().squeeze()
# print(pred_y,'prediction number')
# print(test_y[:10].cpu().numpy(),'real number')


################## GPU 加速###################################################
#
# 移动数据
# 移动CNN网络
# 移动计算图纸
# 移动到 GPU  .cuda()
# 移动到 CPU   .cpu()


#################################   RNN   ###########################################
# 序列化神经网络，神经网络具有记住之前数据的功能
# 
# 分类
# 回归
###### 普通RNN弊端
#
# 健忘->梯度消失，梯度弥散，梯度爆炸
####################   LSTM   #########################################################
# LSTM比普通RNN,  多了三个控制器: 输入控制，输出控制，忘记控制 
#
# 主线剧情，分线剧情
#



#################################   RNN 分类  ###########################################

# # Hyper parameters
# EPOCH = 1
# BATCH_SIZE = 64
# TIME_STEP = 28                 # rnn time step/image height, 考虑多少时间点的数据
# INPUT_SIZE = 28                # rnn input size /image width
# LR = 0.01
# DOWNLODA_MNIST = False


# train_data = torchvision.datasets.MNIST(
#     root='./mnist',
#     train = True,
#     transform = torchvision.transforms.ToTensor(),
#     download = DOWNLODA_MNIST,
# )

# train_loader = torch.utils.data.DataLoader(
#     dataset=train_data,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     )


# test_data = torchvision.datasets.MNIST(root='./mnist',train=False,transform=torchvision.transforms.ToTensor())
# test_x = Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:2000].cuda()/255.
# test_y = test_data.test_labels[:2000].cuda()

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN,self).__init__()

#         self.rnn = nn.LSTM(
#             input_size=INPUT_SIZE,
#             hidden_size=64,
#             num_layers=1,  #私有细胞的数量，越大能力越强
#             batch_first=True,
#         )
#         self.out = nn.Linear(64,10)

#     def forward(self,x):
#         r_out, (h_n,h_c) = self.rnn(x,None)  # x (batch, time_step, input_size) None,最初的hidden state
#         # hidden_n 分线程hidden state, hidden_c 主线程hidden state
#         out = self.out(r_out[:,-1,:]) # (batch, time step, input) 选择最后一个维度的 out
#         return out

# rnn = RNN()
# rnn.cuda()
# print(rnn)


# optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
# loss_func = nn.CrossEntropyLoss()  # 数字标签

# for epoch in range(EPOCH):
#     for step,(x,y) in enumerate(train_loader):
#         b_x = Variable(x.view(-1,28,28)).cuda()           # reshape x to (batch, time_step,input_size)
#         b_y = Variable(y).cuda()
#         output = rnn(b_x)

#         loss = loss_func(output,b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if step % 50 == 0:
#             test_output = rnn(test_x)
#             pred_y = torch.max(test_output,1)[1].cpu().data.numpy().squeeze()
#             #pdb.set_trace()
#             #accuracy = sum(pred_y == test_y) / test_y.size
#             accuracy = sum(pred_y ==test_y.cpu().data.numpy())/test_y.size(0)
#             print('Epoch:  ',epoch,'| train loss: %.4f' % loss.cpu().data.numpy(),'| test accuracy: ', accuracy)


# # print 10 prediction from test data

# test_output = rnn(test_x[:10].view(-1,28,28))
# pred_y = torch.max(test_output,1)[1].cpu().data.numpy().squeeze()
# print(pred_y,'prediction number')
# print(test_y[:10].cpu().data.numpy(),'real number')



#################################   RNN 回归  ###########################################
#
#
#
####### 例子，sin 预测 cos
#

# #torch.manual_seed(1)  # reproducible

# # Hyper Parameters
# TIME_STEP = 10   # run time step
# INPUT_SIZE = 1   # rnn input size, 每个时间点输入一个数据
# LR = 0.02        # learning rate

# # show data
# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(steps)  # float32 for converting torch FloatTensor
# y_np = np.cos(steps)

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()

#         self.rnn = nn.RNN(
#             input_size=INPUT_SIZE,
#             hidden_size=32,        # rnn hidden unit， 隐藏神经元
#             num_layers=1,          # number of rnn layer
#             batch_first=True,      # input & output will has batch size as ls dimension. e.g.,(batch, time_step, input_size)
#         )
#         self.out = nn.Linear(32,1)  # 输出一个时间点上的y坐标
    
#     def forward(self,x,h_state):
#         # x (batch, time_step, input_size)
#         # h_state (n_layers, batch, hidden_size)
#         # r_out (batch, time_step, hidden_size)
#         r_out, h_state = self.rnn(x, h_state)    # 输入x 数据, hidden_state 记忆  


#         outs = []  # 每一次加工的产物
#         for time_step in range(r_out.size(1)):    # calculate output for each time step
#             outs.append(self.out(r_out[:,time_step,:]))
#         return torch.stack(outs, dim=1),h_state

#         # # instead, for simplicity, you can replace above codes by follows
#         # r_out = r_out.view(-1,32)
#         # outs = self.out(r_out)
#         # outs = outs.view(-1, TIME_STEP, 1)
#         # return outs, h_state

#         # # or even simpler,since nn.Linear can accept inputs of any dimension 
#         # # and returns outputs with the same dimension except for the last
#         # outs = self.out(r_out)
#         # return outs


# rnn = RNN()
# print(rnn)

# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  #optimize all rnn parameters
# loss_func = nn.MSELoss()

# h_state = None  # for initial hidden state

# for step in range(100):
#     start, end = step * np.pi, (step+1)*np.pi  # time range
#     # use sin predicts cos
#     steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False) # float32 for converting torch FloatTensor
#     x_np = np.sin(steps)
#     y_np = np.cos(steps)

#     x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    # shape (batch,time_step, input_size)
#     y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

#     prediction, h_state = rnn(x, h_state)                    # rnn output
#     #!! next step is important !!!
#     h_state = Variable(h_state.data)             # repack the hidden state, break the connection from last iteration

#     loss = loss_func(prediction,y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

    
#     print('Step: ', step,'   Loss: ', loss.data.numpy())


################################### 自编码  #################################################
#    非监督学习
#
###########  编码-〉解码
#
#  自编码超越了PCA

# # Hper Parameters
# EPOCH = 10
# BATCH_SIZE = 64
# LR = 0.005
# DOWNLODA_MNIST = False
# N_TEST_IMG = 5


# # MNIST digits dataset
# train_data = torchvision.datasets.MNIST(
#     root='./mnist',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=DOWNLODA_MNIST,
# )

# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



# # 定义神经网络结构
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder,self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(28*28,128),
#             nn.Tanh(),
#             nn.Linear(128,64),
#             nn.Tanh(),
#             nn.Linear(64,12),
#             nn.Tanh(),
#             nn.Linear(12,3),        # compress to 3 feature which can be visualized in plt
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(3,12),
#             nn.Tanh(),
#             nn.Linear(12,64),
#             nn.Tanh(),
#             nn.Linear(64,128),
#             nn.Tanh(),
#             nn.Linear(128,28*28),
#             nn.Sigmoid(),           # compress to a range (0,1)
#         )

#     def forward(self,x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


# autoencoder = AutoEncoder().cuda()
# optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
# loss_func = nn.MSELoss()

# for epoch in range(EPOCH):
#     for step, (x,b_label) in enumerate(train_loader):
#         b_x = Variable(x.view(-1, 28*28)).cuda()  # batch x, shape (batch, 28*28)
#         b_y = Variable(x.view(-1, 28*28)).cuda()  # batch y, shape (batch, 28*28)

#         encoded, decoded = autoencoder(b_x)

#         loss = loss_func(decoded, b_y)         # mean square error
#         optimizer.zero_grad()                  # clear gradients for this training step
#         loss.backward()                        # back propagation, compute gradients
#         optimizer.step()                       # apply gradients

#         if step % 100 == 0:
#             print('Epoch: ', epoch, '| Train Loss: %.4f' % loss.cpu().data.numpy())

################################### DQN 强化学习  #################################################
#
#
#  神经网络 + Q-learning   留待需要的时候再学习
#
#  直接使用神经网络的

#import gym

################################### GAN 对抗神经网路  ###############################################################
#
#
# 凭空捏造结果  
#


# # Hyper Parameters
# BATCH_SIZE = 64
# LR_G = 0.0001  # learning rate for generator
# LR_D = 0.0001  # learning rate for discriminator
# N_IDEAS = 5    # think of this as number of ideas for generating an art work (Generator)
# ART_COMPONENT = 15 # it could be total point G can draw in the canvas
# PAINT_POINTS = np.vstack([np.linspace(-1,1,ART_COMPONENT) for _ in range(BATCH_SIZE)])


# def artist_works():   # painting from the famous artist(real target)
#     a = np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
#     paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
#     paintings = torch.from_numpy(paintings).float()
#     return paintings


# G = nn.Sequential(  # Generator
#     nn.Linear(N_IDEAS,128),  # random ideas from normal distribution
#     nn.ReLU(),
#     nn.Linear(128,ART_COMPONENT)  # making a painting from these random ideas
# )


# D = nn.Sequential(                      # Discriminator
#     nn.Linear(ART_COMPONENT, 128),      # receive art work either from the famous artist or a newbie like G
#     nn.ReLU(),
#     nn.Linear(128,1),
#     nn.Sigmoid(),                       # tell the probability that the art work is made by artist
# )

# opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
# opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)


# for step in range(10000):
#     artist_paintings = Variable(artist_works())                   # real painting from artist
#     G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))          # random ideas
#     G_paintings = G(G_ideas)                             # fake painting from G (random ideas)

#     prob_artist0 = D(artist_paintings)                   # D try to increase this prob
#     prob_artist1 = D(G_paintings)                        # D try to reduce this prob

#     D_loss = - (torch.mean(torch.log(prob_artist0)) + torch.mean(torch.log(1. - prob_artist1)))
#     G_loss = torch.mean(torch.log(1. - prob_artist1))

#     opt_D.zero_grad()
#     D_loss.backward(retain_graph=True)  # reusing computational graph
#     opt_D.step()


#     opt_G.zero_grad()
#     G_loss.backward()
#     opt_G.step()

#     if step % 1000 == 0:
#         print('D loss:', D_loss.data.numpy(), '| G loss:  ',G_loss.data.numpy())

##################################  Pytorch 动态  ###############################################################
# time_step 也是变化的
# 
# 
# 
# 
# 
#     dynamice_rnn() 函数 ， 不同长度的  time_step
#     构建动态图纸
#


##################################  批标准化  #####################################################################
#
#
# 对每一层进行 normalization 
# 这一层插入在 全连结层之后-〉 Normalization-〉激活函数之前
