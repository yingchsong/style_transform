import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms import ToTensor,transforms
from torchvision import models

# 读取图片
region = cv2.imread('dog.jpg')
style = cv2.imread('sky.jpg')


#修剪图片
region=cv2.resize(region[800:2400,:],(256,256))
style= cv2.resize(style,(256,256))
input = region[:]

#调用一个预训练过的VGG19神经网络
class VGG(nn.Module):
    def __init__(self) -> None:
        super(VGG,self).__init__()
        self.layers = ['0','5','10','19','28']
        self.vgg  = models.vgg19(pretrained =  True).features
    
    #这里不需要返回输出值，只返回规定层即可
    def forward(self,x):
        res=[]
        for name,layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                res+=[x]

        return res
    
#自定义一个CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers=[3,9,15,21,24]
        self.features = nn.Sequential(

        
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) ,#16*128*128
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) ,#32*64*64,


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #64*32*32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #128*16*16
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #256*8*8
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) ,#256*4*4

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #512*2*2
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #1024*1*1

        )
        
        
    #这里不需要返回输出值，只返回规定层即可
    def forward(self, x):
        res=[]
        for i in range(24):
            x = self.features[i](x)
            if (i+1) in  self.layers:
                res+=[x]

        return res


transform = transforms.Compose([
    
    transforms.ToTensor(),  # 将图像转换为Tensor
     # 标准化
])
#升维，把三维图片变成（batch_size,channels_nums,w,h)符合神经网络训练格式的tensor
input = transform(input).unsqueeze(0)
region = transform(region).unsqueeze(0)
style = transform(style).unsqueeze(0)

#调用模型
cnn=VGG()
#设置一个用来训练的tensor
a  = torch.ones(1,3,256,256)
#设置梯度
a.requires_grad_(True)
#优化器
optimizer = torch.optim.Adam([a],lr=0.003)


#迭代步数
step = 2001
#训练
for i in range(step):
    #分别将白噪声图片（这里使用原始图片），原始图片，风格图片导入神经网络
    input_features = cnn(input*a)
    region_features = cnn(region)
    style_features = cnn(style)
    #每一个轮次开始前设定内容损失和风格损失为0
    content_loss=0
    style_loss=0
    #zip将三个列表压缩成一个列表，其中f1,f2,f3为三个特征，循环一共五次
    for f1,f2,f3 in zip(input_features,region_features,style_features):
        
        content_loss = torch.mean((f1-f2)**2)+content_loss
        #取得参数
        _ , c , w , h =f1.size()
        #展开
        f1 = f1.view(c,w*h)
        f3 = f3.view(c,w*h)

        #求garm矩阵
        f1 = torch.mm(f1,f1.T)
        f3 = torch.mm(f3,f3.T)

        #求每层的损失
        style_loss = torch.sum((f1-f3)**2)/(4*c*w*h)+style_loss
    

    #内容损失和风格损失前有系数α和β，分别为1，100
    loss = content_loss+100*style_loss
    #结束后梯度设为0
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()

    print("STEP:{}/{},loss: {}".format(i+1,step,loss.item()))
    #每隔一定数量输出结果
    if i%400==0:
        img = (input*a).squeeze()
        img = img.detach().numpy().T
        img  = np.rot90(img,k=3)
        np.savetxt('new'+str(i)+'.csv',img.reshape(256*256,3),delimiter=',')
        cv2.imwrite('new'+str(i)+'.jpg',img*255)