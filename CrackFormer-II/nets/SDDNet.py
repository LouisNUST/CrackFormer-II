import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = PWConv(in_channel, depth)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2,groups=in_channel)
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3,groups=in_channel)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4,groups=in_channel)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        # image_features = self.mean(x)
        image_features = F.relu(self.conv(x))
        # image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = F.relu(self.atrous_block1(x))
 
        atrous_block2 = F.relu(self.atrous_block2(x))
 
        atrous_block3 = F.relu(self.atrous_block3(x))
 
        atrous_block4 = F.relu(self.atrous_block4(x))
 
        net = F.relu(self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block2,
                                              atrous_block3, atrous_block4], dim=1)))
        return net

class DWConv(nn.Module):
    def __init__(self, dim_in,stride=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride,1,groups=dim_in,bias=True)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = self.dwconv(x)
        x=self.relu(x)
        return x
class PWConv(nn.Module):
    def __init__(self, dim_in,dim_out):
        super(PWConv, self).__init__()
        self.pwconv = nn.Conv2d(dim_in, dim_out,1,1)

    def forward(self, x):
        x = self.pwconv(x)
        return x  
class Conv3_3(nn.Module):
    def __init__(self, dim_in,dim_out):
        super(Conv3_3, self).__init__()
        self.Conv = nn.Conv2d(dim_in, dim_out,3,1,1)

    def forward(self, x):
        x = self.Conv(x)
        return x      
class DepSep(nn.Module):
    def __init__(self, dim_in,stride=1):
        super(DepSep, self).__init__()
        self.pw1=PWConv(dim_in,int(0.5*dim_in))
        self.dw1=DWConv(int(0.5*dim_in))
        self.norm1=nn.BatchNorm2d(int(1.5*dim_in))
        self.pw2=PWConv(int(1.5*dim_in),int(0.5*dim_in))
        self.dw2=DWConv(int(0.5*dim_in))
        self.norm2=nn.BatchNorm2d(int(2*dim_in))
        self.pw3=PWConv(int(2*dim_in),int(0.5*dim_in))
        self.dw3=DWConv(int(0.5*dim_in))
        self.norm3=nn.BatchNorm2d(int(2.5*dim_in))
        
        self.pw4=PWConv(int(dim_in*2.5),dim_in)
        self.dw4=DWConv(dim_in,stride)
        self.residual_conv=nn.Conv2d(dim_in,dim_in,3,stride=stride,padding=1)
    def forward(self, x):
        input_x=x
        x=self.pw1(x)
        pw1=x=self.dw1(x)
        x=torch.cat([input_x,pw1],dim=1)
        x=self.norm1(x)
        x=self.pw2(x)
        pw2=x=self.dw2(x)
        x=torch.cat([input_x,pw1,pw2],dim=1)
        x=self.norm2(x)
        x=self.pw3(x)
        pw3=x=self.dw3(x)
        x=torch.cat([input_x,pw1,pw2,pw3],dim=1)
        x=self.norm3(x)
        x=self.pw4(x)
        x=self.dw4(x)
        output=x+self.residual_conv(input_x)
        return output
        # return x
    
class Decoder(nn.Module):
    def __init__(self, dim_in,class_num):
        super(Decoder, self).__init__()  
        # self.upsample=nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample=nn.Upsample(scale_factor=4,mode='bilinear')
        self.pw_densep=PWConv(dim_in,int(0.5*dim_in))
        self.pw2=PWConv(int(1.5*dim_in),dim_in)
        self.dw1=DWConv(dim_in)
        self.pw3=PWConv(dim_in,int(0.5*dim_in))
        self.dw2=DWConv(int(0.5*dim_in))
        # self.upsample1=nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.final=Conv3_3(int(0.5*dim_in)+16,class_num)
        self.pw_conv=PWConv(64,16)
        
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x,DenSepX,ConvX):
        
        x=self.upsample(x)
        DenSepX=self.pw_densep(DenSepX)
        x=torch.cat([x,DenSepX],dim=1)
        
        x=self.pw2(x)
        x=self.dw1(x)
        x=self.pw3(x)
        x=self.dw2(x)
        x=self.upsample1(x)
        ConvX=self.pw_conv(ConvX)
        x=torch.cat([x,ConvX],dim=1)
        x=self.final(x)
        # x=self.softmax(x)
        return x
class SDDNet(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(SDDNet, self).__init__()
        self.dim=64
        self.conv1=nn.Conv2d(channel_in,64,3,1,1)
        
        self.conv2=nn.Conv2d(64,self.dim,3,2,1)
        
        # self.conv2=nn.Conv2d(64,self.dim,3,1,1)
        
        self.DenSep1=DepSep(self.dim,2)
        self.DenSep2=DepSep(self.dim,2)
        self.DenSep3=DepSep(self.dim,2)
        self.DenSep4=nn.Sequential(*[DepSep(self.dim),DepSep(self.dim),DepSep(self.dim),DepSep(self.dim),DepSep(self.dim),DepSep(self.dim)])
        self.ASPP=ASPP(self.dim,self.dim)
        self.decoder=Decoder(self.dim,channel_out)
    def forward(self, x):
        x = self.conv1(x)
        ConvX=x
        x = self.conv2(x)
        x = self.DenSep1(x)
        DenSepX=x
        x = self.DenSep2(x)
        x = self.DenSep3(x)
        x = self.DenSep4(x)
        x = self.ASPP(x)
        x = self.decoder(x,DenSepX,ConvX)
        x=x.squeeze(1)
        return x
    
