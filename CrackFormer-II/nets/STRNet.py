import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DWConv(nn.Module):
    def __init__(self, dim_in,stride=1,size=3):
        super(DWConv, self).__init__()
        if size==5:
            self.dwconv = nn.Conv2d(dim_in, dim_in, 5, stride, 2, bias=True, groups=dim_in)
            # self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride, 1, bias=True)
        else:
            self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride, 1, bias=True, groups=dim_in)
            # self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride, 1, bias=True)
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
    
class Concatenate(nn.Module):
    def __init__(self, channel_in):
        super(Concatenate, self).__init__()
        # self.final=Conv3_3(channel_in*6,channel_in)
        self.final=Conv3_3(channel_in*5,1)
        # self.final_conv=Conv3_3(channel_in,1)

    # def forward(self, convx,coarsex,upsamplex,aux_x):
    def forward(self, convx,coarsex,upsamplex):
        # x=torch.cat([convx,coarsex,upsamplex,aux_x],dim=1)
        x=torch.cat([convx,coarsex,upsamplex],dim=1)
        x=self.final(x)
        # x=self.final_conv(x)
        return x
class HSwish(nn.Module):
    def __init__(self,inplace=True):
        super(HSwish,self).__init__()
        self.inplace=inplace
    def forward(self, x):
        if self.inplace:
            x.mul_(F.relu6(x+3)/6)
            return x
        else:
            return x*(F.relu6(x+3)/6)
class HSigmoid(nn.Module):
    def __init__(self):
        super(HSigmoid,self).__init__()
    def forward(self, x):
        return F.relu6(x+3)/6
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        self.theta=torch.ones(1,requires_grad=True)
    def forward(self, x):
        return x*F.sigmoid(self.theta*x)
class STR_module1(nn.Module):
    def __init__(self, channel_in,A,B,fx,dwsize=3):
        super(STR_module1, self).__init__()
        if fx=="relu":
            self.activate=nn.ReLU()
        else:
            # self.activate=HSwish()
            self.activate=Swish()
        self.block3=nn.Sequential(
            PWConv(channel_in,int(channel_in*B)),
            PWConv(int(channel_in*B),int(channel_in*A)),
            nn.BatchNorm2d(int(channel_in*A)),
            self.activate
        )
        self.block4=nn.Sequential(
            DWConv(int(channel_in*A),size=dwsize),
            nn.BatchNorm2d(int(channel_in*A))
        )
        self.block10=nn.Sequential(
            self.activate,
            PWConv(int(channel_in*A),int(channel_in*B)),
            PWConv(int(channel_in*B),channel_in)
        )
    def forward(self,x):
        residual=x
        x=self.block3(x)
        x=self.block4(x)
        x=self.block10(x)
        return x+residual
class Block567(nn.Module):
    def __init__(self, channel_in):
        super(Block567, self).__init__()
        self.block5=nn.AdaptiveAvgPool2d(1)
        self.block6=nn.Linear(channel_in,channel_in//4)
        self.block7=nn.Sequential(
            nn.Linear(channel_in//4,channel_in),
            HSigmoid()
        )
        
    def forward(self,x):
        # print(x.shape)
        residual=x
        x=self.block5(x).squeeze(-1).squeeze(-1)
        x=self.block6(x)
        x=self.block7(x).unsqueeze(-1).unsqueeze(-1)
        return x+residual
class STR_module2(nn.Module):
    def __init__(self, channel_in,A,B,fx,stride=1,dwsize=3):
        super(STR_module2, self).__init__()
        self.channel_out=channel_in
        if fx=="relu":
            self.activate=nn.ReLU()
        else:
            # self.activate=HSwish()
            self.activate=Swish()
        self.block3=nn.Sequential(
            PWConv(channel_in,int(channel_in*B)),
            PWConv(int(channel_in*B),int(channel_in*A)),
            nn.BatchNorm2d(int(channel_in*A)),
            self.activate
        )
        self.block4=nn.Sequential(
            DWConv(int(channel_in*A),stride,dwsize),
            nn.BatchNorm2d(int(channel_in*A))
        )
        self.block567=Block567(int(channel_in*A))
        self.block10=nn.Sequential(
            self.activate,
            PWConv(int(channel_in*A),int(channel_in*B)),
            PWConv(int(channel_in*B),self.channel_out)
        )
        self.residual_conv=nn.Conv2d(channel_in,self.channel_out,3,stride,1)
    def forward(self,x):
        residual=self.residual_conv(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block567(x)*x #block89
        x=self.block10(x)
        return x+residual
class STR_module3(nn.Module):
    def __init__(self, channel_in,A,B,fx,dwsize=3):
        super(STR_module3, self).__init__()
        self.channel_out=channel_in
        if fx=="relu":
            self.activate=nn.ReLU()
        else:
            # self.activate=HSwish()
            self.activate=Swish()
        self.block1=nn.Sequential(
            PWConv(channel_in,int(channel_in*B)),
            nn.BatchNorm2d(int(channel_in*B)),
            self.activate
        )
        self.block2=nn.Sequential(
            DWConv(int(channel_in*B),2,dwsize),
            nn.BatchNorm2d(int(channel_in*B)),
            self.activate,
            PWConv(int(channel_in*B),channel_in)
        )
        self.STR_module2=STR_module2(channel_in,A,B,fx)
        self.pw_final=nn.Sequential(
            PWConv(channel_in+int(channel_in*B),self.channel_out),
            nn.BatchNorm2d(self.channel_out)
        )
    def forward(self,x):
        residual=x
        x_1=x=self.block1(x)
        x=self.block2(x) 
        x_2=self.STR_module2(x)
        x_2=F.upsample(x_2,scale_factor=2,mode='bilinear') 
        x=torch.cat([x_1,x_2],dim=1) #block 11
        x=self.pw_final(x)
        return x+residual
class Attention_decoder(nn.Module):
    def __init__(self,in_channel):
        super(Attention_decoder,self).__init__()
        self.dim=in_channel
        self.conv=nn.Sequential(
            Conv3_3(in_channel,in_channel),
            nn.BatchNorm2d(in_channel)
        )
        self.pw_Q=nn.Sequential(
            PWConv(in_channel,in_channel//2),nn.BatchNorm2d(in_channel//2)
            )
        self.pw_K=nn.Sequential(
            PWConv(in_channel,in_channel//2),nn.BatchNorm2d(in_channel//2)
            )
        self.pw_V=PWConv(in_channel,in_channel//2)
        self.pw_out=PWConv(in_channel//2,in_channel)
        self.pw_out1=nn.Sequential(PWConv(2*in_channel,in_channel),
                                   nn.BatchNorm2d(in_channel),
                                   nn.Dropout(0.2))
        self.tr_conv=nn.ConvTranspose2d(in_channel,in_channel,3,2,1,1)
        # self.tr_conv=nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self, x,x_conv):
        
        x=self.conv(x) 
        q=self.pw_Q(x)  #B C H W
        k=self.pw_K(x)
        v=self.pw_V(x)
        B,C,H,W=q.shape
        q=q.reshape(B,C,-1).permute(0,2,1)  # B H*W C
        v=v.reshape(B,C,-1).permute(0,2,1)  # B H*W C
        mul=torch.einsum("bnc,bchw->bnhw",q,k) #B H*W H W
        M1=torch.einsum('bnhw,bnc->bchw',mul,v)
        M2=F.softmax(M1/math.sqrt(self.dim),dim=1)
        x_ad=self.pw_out(M2)
        # print(x_ad.shape,x_conv.shape)
        x=torch.cat([x_ad,x_conv],dim=1)
        x=self.pw_out1(x)
        out=self.tr_conv(x)
        
        return out
    
class Coarse_Upsample(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(Coarse_Upsample, self).__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv=Conv3_3(channel_in,channel_out)
        self.norm=nn.BatchNorm2d(channel_out)
        self.relu=nn.ReLU()
    def forward(self, x):
        x=self.upsample(x)
        x=self.conv(x)
        x=self.norm(x)
        x=self.relu(x)
        return x
class STR_module_full(nn.Module):
    def __init__(self, channel_in):
        super(STR_module_full, self).__init__()
        self.STR1=STR_module2(channel_in,1,1,'relu',2)
        self.STR2=STR_module1(channel_in,4.5,1,'relu')
        self.STR3=STR_module1(channel_in,5.5,1.5,'relu')
        self.STR4=STR_module2(channel_in,6,2.5,'swish',2,dwsize=5)
        self.STR5=STR_module3(channel_in,15,2.5,'swish',dwsize=5)
        self.STR6=STR_module3(channel_in,15,2.5,'swish',dwsize=5)
        self.STR7=STR_module2(channel_in,7.5,3,'swish',dwsize=5)
        self.STR8=STR_module3(channel_in,9,3,'swish',dwsize=5)
        self.STR9=STR_module2(channel_in,18,6,'swish',2,dwsize=5)
        self.STR10=STR_module3(channel_in,36,6,'swish',dwsize=5)
        self.STR11=STR_module3(channel_in,36,6,'swish',dwsize=5)
        # self.STR9=STR_module2(channel_in,12,6,'swish',2,dwsize=5)
        # self.STR10=STR_module3(channel_in,12,6,'swish',dwsize=5)
        # self.STR11=STR_module3(channel_in,12,6,'swish',dwsize=5)
        self.conv_up=Conv3_3(channel_in,channel_in)
    def forward(self,x):
        x=self.STR1(x)
        x=self.STR2(x)
        out_up=x=self.STR3(x)
        out_up=self.conv_up(out_up)
        x=self.STR4(x)
        x=self.STR5(x)
        x=self.STR6(x)
        x=self.STR7(x)
        out_aux=x=self.STR8(x)
        x=self.STR9(x)
        x=self.STR10(x)
        out=self.STR11(x)
        # return out,out_up,out_aux
        return out,out_up
class STRNet(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(STRNet, self).__init__()
        self.dim=24
        self.stem=nn.Sequential(
            Conv3_3(3,self.dim),
            nn.BatchNorm2d(self.dim),
            HSwish()            
        )
        self.conv_stem=Conv3_3(self.dim,self.dim)
        self.STR=STR_module_full(self.dim)
        self.coarse_up=Coarse_Upsample(self.dim,3*self.dim)
        self.Max_pool=nn.MaxPool2d(2,stride=2)
        self.attention1=Attention_decoder(self.dim)
        self.attention2=Attention_decoder(self.dim)
        # self.attention3=Attention_decoder(self.dim)
        self.Cat=Concatenate(self.dim)
    def forward(self, x):
      
        convx=x=self.conv_stem(self.stem(x))
  
        # x,x_up,x_aux=self.STR(x)
        x,x_up=self.STR(x)
  
        x_coarse=self.attention1(x,x)
        
        x_coarse=self.attention2(x_coarse,x_coarse)
        
        # x_aux=F.upsample_bilinear(self.attention3(x_aux,x_aux),scale_factor=2)
        # x_aux=F.upsample_bilinear(x_aux,scale_factor=4)

        x_coarse=self.coarse_up(x_coarse)

        x_up=F.upsample(x_up,scale_factor=2,mode='bilinear')
        # x=self.Cat(convx,x_coarse,x_up,x_aux)
        x=self.Cat(convx,x_coarse,x_up)
        x=x.squeeze(1)
        return x
