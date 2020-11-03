import torch.nn as nn
from .gen_resblock import GenBlock
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), n_classes=0):
        super(Generator, self).__init__()
        
        
        b = np.load('ResNetGenerator_850000.npz')
        lst = b.files
        for item in lst:
            print(item, b[item].shape)
        
        
        
        
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = args.gf_dim
        
        print("XXXX", args.latent_dim, self.bottom_width, self.ch)
        
        
        
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch * 16)
        self.block2 = GenBlock(self.ch*16, self.ch*16, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = GenBlock(self.ch*16, self.ch*8, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = GenBlock(self.ch*8, self.ch*4, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = GenBlock(self.ch*4, self.ch*2, activation=activation, upsample=True, n_classes=n_classes)
        self.block6 = GenBlock(self.ch*2, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b7 = nn.BatchNorm2d(self.ch)
        self.l7 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)
        
        
        
        mapping = { 
            
            
            "l1.weight"          :    "l1/W",             #torch.Size([128]) (1024,)
            "l1.bias"          :    "l1/b",             #torch.Size([128]) (1024,)
            
            
            
          
            "block2.c1.bias"          :    "block2/c1/b",             #torch.Size([128]) (1024,)
            "block2.c1.weight"        :    "block2/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block2.c2.bias"          :    "block2/c2/b",             #torch.Size([256]) (1024,)
            "block2.c2.weight"        :    "block2/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block2.b1.bias"          :    "block2/b1/betas/W",             #torch.Size([128]) (1024,)
            "block2.b1.weight"        :    "block2/b1/gammas/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block2.b2.bias"          :    "block2/b2/betas/W",             #torch.Size([256]) (1024,)
            "block2.b2.weight"        :    "block2/b2/gammas/W", 
            
            "block2.c_sc.bias"        :    "block2/c_sc/b",           #torch.Size([256]) (1024,)
            "block2.c_sc.weight"      :    "block2/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "block3.c1.bias"          :    "block3/c1/b",             #torch.Size([128]) (1024,)
            "block3.c1.weight"        :    "block3/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block3.c2.bias"          :    "block3/c2/b",             #torch.Size([256]) (1024,)
            "block3.c2.weight"        :    "block3/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block3.b1.bias"          :    "block3/b1/betas/W",             #torch.Size([128]) (1024,)
            "block3.b1.weight"        :    "block3/b1/gammas/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block3.b2.bias"          :    "block3/b2/betas/W",             #torch.Size([256]) (1024,)
            "block3.b2.weight"        :    "block3/b2/gammas/W", 
            
            "block3.c_sc.bias"        :    "block3/c_sc/b",           #torch.Size([256]) (1024,)
            "block3.c_sc.weight"      :    "block3/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "block4.c1.bias"          :    "block4/c1/b",             #torch.Size([128]) (1024,)
            "block4.c1.weight"        :    "block4/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block4.c2.bias"          :    "block4/c2/b",             #torch.Size([256]) (1024,)
            "block4.c2.weight"        :    "block4/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block4.b1.bias"          :    "block4/b1/betas/W",             #torch.Size([128]) (1024,)
            "block4.b1.weight"        :    "block4/b1/gammas/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block4.b2.bias"          :    "block4/b2/betas/W",             #torch.Size([256]) (1024,)
            "block4.b2.weight"        :    "block4/b2/gammas/W",
            
            "block4.c_sc.bias"        :    "block4/c_sc/b",           #torch.Size([256]) (1024,)
            "block4.c_sc.weight"      :    "block4/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "block5.c1.bias"          :    "block5/c1/b",             #torch.Size([128]) (1024,)
            "block5.c1.weight"        :    "block5/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block5.c2.bias"          :    "block5/c2/b",             #torch.Size([256]) (1024,)
            "block5.c2.weight"        :    "block5/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block5.b1.bias"          :    "block5/b1/betas/W",             #torch.Size([128]) (1024,)
            "block5.b1.weight"        :    "block5/b1/gammas/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block5.b2.bias"          :    "block5/b2/betas/W",             #torch.Size([256]) (1024,)
            "block5.b2.weight"        :    "block5/b2/gammas/W",  
            
            "block5.c_sc.bias"        :    "block5/c_sc/b",           #torch.Size([256]) (1024,)
            "block5.c_sc.weight"      :    "block5/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            
            "block6.c1.bias"          :    "block6/c1/b",             #torch.Size([128]) (1024,)
            "block6.c1.weight"        :    "block6/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block6.c2.bias"          :    "block6/c2/b",             #torch.Size([256]) (1024,)
            "block6.c2.weight"        :    "block6/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block6.b1.bias"          :    "block6/b1/betas/W",             #torch.Size([128]) (1024,)
            "block6.b1.weight"        :    "block6/b1/gammas/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block6.b2.bias"          :    "block6/b2/betas/W",             #torch.Size([256]) (1024,)
            "block6.b2.weight"        :    "block6/b2/gammas/W",
            
            "block6.c_sc.bias"        :    "block6/c_sc/b",           #torch.Size([256]) (1024,)
            "block6.c_sc.weight"      :    "block6/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            "b7.weight"               :    "b7/gamma",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            "b7.bias"                 :    "b7/beta",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "l7.weight"               :    "l7/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            "l7.bias"                 :    "l7/b",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
              
                
                
                
                }
        
        for n,p in self.named_parameters():
            
            try:
                source = mapping[n]
                src = b[source]
                if("betas" in source or "gammas" in source):
                    src = src.mean(0)
                    print(n, source, p.size(), src.shape, type(b[source]))

                    p.data = torch.from_numpy(src).cuda()
                else:
                    print(n, source, p.size(), src.shape, type(b[source]))

                    p.data = torch.from_numpy(src).cuda()
            except:
                pass


    def forward(self, z):

        h = z
        h = self.l1(h).view(-1, self.ch*16, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.b7(h)
        h = self.activation(h)
        h = nn.Tanh()(self.l7(h))
        return h


"""Discriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = 64 #int(args.df_dim)
        
        b = np.load('SNResNetProjectionDiscriminator_850000.npz')
        
#         lst = b.files
#         for item in lst:
#             print(item, b[item].shape)
    
    
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch*2, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch*2, self.ch*4, activation=activation, downsample=True)
        self.block4 = DisBlock(args, self.ch*4, self.ch*8, activation=activation, downsample=True)
        self.block5 = DisBlock(args, self.ch*8, self.ch*16, activation=activation, downsample=True)
        self.block6 = DisBlock(args, self.ch*16, self.ch*16, activation=activation, downsample=False)
        self.l7 = nn.Linear(self.ch*16, 1, bias=True)
        if args.d_spectral_norm:
            self.l7 = nn.utils.spectral_norm(self.l7)
        
        
        mapping = { 
            "block1.c1.bias"          :    "block1/c1/b",             #torch.Size([128]) (1024,)
            "block1.c1.weight_orig"   :    "block1/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block1.c2.bias"          :    "block1/c2/b",             #torch.Size([256]) (1024,)
            "block1.c2.weight_orig"   :    "block1/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block1.c_sc.bias"        :    "block1/c_sc/b",           #torch.Size([256]) (1024,)
            "block1.c_sc.weight_orig" :    "block1/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
          
            "block2.c1.bias"          :    "block2/c1/b",             #torch.Size([128]) (1024,)
            "block2.c1.weight_orig"   :    "block2/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block2.c2.bias"          :    "block2/c2/b",             #torch.Size([256]) (1024,)
            "block2.c2.weight_orig"   :    "block2/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block2.c_sc.bias"        :    "block2/c_sc/b",           #torch.Size([256]) (1024,)
            "block2.c_sc.weight_orig" :    "block2/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "block3.c1.bias"          :    "block3/c1/b",             #torch.Size([128]) (1024,)
            "block3.c1.weight_orig"   :    "block3/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block3.c2.bias"          :    "block3/c2/b",             #torch.Size([256]) (1024,)
            "block3.c2.weight_orig"   :    "block3/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block3.c_sc.bias"        :    "block3/c_sc/b",           #torch.Size([256]) (1024,)
            "block3.c_sc.weight_orig" :    "block3/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "block4.c1.bias"          :    "block4/c1/b",             #torch.Size([128]) (1024,)
            "block4.c1.weight_orig"   :    "block4/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block4.c2.bias"          :    "block4/c2/b",             #torch.Size([256]) (1024,)
            "block4.c2.weight_orig"   :    "block4/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block4.c_sc.bias"        :    "block4/c_sc/b",           #torch.Size([256]) (1024,)
            "block4.c_sc.weight_orig" :    "block4/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            "block5.c1.bias"          :    "block5/c1/b",             #torch.Size([128]) (1024,)
            "block5.c1.weight_orig"   :    "block5/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block5.c2.bias"          :    "block5/c2/b",             #torch.Size([256]) (1024,)
            "block5.c2.weight_orig"   :    "block5/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block5.c_sc.bias"        :    "block5/c_sc/b",           #torch.Size([256]) (1024,)
            "block5.c_sc.weight_orig" :    "block5/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            
            
            "block6.c1.bias"          :    "block6/c1/b",             #torch.Size([128]) (1024,)
            "block6.c1.weight_orig"   :    "block6/c1/W",             #torch.Size([128, 128, 3, 3]) (1024, 1024, 3, 3)
            "block6.c2.bias"          :    "block6/c2/b",             #torch.Size([256]) (1024,)
            "block6.c2.weight_orig"   :    "block6/c2/W",             #torch.Size([256, 128, 3, 3]) (1024, 1024, 3, 3)
            "block6.c_sc.bias"        :    "block6/c_sc/b",           #torch.Size([256]) (1024,)
            "block6.c_sc.weight_orig" :    "block6/c_sc/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            
            
            "l7.weight_orig"          :    "l7/W",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
            "l7.bias"                 :    "l7/b",           #torch.Size([256, 128, 1, 1]) (1024, 1024, 1, 1)
              
                
                
                
                }
        
        for n,p in self.named_parameters():
            
#             try:
            source = mapping[n]

            p.data = torch.from_numpy(b[source]).cuda()
#             print(n, source, p.size(), b[source].shape, type(b[source]))

                        
#             except:
#                 pass
            
            
            
    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l7(h)

        return output
