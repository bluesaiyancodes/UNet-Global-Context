import torch
import torch.nn as nn
import torch.nn.functional as F

from models.init_weights import init_weights

class UNetWithoutSkip(nn.Module):
    def __init__(self):
        super(UNetWithoutSkip, self).__init__()

        # feature channels
        n = 16
        filter = [n, n*2, n*4, n*8, n*16]

        
        # Encoder
        self.enc1 = nn.Conv2d(1, filter[0], 3, padding=1)
        self.enc2 = nn.Conv2d(filter[0], filter[1], 3, padding=1)
        self.enc3 = nn.Conv2d(filter[1], filter[2], 3, padding=1)
        self.enc4 = nn.Conv2d(filter[2], filter[3], 3, padding=1)
        # Global Context
        self.global_context = nn.Conv2d(filter[3], filter[4], 3, padding=1)
        # Decoder
        self.dec4 = nn.Conv2d(filter[4], filter[3], 3, padding=1)
        self.dec3 = nn.Conv2d(filter[3], filter[2], 3, padding=1)
        self.dec2 = nn.Conv2d(filter[2], filter[1], 3, padding=1)
        self.dec1 = nn.Conv2d(filter[1], filter[0], 1)
        self.final = nn.Conv2d(filter[0], 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
        
    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))
        e3 = F.relu(self.enc3(F.max_pool2d(e2, 2)))
        e4 = F.relu(self.enc4(F.max_pool2d(e3, 2)))
        g = F.relu(self.global_context(F.max_pool2d(e4, 2)))

        d4 = F.relu(self.dec4(g))
        d3 = F.relu(self.dec3(d4))
        d2 = F.relu(self.dec2(d3))
        d1 = F.relu(self.dec1(d2))
        
        out = self.final(F.interpolate(d1, x.shape[2:]))  
        return out, g