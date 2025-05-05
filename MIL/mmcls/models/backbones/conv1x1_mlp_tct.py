import torch.nn as nn

from ..builder import BACKBONES
 
 
@BACKBONES.register_module()
class MLP_TCT(nn.Module):
 
    def __init__(self, in_channels, input_num,  output_num=2, dropout=0.):
        super(MLP_TCT, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_num, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_num),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):  # should return a tuple
        B, C, H, W = x.shape
        x = self.net1(x)
        x = x.view(B, -1)
        return self.net2(x)
 
    def init_weights(self, pretrained=None):
        pass 
       