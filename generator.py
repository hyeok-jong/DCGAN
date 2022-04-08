import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tc1 = self.make_trans_conv(100, 64*8, 4, 1, 0)
        self.tc2 = self.make_trans_conv(64*8, 64*4, 4, 2, 1)
        self.tc3 = self.make_trans_conv(64*4, 64*2, 4, 2, 1)
        self.tc4 = self.make_trans_conv(64*2, 64*1, 4, 2, 1)
        self.tc5 = self.make_trans_conv(64*1, 3, 4, 2, 1, True)

    def make_trans_conv(self, in_c, out_c, k, s, p, last = False):
        layers = list()
        layers.append(
            nn.ConvTranspose2d(
                in_channels = in_c,
                out_channels = out_c,
                kernel_size = k,
                stride = s,
                padding = p,
                bias = False
            )
        )

        if last == False:
            layers.append(nn.BatchNorm2d(num_features = out_c))
            layers.append(nn.ReLU(inplace = False))             # What is inplace option ? Normally in_place means it change itself. e.g) pandas. See datails link below
                                                               # https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        
        else:
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self,x):
        
        x = self.tc1(x)
        x = self.tc2(x)
        x = self.tc3(x)
        x = self.tc4(x)
        x = self.tc5(x)

        return x



if __name__ == "__main__":

    import torch
    from torchinfo import summary

    generator = Generator()

    print(summary(generator, (64, 100,1,1)))