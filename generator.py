import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, resolution):
        super().__init__()

        self.resolution = resolution
        if resolution == "low":
            self.tc4_1 = self.make_trans_conv(64*1, 64*1, 4, 2, 1)
        elif resolution == "mid":
            self.tc3_1 = self.make_trans_conv(64*2, 64*2, 4, 2, 1)
        elif resolution == "high":
            self.tc2_1 = self.make_trans_conv(64*4, 64*4, 4, 2, 1)
        elif resolution == "highest":
            self.tc1_1 = self.make_trans_conv(64*8, 64*8, 4, 2, 1)

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
        if self.resolution == "highest":
            x = self.tc1_1(x)
        x = self.tc2(x)
        if self.resolution == "high":
            x = self.tc2_1(x)
        x = self.tc3(x)
        if self.resolution == "mid":
            x = self.tc3_1(x)
        x = self.tc4(x)
        if self.resolution == "low":
            x = self.tc4_1(x)
        x = self.tc5(x)

        return x



if __name__ == "__main__":

    from torchinfo import summary

    generator = Generator("high")

    print(summary(generator, (64, 100,1,1)))
