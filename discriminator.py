import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, resolution):
        super().__init__()

        self.resolution = resolution
        if resolution == "low":
            self.c1_1 = self.make_conv(64*1, 64*1, 4, 2, 1)
        elif resolution == "mid":
            self.c2_1 = self.make_conv(64*2, 64*2, 4, 2, 1)
        elif resolution == "high":
            self.c3_1 = self.make_conv(64*4, 64*4, 4, 2, 1)
        elif resolution == "highest":
            self.c4_1 = self.make_conv(64*8, 64*8, 4, 2, 1)

        self.c1 = self.make_conv(3, 64*1, 4, 2, 1, first = True)
        self.c2 = self.make_conv(64*1, 64*2, 4, 2, 1)
        self.c3 = self.make_conv(64*2, 64*4, 4, 2, 1)
        self.c4 = self.make_conv(64*4, 64*8, 4, 2, 1)
        self.c5 = self.make_conv(64*8, 1, 4, 1, 0, last = True)

    def make_conv(self, in_c, out_c, k, s, p, first = False, last = False):
        layers = list()
        layers.append(
            nn.Conv2d(
                in_channels = in_c,
                out_channels = out_c,
                kernel_size = k,
                stride = s,
                padding = p,
                bias = False
            )
        )

        if (first == False) and (last == False):
            layers.append(nn.BatchNorm2d(num_features = out_c))
            layers.append(nn.LeakyReLU(negative_slope = 0.2, inplace = False))             # What is inplace option ? Normally in_place means it change itself. e.g) pandas. See datails link below
                                                               # https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        
        elif first == True:
            layers.append(nn.LeakyReLU(negative_slope = 0.2, inplace = False))   
        elif last == True:
            layers.append(nn.Sigmoid())


        return nn.Sequential(*layers)

    def forward(self,x):
        
        x = self.c1(x)
        if self.resolution == "low":
            x = self.c1_1(x)
        x = self.c2(x)
        if self.resolution == "mid":
            x = self.c2_1(x)
        x = self.c3(x)
        if self.resolution == "high":
            x = self.c3_1(x)
        x = self.c4(x)
        if self.resolution == "highest":
            x = self.c4_1(x)
        x = self.c5(x)

        return x


if __name__ == "__main__":

    from torchinfo import summary

    discriminator = Discriminator()

    print(summary(discriminator, (100,3,64,64), "base"))
