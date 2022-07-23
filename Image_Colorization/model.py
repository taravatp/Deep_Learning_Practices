import torch.nn as nn
import torch
from torchsummary import summary


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding='same'),  # thing about padding arg
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.down(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.bottleneck(x)


class BaseModel(nn.Module):
    def __init__(self, kernel, num_filters, num_colors, in_channels=1):
        super(BaseModel, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.downs.append(DownConv(in_channels, num_filters, kernel))
        self.downs.append(DownConv(num_filters, num_filters * 2, kernel))
        self.bottleneck = Bottleneck(num_filters * 2, num_filters * 2, kernel)
        self.ups.append(UpConv(num_filters * 2, num_filters, kernel))
        self.ups.append(UpConv(num_filters, num_colors, kernel))
        self.final_layer = nn.Conv2d(num_colors, num_colors, kernel, padding='same')

    def forward(self, x):
        for down in self.downs:
            x = down(x)
        x = self.bottleneck(x)
        for up in self.ups:
            x = up(x)
        x = self.final_layer(x)
        return x


class CustomUNET(nn.Module):
    def __init__(self, kernel, num_filters, num_colors, in_channels=1):
        super(CustomUNET, self).__init__()

        self.first_layer = DownConv(in_channels, num_filters, kernel)
        self.second_layer = DownConv(num_filters, num_filters * 2, kernel)
        self.third_layer = Bottleneck(num_filters * 2, num_filters * 2, kernel)
        self.fourth_layer = UpConv(num_filters * 2 * 2, num_filters, kernel)
        self.fifth_layer = UpConv(num_filters * 2, num_colors, kernel)
        self.sixth_layer = nn.Conv2d(in_channels + num_colors, num_colors, kernel, padding='same')

    def forward(self, x):
        first = self.first_layer(x)
        second = self.second_layer(first)
        third = self.third_layer(second)
        fourth = self.fourth_layer(torch.cat([second, third], dim=1))
        fifth = self.fifth_layer(torch.cat([first, fourth], dim=1))
        sixth = self.sixth_layer(torch.cat([x, fifth], dim=1))

        return sixth


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, enc_or_dec='encoder'):
        super(SkipConnection, self).__init__()
        self.enc_or_dec = enc_or_dec
        self.skip_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.enc_or_dec == 'encoder':
            return self.skip_encoder(x)
        if self.enc_or_dec == 'decoder':
            return self.skip_decoder(x)


class CustomUnetWithResiduals(nn.Module):
    def __init__(self, kernel, num_filters, num_colors, in_channels=1):
        super(CustomUnetWithResiduals, self).__init__()
        self.first_layer = DownConv(in_channels, num_filters, kernel)
        self.skip_first_layer = SkipConnection(in_channels, num_filters, enc_or_dec='encoder')

        self.second_layer = DownConv(num_filters, num_filters * 2, kernel)
        self.skip_second_layer = SkipConnection(num_filters, num_filters * 2, enc_or_dec='encoder')

        self.third_layer = Bottleneck(num_filters * 2, num_filters * 2, kernel)

        self.fourth_layer = UpConv(num_filters * 2 * 2, num_filters, kernel)
        self.skip_fourth_layer = SkipConnection(num_filters * 2 * 2, num_filters, enc_or_dec='decoder')

        self.fifth_layer = UpConv(num_filters * 2, num_colors, kernel)
        self.skip_fifth_layer = SkipConnection(num_filters * 2, num_colors, enc_or_dec='decoder')

        self.sixth_layer = nn.Conv2d(in_channels + num_colors, num_colors, kernel, padding='same')

    def forward(self, x):
        first = self.first_layer(x)
        skip = self.skip_first_layer(x)
        first = first + skip

        second = self.second_layer(first)
        skip = self.skip_second_layer(first)
        second = second + skip

        third = self.third_layer(second)
        third = third + second

        fourth = self.fourth_layer(torch.cat([second, third], dim=1))
        skip = self.skip_fourth_layer(torch.cat([second, third], dim=1))
        fourth = fourth + skip

        fifth = self.fifth_layer(torch.cat([first, fourth], dim=1))
        skip = self.skip_fifth_layer(torch.cat([first, fourth], dim=1))
        fifth = fifth + skip

        sixth = self.sixth_layer(torch.cat([x, fifth], dim=1))

        return sixth


if __name__ == "__main__":
    baseModel = CustomUnetWithResiduals(kernel=3, num_filters=8, num_colors=24)
    inp = torch.zeros((1, 1, 32, 32))
    out = baseModel(inp)
    print(baseModel)
    # print(output.shape)
    # print(summary(baseModel, input_size=(1, 32, 32)))
