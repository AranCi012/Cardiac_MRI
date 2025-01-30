import torch
import torch.nn as nn
import torchvision.models as models

class UNetVGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(UNetVGG16, self).__init__()

        # Carica VGG16 senza BatchNorm
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None).features  

        # Encoder (senza BatchNorm)
        self.enc1 = nn.Sequential(*vgg16[:6])    # 160x160 -> 160x160
        self.enc2 = nn.Sequential(*vgg16[6:13])  # 160x160 -> 80x80
        self.enc3 = nn.Sequential(*vgg16[13:23]) # 80x80 -> 40x40
        self.enc4 = nn.Sequential(*vgg16[23:33]) # 40x40 -> 20x20
        self.enc5 = nn.Sequential(*vgg16[33:43]) # 20x20 -> 10x10

        # Bottleneck senza BatchNorm
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder con upsampling tramite ConvTranspose2d
        self.dec5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)  # 10x10 -> 20x20
        self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)   # 20x20 -> 40x40
        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)   # 40x40 -> 80x80
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)    # 80x80 -> 160x160
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)     # 80x80 -> 160x160

        # Ultima convoluzione con Sigmoid
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # 160x160
        enc2_out = self.enc2(enc1_out)  # 80x80
        enc3_out = self.enc3(enc2_out)  # 40x40
        enc4_out = self.enc4(enc3_out)  # 20x20
        enc5_out = self.enc5(enc4_out)  # 10x10

        # Bottleneck
        bottleneck_out = self.bottleneck(enc5_out)

        # Decoder con upsampling
        dec5_out = self.dec5(bottleneck_out)  # 10x10 -> 20x20
        dec4_out = self.dec4(dec5_out)       # 20x20 -> 40x40
        dec3_out = self.dec3(dec4_out)       # 40x40 -> 80x80
        dec2_out = self.dec2(dec3_out)       # 80x80 -> 160x160
        dec1_out = self.dec1(dec2_out)       # 80x80 -> 160x160

        # Output finale con Sigmoid
        out = self.final_conv(dec1_out)
        return torch.sigmoid(out)

# Test con immagine 160x160
if __name__ == "__main__":
    model = UNetVGG16(pretrained=True)
    x = torch.randn(1, 3, 160, 160)  # Input: batch size 1, 3 canali (RGB), 160x160
    y = model(x)
    print(f'Final output shape: {y.shape}')  # Output: torch.Size([1, 1, 160, 160])
