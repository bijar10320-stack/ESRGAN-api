import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import io
import numpy as np


# RDB and RRDB definitions

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        fea = self.conv_first(x)
        body_out = self.body(fea)
        fea = self.conv_body(body_out) + fea
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_hr(fea))
        out = self.conv_last(fea)
        return out


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RRDBNet().to(device).eval()


# Load checkpoint with key fix

checkpoint = torch.load(r"C:\Users\pc\PycharmProjects\PythonProject13\ESRGAN_x4.pth", map_location=device)
if "params_ema" in checkpoint:
    checkpoint = checkpoint["params_ema"]

# rename keys conv_up1 -> upconv1 and conv_up2 -> upconv2
new_ckpt = {}
for k, v in checkpoint.items():
    if k.startswith("conv_up1"):
        new_ckpt[k.replace("conv_up1", "upconv1")] = v
    elif k.startswith("conv_up2"):
        new_ckpt[k.replace("conv_up2", "upconv2")] = v
    else:
        new_ckpt[k] = v

model.load_state_dict(new_ckpt, strict=True)

def super_resolve(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(img_tensor)

    sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr = (sr.clip(0, 1) * 255).astype(np.uint8)

    sr_img = Image.fromarray(sr)

    buf = io.BytesIO()
    sr_img.save(buf, format="PNG")
    return buf.getvalue()



