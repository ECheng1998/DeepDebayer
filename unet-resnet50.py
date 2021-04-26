import cv2, math, sys, torch, torchvision, time
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from math import exp
from PIL import Image as im
from torchsummary import summary
from torch import tensor
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from google.colab import files

class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()
    
# ------------------ #
#        SSIM
# ------------------ #

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def rgb2bayer(img):
    temp = img.numpy()
    temp = temp.transpose(1,2,0)

    w,h,c = temp.shape
    resArray = np.zeros((w, h, 3), dtype=np.float)
    resArray[::2, ::2, 2] = temp[::2, ::2, 2]
    resArray[1::2, ::2, 1] = temp[1::2, ::2, 1]
    resArray[::2, 1::2, 1] = temp[::2, 1::2, 1]
    resArray[1::2, 1::2, 0] = temp[1::2, 1::2, 0]
    resArray = resArray.astype('float32')

    temp = torch.from_numpy(resArray)
    temp = temp.permute(2,0,1)
    return temp

def tensorStack(tenS):
    temp = []
    for i in tenS:
        i = rgb2bayer(i)
        temp.append(i)
    temp = torch.stack(temp)
    return temp
    
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
      return 100
    IXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def train_model(train_dl, model, method):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    if method == 'MSSSIM':
        criterion = MS_SSIM_L1_LOSS()
        for epoch in range(60):
            for data in train_dl:
                inputs, labels = data
                temp = tensorStack(inputs)
                inputs, temp = inputs.cuda(), temp.cuda()
                optimizer.zero_grad()
                outputs = model(temp)
                loss = criterion(inputs, outputs)
                loss.backward()
                optimizer.step()
            if ((epoch+1) % 10 == 0):
                print(loss, epoch)

    elif method == 'MSE':
        criterion = nn.MSELoss()
        for epoch in range(60):
            running_loss = 0.0
            for data in train_dl:
                inputs, labels = data
                temp = tensorStack(inputs)
                inputs, temp = inputs.cuda(), temp.cuda()
                optimizer.zero_grad()
                outputs = model(temp)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if ((epoch+1) % 10 == 0):
                print(running_loss, 'Epochs = ', epoch)

    else:
        print('Non-valid loss method \n')
        return 0
            
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
])

BSDpreprocess = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class OutBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.conv32 = nn.Conv2d(32, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.outConv = nn.Conv2d(32, 3, padding=padding, kernel_size=3, stride=stride)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.conv32(x)
        x = self.dropout(x)
        x = self.outConv(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = OutBlock(64, 32)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

# ------------------ #
#      Training
# ------------------ #
batchSize = 32
model = UNetWithResnet50Encoder().cuda()
method = 'MSE'

trainpath = 'drive/MyDrive/5th Yr Project/TrainingData/FiveK'
trainset = datasets.ImageFolder(root=trainpath,transform=preprocess)
trainloader = DataLoader(dataset=trainset,batch_size=batchSize, shuffle=True)
t0 = time.time()
print('Training with FiveK dataset')
train_model(trainloader,model,method)
print('Training time for FiveK')
print('{} seconds'.format(time.time() - t0))

trainpath = 'drive/MyDrive/5th Yr Project/TrainingData/BSD300'
trainset = datasets.ImageFolder(root=trainpath,transform=preprocess)
trainloader = DataLoader(dataset=trainset,batch_size=batchSize, shuffle=True)
print('Training with BSD300')
train_model(trainloader,model,method)
print('Training time for BSD300')
print('{} seconds'.format(time.time() - t0))

trainpath = 'drive/MyDrive/5th Yr Project/TrainingData/BSD300'
trainset = datasets.ImageFolder(root=trainpath,transform=BSDpreprocess)
trainloader = DataLoader(dataset=trainset,batch_size=batchSize, shuffle=True)
print('Training with BSD300 Augments')
train_model(trainloader,model,method)
print('Training time for BSD300 Augments')
print('{} seconds'.format(time.time() - t0))

trainpath = 'drive/MyDrive/5th Yr Project/TrainingData/DTD'
trainset = datasets.ImageFolder(root=trainpath,transform=preprocess)
trainloader = DataLoader(dataset=trainset,batch_size=batchSize, shuffle=True)
print('Training with DTD')
train_model(trainloader,model,method)
print('Training time total')
print('{} seconds'.format(time.time() - t0))

# ------------------ #
#   Testing Kodak
# ------------------ #
testpath = 'drive/MyDrive/5th Yr Project/TestData/Kodak'
testset = datasets.ImageFolder(root=testpath,transform=preprocess)
testloader = DataLoader(dataset=testset,batch_size=24, shuffle=False)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cpu()
        bayered = tensorStack(images)
        bayered = bayered.cuda()
        outputs = model(bayered)
        outputs = outputs.cpu()
        imshow(torchvision.utils.make_grid(outputs))

# ------------------ #
#   Printing Stats
# ------------------ #

print('Kodak Dataset')
avgPSNR = 0

for i in range(len(testset)):
    im1 = tf.image.convert_image_dtype(images[i], tf.float32)
    im2 = tf.image.convert_image_dtype(outputs[i], tf.float32)
    psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
    avgPSNR += psnr2
    print('PSNR of image ', i, ' = ', psnr2)

print('Average PSNR = ', avgPSNR/len(testset))
print('-------------SSIM-------------')

for i in range(len(testset)):
    temp1 = torch.unsqueeze(images[i],0)
    temp2 = torch.unsqueeze(outputs[i],0)
    print('SSIM of image ', i, ' = ', ssim(temp1,temp2))

print('Average SSIM = ', ssim(images,outputs))

# ------------------ #
#    Testing McM
# ------------------ #

testpath = 'drive/MyDrive/Colab Notebooks/McMaster'
testset = datasets.ImageFolder(root=testpath,transform=preprocess)
testloader = DataLoader(dataset=testset,batch_size=18, shuffle=False)

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cpu()
        bayered = tensorStack(images)
        bayered = bayered.cuda()
        outputs = model(bayered)
        outputs = outputs.cpu()
        imshow(torchvision.utils.make_grid(outputs))

# ------------------ #
#   Printing Stats
# ------------------ #

print('McMaster Dataset')
avgPSNR = 0

for i in range(len(testset)):
    im1 = tf.image.convert_image_dtype(images[i], tf.float32)
    im2 = tf.image.convert_image_dtype(outputs[i], tf.float32)
    psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
    print('PSNR of image ', i, ' = ', psnr2)
    avgPSNR += psnr2

print('Average PSNR = ', avgPSNR/len(testset))
print('-------------SSIM-------------')

for i in range(len(testset)):
    temp1 = torch.unsqueeze(images[i],0)
    temp2 = torch.unsqueeze(outputs[i],0)
    print('SSIM of image ', i, ' = ', ssim(temp1,temp2))

print('Average SSIM = ', ssim(images,outputs))
