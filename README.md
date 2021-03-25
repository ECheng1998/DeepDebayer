# Pytorch DeepDebayer
This project uses a U-net with ResNet50 encoders to demosaic an image.
The inputs to this are images in RGB format. The images are converted into their Bayer representations and fed into the neural network. The output is the demosaiced version of the Bayer image. The outputs have better clarity i.e. less artifacts.

<img src="https://i.imgur.com/UYTi1uB.png" width="200" height="300"> ----> <img src="https://i.stack.imgur.com/9PrQc.jpg" width="200" height="300">

Written in Python using Pytorch backend

# References
[1] https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder
[2] https://github.com/psyrocloud/MS-SSIM_L1_LOSS
