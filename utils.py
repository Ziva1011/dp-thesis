# import torch
import numpy as np
from PIL import Image
import os

# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=True)


# import segmentation_models_pytorch as smp

# model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=3,                      # model output channels (number of classes in your dataset)
# )

data_dir_train= '/media/datasets/MSD/Task03_Liver/train_split'
data_dir_val= '/media/datasets/MSD/Task03_Liver/val_split'
# image_list = [np.load(os.path.join(data_dir, image)) for image in os.listdir(data_dir)]
# print(image_list)
dataset = NiftiSegmentationDataset(data_dir)
# dataset_torch = torch.from_numpy(dataset)

# dataloader= data.DataLoader(dataset=dataset_torch,
#                                       batch_size=2,
#                                       shuffle=True)