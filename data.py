import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTImageProcessor

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_VGG16 = models.VGG16_Weights.DEFAULT.transforms()

data_transforms_ResNet50 = models.ResNet50_Weights.DEFAULT.transforms()

data_transforms_ViT_b_16 = models.ViT_B_16_Weights.DEFAULT.transforms()

data_transforms_AlexNet = models.AlexNet_Weights.DEFAULT.transforms()

data_transforms_ViTForImageClassification = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

data_transforms_EfficientNet = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomRotation(30),
    ]
)
