import torch.nn as nn
import torch.nn.functional as F
from torch import hub
import torchvision.models as models
nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model_vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.model_vgg.classifier[6] = nn.Linear(self.model_vgg.classifier[6].in_features, 500)

    def forward(self, x):
        return(self.model_vgg(x))
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model_resnet.fc = nn.Linear(self.model_resnet.fc.in_features, 500)

    def forward(self, x):
        return(self.model_resnet(x))

class ViT_b_16(nn.Module):
    def __init__(self):
        super(ViT_b_16, self).__init__()
        self.model_vit = models.vit_b_16(models.ViT_B_16_Weights.DEFAULT)
        self.model_vit.heads[0] = nn.Linear(self.model_vit.heads[0].in_features, 500)
        
    def forward(self, x):
        return(self.model_vit(x))

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model_alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.model_alexnet.classifier[6] = nn.Linear(self.model_alexnet.classifier[6].in_features, 500)

    def forward(self, x):
        return(self.model_alexnet(x))
    
class ResNet50_frozen(nn.Module):
    def __init__(self):
        super(ResNet50_frozen, self).__init__()
        self.model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model_resnet.fc = nn.Linear(self.model_resnet.fc.in_features, 500)
        for name, param in self.model_resnet.named_parameters():
            if "layer4" not in name:  # On entraîne uniquement la dernière couche "layer4" et la couche de sortie
                param.requires_grad = False

    def forward(self, x):
        return(self.model_resnet(x))
    
class Dino(nn.Module):
    def __init__(self):
        super(Dino, self).__init__()
        self.model_dino = hub.load("facebookresearch/dino:main", "dino_vits8")
        self.model_dino.head = nn.Linear(768, 500)
        for param in self.model_dino.patch_embed.parameters():
            param.requires_grad = False
        for block in self.model_dino.blocks[:-4]:  # Freeze all but the last 4 blocks
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x):
        return(self.model_dino(x))

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model_efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.model_efficientnet.classifier[1] = nn.Linear(self.model_efficientnet.classifier[1].in_features, 500)

    def forward(self, x):
        return(self.model_efficientnet(x))