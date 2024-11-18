"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_VGG16, data_transforms_ResNet50, data_transforms_ViT_b_16, data_transforms_AlexNet, data_transforms_ViTForImageClassification, data_transforms_EfficientNet
from model import Net
from model import VGG16, ResNet50, ViT_b_16, AlexNet, ResNet50_frozen, ViTForImageClassification_, EfficientNet


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "VGG16":
            return VGG16()
        elif self.model_name == "ResNet50":
            return ResNet50()
        elif self.model_name == "ViT_b_16":
            return ViT_b_16()  
        elif self.model_name == "AlexNet":
            return AlexNet()
        elif self.model_name == "ResNet50_frozen":
            return ResNet50_frozen()
        elif self.model_name == "ViTForImageClassification":
            return ViTForImageClassification_()
        elif self.model_name == "EfficientNet":
            return EfficientNet()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "VGG16":
            return data_transforms_VGG16
        elif self.model_name == "ResNet50":
            return data_transforms_ResNet50
        elif self.model_name == "ViT_b_16":
            return data_transforms_ViT_b_16
        elif self.model_name == "AlexNet":
            return data_transforms_AlexNet
        elif self.model_name == "ResNet50_frozen":
            return data_transforms_ResNet50
        elif self.model_name == "ViTForImageClassification":
            return data_transforms_ViTForImageClassification
        elif self.model_name == "EfficientNet":
            return data_transforms_EfficientNet
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
