import torch.nn as nn
from torchvision.models import mobilenet_v3_large

def build_classification_model(num_classes=2, pretrained=True):
    """Build MobileNetV3_Large model for oil spill classification."""
    model = mobilenet_v3_large(pretrained=pretrained)
    
    # Modify classifier for binary classification
    num_features = model.classifier[0].in_features
    model.classifier[0] = nn.Linear(num_features, 1280)
    model.classifier[3] = nn.Linear(1280, num_classes)
    
    return model
