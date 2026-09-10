import segmentation_models_pytorch as smp

def build_segmentation_model():
    """Build U-Net with ResNet34 backbone for oil spill segmentation."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
