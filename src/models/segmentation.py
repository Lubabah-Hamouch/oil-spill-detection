import segmentation_models_pytorch as smp

def build_segmentation_model(encoder_weights="imagenet"):
    """Build U-Net with ResNet34 backbone for oil spill segmentation.

    encoder_weights="imagenet" initialises the encoder from ImageNet (training). Pass None when a
    trained checkpoint will be loaded on top: the architecture is identical, it just skips
    downloading and allocating weights that would be overwritten anyway.
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
