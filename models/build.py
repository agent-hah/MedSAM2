import models

model_2d = {
    "UNet",
    "FR_UNet",
    "Att_UNet",
    "CSNet",
    "UNet_Nested",
    "MAA_Net",
    "Res_UNet",
    "UNet_DP"
    
}

# model_3d = {
#     "UNet_3D",
#     "FR_UNet_3D",
#     "CSNet_3D",
#     "Att_UNet_3D",
#     "Res_UNet_3D",
#     "UNet_Nested_3D",
#     "IPN",
#     "PSC",
#     "SVS_Net",
#     "QS_UNet",
#     "QS_FRUNet"

# }



def build_model(config):
    if config.MODEL.TYPE in model_2d:
        return getattr(models, config.MODEL.TYPE)(
            num_classes=2,
            num_channels=8
        ), True
    else:
        return getattr(models, config.MODEL.TYPE)(
            num_classes=2,
            num_channels=1
        ), False
    



def build_wsl_model(config):
    if config.MODEL.TYPE in model_2d:
        return getattr(models, config.MODEL.TYPE)(
            num_classes=3,
            num_channels=8
        ), True
    elif config.MODEL.TYPE == "UNet_CCT":
        return getattr(models, config.MODEL.TYPE)(
                num_classes=2,
                num_channels=8
            ), True
    elif config.MODEL.TYPE == "VSS_Net":
        return getattr(models, config.MODEL.TYPE)(
            num_classes=2,
            num_channels=1,
            input_reduce=[0, 1, 2, 3, 4, 5, 6, 7],  # full 8-frame sequence
        ), False
    else:
        return getattr(models, config.MODEL.TYPE)(
            num_classes=3,
            num_channels=1
        ), False



