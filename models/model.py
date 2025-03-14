import torch
from torch import nn
from .image_encoder import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200



def create_model(args,in_channels=1):
    # 假设您在 args 中定义了以下参数
    model_depth = args.model_depth  # 例如 10, 18, 34, 50, 101, 152, 200
    sample_input_D = args.sample_input_D  # 输入数据的深度
    sample_input_H = args.sample_input_H  # 输入数据的高度
    sample_input_W = args.sample_input_W  # 输入数据的宽度
    num_seg_classes = args.num_classes  # 分割类别数

    if model_depth == 10:
        model = resnet10(sample_input_D=sample_input_D,
                         sample_input_H=sample_input_H,
                         sample_input_W=sample_input_W,
                         num_seg_classes=num_seg_classes,
                         in_channels=in_channels)
    elif model_depth == 18:
        model = resnet18(sample_input_D=sample_input_D,
                         sample_input_H=sample_input_H,
                         sample_input_W=sample_input_W,
                         num_seg_classes=num_seg_classes,
                         in_channels=in_channels)
    elif model_depth == 34:
        model = resnet34(sample_input_D=sample_input_D,
                         sample_input_H=sample_input_H,
                         sample_input_W=sample_input_W,
                         num_seg_classes=num_seg_classes,
                         in_channels=in_channels)
    elif model_depth == 50:
        model = resnet50(sample_input_D=sample_input_D,
                         sample_input_H=sample_input_H,
                         sample_input_W=sample_input_W,
                         num_seg_classes=num_seg_classes,
                         in_channels=in_channels)
    elif model_depth == 101:
        model = resnet101(sample_input_D=sample_input_D,
                          sample_input_H=sample_input_H,
                          sample_input_W=sample_input_W,
                          num_seg_classes=num_seg_classes,
                          in_channels=in_channels)
    elif model_depth == 152:
        model = resnet152(sample_input_D=sample_input_D,
                          sample_input_H=sample_input_H,
                          sample_input_W=sample_input_W,
                          num_seg_classes=num_seg_classes,
                          in_channels=in_channels)
    elif model_depth == 200:
        model = resnet200(sample_input_D=sample_input_D,
                          sample_input_H=sample_input_H,
                          sample_input_W=sample_input_W,
                          num_seg_classes=num_seg_classes,
                          in_channels=in_channels)
    else:
        raise ValueError("Unsupported model depth, must be one of 10, 18, 34, 50, 101, 152, 200")
    
    return model