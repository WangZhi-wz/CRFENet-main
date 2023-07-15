"""
来源：
https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
"""
import argparse
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd.grad_mode import F
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from model.GlobalEnhancedCoordinateAttention.GECAttention_ablation_3 import Model as create_model
from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_DP0505 as UNet

# from model.SGRLNet.SGRLNet_4_stage1_3_3_12_1 import Model as create_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default= r'D:\datasets\paper2_dataset\kvasir82',
        help='Input image path')
    parser.add_argument('--weight-path', type=str,
                        default= r'C:\Users\15059\Desktop\Paper2模型\0508_DP_AFF\7_Kvasir.pth',
                        help='weight path of the model')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def find_image_file(source_path, file_lst):
    """
    递归寻找 文件夹以及子目录的 图片文件。
    :param source_path: 源文件夹路径
    :param file_lst: 输出 文件路径列表
    :return:
    """
    image_ext = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    for dir_or_file in os.listdir(source_path):
        file_path = os.path.join(source_path, dir_or_file)
        if os.path.isfile(file_path):  # 判断是否为文件
            file_name_ext = os.path.splitext(os.path.basename(file_path))  # 文件名与后缀
            if len(file_name_ext) < 2:
                continue
            if file_name_ext[1] in image_ext:  # 后缀在后缀列表中
                file_lst.append(file_path)
            else:
                continue
        elif os.path.isdir(file_path):  # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
            find_image_file(file_path, file_lst)
        else:
            print('文件夹没有图片' + os.path.basename(file_path))


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    # model = models.resnet50(pretrained=True)
    model = UNet(backbone=dict(
        type='mit_{}'.format('b3'),
        style='pytorch'),
        decode_head=dict(
            type='UPerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=128,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            decoder_params=dict(embed_dim=768),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        neck=None,
        auxiliary_head=None,
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
        pretrained='pretrained/mit_{}.pth'.format('b3')).cuda()
    model.load_state_dict(torch.load(args.weight_path), False)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.decode_head.linear_pred]  # 一个列表
    # target_layers = get_last_conv_name(model)  # getfeature.7.1.conv2 ## 会报错

    img_list = []
    find_image_file(args.image_path, img_list)
    for img_path in img_list:
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]  # [509,515]
        rgb_img = cv2.resize(rgb_img, (352, 352))
        rgb_img = np.float32(rgb_img) / 255
        # rgb_img = np.transpose(rgb_img, [1, 2, 0])
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=None)
        gb = F.upsample(gb, size=(352, 352), mode='bilinear', align_corners=False)


        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        img_name = os.path.basename(img_path).split('.')[0]
        base_path = os.path.dirname(img_path).split('dataset')[-1]
        # 把路径前的斜杠去掉
        prefix = '/\\'  # 定义要判断的前缀字符
        while base_path.startswith(tuple(prefix)):  # 判断字符串是否以指定的前缀字符开头
            base_path = base_path[1:]  # 去掉开头的字符
        base_path = os.path.join(args.method, base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        cv2.imwrite(os.path.join(base_path, f'{img_name}_cam.jpg'), cam_image)
        cv2.imwrite(os.path.join(base_path, f'{img_name}_gb.jpg'), gb)
        cv2.imwrite(os.path.join(base_path, f'{img_name}_cam_gb.jpg'), cam_gb)
