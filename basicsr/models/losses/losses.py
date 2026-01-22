import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss  # 把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss  # 把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.size()) == 4

        return (
            self.loss_weight
            * self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction


#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'}
        vgg_type (str): VGG type. Options: 'vgg16', 'vgg19'. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must be in range [0, 1]. Default: True.
        requires_grad (bool): If True, the parameters of VGG will be optimized.
            Default: False.
    """

    def __init__(
        self,
        layer_name_list,
        vgg_type="vgg19",
        use_input_norm=True,
        requires_grad=False,
    ):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # Get VGG model
        if vgg_type == "vgg16":
            vgg = torchvision.models.vgg16(pretrained=True)
        elif vgg_type == "vgg19":
            vgg = torchvision.models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unsupported VGG type: {vgg_type}")

        # VGG layer name to index mapping
        self.names = {
            "conv1_1": 0,
            "relu1_1": 1,
            "conv1_2": 2,
            "relu1_2": 3,
            "pool1": 4,
            "conv2_1": 5,
            "relu2_1": 6,
            "conv2_2": 7,
            "relu2_2": 8,
            "pool2": 9,
            "conv3_1": 10,
            "relu3_1": 11,
            "conv3_2": 12,
            "relu3_2": 13,
            "conv3_3": 14,
            "relu3_3": 15,
            "conv3_4": 16,
            "relu3_4": 17,
            "pool3": 18,
            "conv4_1": 19,
            "relu4_1": 20,
            "conv4_2": 21,
            "relu4_2": 22,
            "conv4_3": 23,
            "relu4_3": 24,
            "conv4_4": 25,
            "relu4_4": 26,
            "pool4": 27,
            "conv5_1": 28,
            "relu5_1": 29,
            "conv5_2": 30,
            "relu5_2": 31,
            "conv5_3": 32,
            "relu5_3": 33,
            "conv5_4": 34,
            "relu5_4": 35,
            "pool5": 36,
        }

        # Find the maximum index needed
        max_idx = 0
        for layer_name in layer_name_list:
            max_idx = max(max_idx, self.names[layer_name])

        # Only keep the layers we need
        self.vgg_net = nn.Sequential(*list(vgg.features.children())[: max_idx + 1])

        # Normalization for ImageNet pretrained VGG
        if self.use_input_norm:
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w). Values should be in [0, 1].

        Returns:
            dict: A dict containing the specified VGG features.
        """
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for name, module in self.vgg_net._modules.items():
            x = module(x)
            layer_name = list(self.names.keys())[
                list(self.names.values()).index(int(name))
            ]
            if layer_name in self.layer_name_list:
                output[layer_name] = x.clone()

        return output


class PerceptualLoss(nn.Module):
    """Perceptual loss with VGG network.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Example: {'relu1_1': 1.0, 'relu2_1': 1.0, 'relu3_1': 1.0, 'relu4_1': 1.0, 'relu5_1': 1.0}
        vgg_type (str): VGG type. Options: 'vgg16', 'vgg19'. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Default: True.
        perceptual_weight (float): Weight for perceptual loss. Default: 1.0.
        style_weight (float): Weight for style loss. If > 0, style loss will be computed. Default: 0.
        criterion (str): Criterion for perceptual loss. Options: 'l1', 'l2'. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights={"relu3_1": 1.0, "relu4_1": 1.0, "relu5_1": 1.0},
        vgg_type="vgg19",
        use_input_norm=True,
        perceptual_weight=1.0,
        style_weight=0.0,
        criterion="l1",
    ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
        )

        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion == "l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Perceptual loss value.
        """
        # Extract VGG features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # Calculate perceptual loss
        percep_loss = 0.0
        if self.perceptual_weight > 0:
            for k in x_features.keys():
                percep_loss += (
                    self.criterion(x_features[k], gt_features[k])
                    * self.layer_weights[k]
                )
            percep_loss *= self.perceptual_weight

        # Calculate style loss (Gram matrix)
        style_loss = 0.0
        if self.style_weight > 0:
            for k in x_features.keys():
                x_gram = self._gram_matrix(x_features[k])
                gt_gram = self._gram_matrix(gt_features[k])
                style_loss += self.criterion(x_gram, gt_gram) * self.layer_weights[k]
            style_loss *= self.style_weight

        return percep_loss + style_loss

    def _gram_matrix(self, x):
        """Calculate Gram matrix.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
        return gram
