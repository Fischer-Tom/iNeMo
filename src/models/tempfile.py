import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv - bn - relu) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=(1, 1)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding[0]),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        X = self.doubleconv(X)
        return X


class Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.doubleconv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, X1, X2):
        X1 = self.up(X1)
        diffY = torch.tensor([X2.size()[2] - X1.size()[2]])
        diffX = torch.tensor([X2.size()[3] - X1.size()[3]])
        # just incase:
        X1 = F.pad(X1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        X = torch.cat([X2, X1], dim=1)
        X = self.doubleconv(X)
        return X


class Up_noSkip(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, padding=(1, 1)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.doubleconv = DoubleConv(
            in_channels, out_channels, mid_channels, padding=padding
        )

    def forward(self, X):
        X = self.up(X)
        X = self.doubleconv(X)
        return X


class ResNetExt(nn.Module):
    # TODO: How to deal with skip connections in ReplayBuffer
    def __init__(self, pretrained, nb_classes=12, pose_output_size=126, dropout_p=None):
        super().__init__()
        """net = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )"""
        net = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor.add_module("5", net.layer2)
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(1024 + 1024, 1024, 512)
        self.upsample2 = Up(512 + 512, 512, 256)
        self.with_dropout = dropout_p is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, return_all=False):
        x1 = self.extractor(x)
        x2 = self.extractor1(x1)
        x3 = self.extractor2(x2)
        features = self.upsample2(self.upsample1(self.upsample0(x3), x2), x1)

        if return_all:
            return features, x3, x2, x1
        return features


class GlobalLocalConverter(nn.Module):
    def __init__(self, local_size):
        super().__init__()
        self.local_size = local_size
        self.padding = sum(([t - 1 - t // 2, t // 2] for t in local_size[::-1]), [])

    def forward(self, X):
        n, c, h, w = X.shape  # torch.Size([1, 2048, 8, 8])

        X = F.pad(X, self.padding)

        X = F.unfold(X, kernel_size=self.local_size)

        return X


def keypoints_to_pixel_index(keypoints, downsample_rate, original_img_size=(480, 640)):
    # line_size = 9
    # TODO: used to be original_img_size[1]
    line_size = original_img_size[1] // downsample_rate
    # round down, new coordinate (keypoints[:,:,0]//downsample_rate, keypoints[:, :, 1] // downsample_rate)
    return (
        keypoints[:, :, 0] // downsample_rate * line_size
        + keypoints[:, :, 1] // downsample_rate
    ).clamp(min=0, max=7999)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b * e * f
    return out


def get_noise_pixel_index(keypoints, max_size, n_samples, obj_mask=None):
    n = keypoints.shape[0]
    # remove the point in keypoints by set probability to 0 otherwise 1 -> mask [n, size] with 0 or 1
    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    mask = mask.scatter(1, keypoints.type(torch.long), 0.0)
    if obj_mask is not None:
        mask = obj_mask.view(n, -1)
    # generate the sample by the probabilities
    return torch.multinomial(mask, n_samples)


class NetE2E(nn.Module):
    def __init__(
        self,
        pretrain,
        net_type,
        local_size,
        output_dimension,
        n_classes,
        reduce_function=None,
        n_noise_points=0,
        noise_on_mask=True,
        img_shape=(640, 800),
    ):
        # output_dimension = 128
        super().__init__()

        self.net = ResNetExt(pretrain, n_classes)

        self.img_shape = img_shape
        self.size_number = local_size[0] * local_size[1]
        self.output_dimension = output_dimension
        # size_number = reduce((lambda x, y: x * y), local_size)
        if reduce_function:
            reduce_function.register_local_size(local_size)
            self.size_number = 1

        self.reduce_function = reduce_function
        self.net_type = net_type
        self.net_stride = 8
        self.converter = GlobalLocalConverter(local_size)
        self.noise_on_mask = noise_on_mask

        # output_dimension == -1 for abilation study.
        if self.output_dimension == -1:
            self.out_layer = None
        else:
            # 256 -> 128 for resnetext
            self.out_layer = nn.Linear(
                256 * self.size_number,
                self.output_dimension,
            )
            # output_dimension , net_out_dimension[net_type] * size_number

        self.n_noise_points = n_noise_points
        # self.norm_layer = lambda x: F.normalize(x, p=2, dim=1)

    # forward
    def forward_test(self, X, return_all=False):
        # Feature map n, c, w, h -- 1, 128, 128, 128
        res = self.net.forward(X, return_all=return_all)
        if return_all:
            X, y, z = res
        else:
            X = res

        if self.output_dimension == -1:
            return F.normalize(X, p=2, dim=1)
        if self.size_number == 1:
            X = torch.nn.functional.conv2d(
                X,
                self.out_layer.weight.unsqueeze(2).unsqueeze(3),
            )
            # TODO: Why is the bias usually ignored here?
            X = X + self.out_layer.bias.unsqueeze(-1).unsqueeze(-1)
        elif self.size_number > 1:
            X = torch.nn.functional.conv2d(
                X,
                self.out_layer.weight.view(
                    self.output_dimension,
                    256,
                    self.size_number,
                )
                .permute(2, 0, 1)
                .reshape(
                    self.size_number * self.output_dimension,
                    256,
                )
                .unsqueeze(2)
                .unsqueeze(3),
            )
        # n, c, w, h
        # 1, 128, (w_original - 1) // 32 + 1, (h_original - 1) // 32 + 1
        if return_all:
            return F.normalize(X, p=2, dim=1), y, z
        else:
            return F.normalize(X, p=2, dim=1)

    def forward_test_from_exemplar(self, X, return_all=False):
        # Feature map n, c, w, h -- 1, 128, 128, 128
        res = self.net.from_deep_exemplar(X, return_all=return_all)
        if return_all:
            X, y, z = res
        else:
            X = res

        if self.output_dimension == -1:
            return F.normalize(X, p=2, dim=1)
        if self.size_number == 1:
            X = torch.nn.functional.conv2d(
                X,
                self.out_layer.weight.unsqueeze(2).unsqueeze(3),
            )
        elif self.size_number > 1:
            X = torch.nn.functional.conv2d(
                X,
                self.out_layer.weight.view(
                    self.output_dimension,
                    256,
                    self.size_number,
                )
                .permute(2, 0, 1)
                .reshape(
                    self.size_number * self.output_dimension,
                    256,
                )
                .unsqueeze(2)
                .unsqueeze(3),
            )
        # n, c, w, h
        # 1, 128, (w_original - 1) // 32 + 1, (h_original - 1) // 32 + 1
        if return_all:
            return F.normalize(X, p=2, dim=1), y, z
        else:
            return F.normalize(X, p=2, dim=1)

    def forward(
        self,
        X,
        kp,
        mask=None,
        return_maps=False,
        deep=False,
        ds_mask=False,
        bottleneck=False,
    ):
        keypoint_positions = kp
        obj_mask = mask
        # X=torch.ones(1, 3, 224, 300), kps = torch.tensor([[(36, 40), (90, 80)]])
        # n images, k keypoints and 2 states.
        # Keypoint input -> n * k * 2 (k keypoints for n images) (must be position on original image)
        if bottleneck:
            return self.net.bottleneck(X)
        if return_maps:
            return self.net(X, return_all=return_maps)
        n = X.shape[0]  # n = 1

        # downsample_rate = 32
        if deep:
            m = self.net.from_deep_exemplar(X)
        else:
            m = self.net.forward(X)

        # N, C * local_size0 * local_size1, H * W
        X = self.converter(m)
        if ds_mask:
            keypoint_idx = keypoints_to_pixel_index(
                keypoints=keypoint_positions,
                downsample_rate=1,
                original_img_size=(80, 100),
            ).type(torch.long)
        else:
            keypoint_idx = keypoints_to_pixel_index(
                keypoints=keypoint_positions,
                downsample_rate=self.net_stride,
                original_img_size=self.img_shape,
            ).type(torch.long)
        # Never use this reduce_function part.
        if self.reduce_function:
            X = self.reduce_function(X)

        if self.n_noise_points == 0 or True:
            keypoint_all = keypoint_idx
        else:
            if obj_mask is not None:
                if ds_mask == False:
                    obj_mask = F.max_pool2d(
                        obj_mask.unsqueeze(dim=1),
                        kernel_size=self.net_stride,
                        stride=self.net_stride,
                        padding=(self.net_stride - 1) // 2,
                    )
                obj_mask = obj_mask.view(obj_mask.shape[0], -1)
                assert obj_mask.shape[1] == X.shape[2], (
                    "mask_: " + str(obj_mask.shape) + " feature_: " + str(X.shape)
                )
            if self.noise_on_mask:
                keypoint_noise = get_noise_pixel_index(
                    keypoint_idx,
                    max_size=X.shape[2],
                    n_samples=self.n_noise_points,
                    obj_mask=obj_mask,
                )
            else:
                keypoint_noise = get_noise_pixel_index(
                    keypoint_idx,
                    max_size=X.shape[2],
                    n_samples=self.n_noise_points,
                    obj_mask=obj_mask,
                )
            keypoint_all = torch.cat((keypoint_idx, keypoint_noise), dim=1)

        # N, C * local_size0 * local_size1, H * W -> N, H * W, C * local_size0 * local_size1
        X = torch.transpose(X, 1, 2)

        # N, H * W, C * local_size0 * local_size1 -> N, keypoint_all, C * local_size0 * local_size1
        X = batched_index_select(X, dim=1, inds=keypoint_all)

        # L2norm, fc layer, -> dim along d
        if self.out_layer is None:
            X = F.normalize(X, p=2, dim=2)
            X = X.view(n, -1, 256)
        else:
            X = F.normalize(self.out_layer(X), p=2, dim=2)
            X = X.view(n, -1, self.out_layer.weight.shape[0])

        return X

    def cuda(self, device=None):
        self.net.cuda(device=device)
        self.converter.cuda(device=device)
        if self.output_dimension != -1:
            self.out_layer.cuda(device=device)
        return self
