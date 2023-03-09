import torch
import torchvision


class MnasMulti(torch.nn.Module):
    def __init__(self, output_depths, pretrained=True):
        super().__init__()
        MNASNet = torchvision.models.mnasnet1_0(pretrained=pretrained)
        self.conv0 = torch.nn.Sequential(
            MNASNet.layers._modules["0"],
            MNASNet.layers._modules["1"],
            MNASNet.layers._modules["2"],
            MNASNet.layers._modules["3"],
            MNASNet.layers._modules["4"],
            MNASNet.layers._modules["5"],
            MNASNet.layers._modules["6"],
            MNASNet.layers._modules["7"],
            MNASNet.layers._modules["8"],
        )
        self.conv1 = MNASNet.layers._modules["9"]
        self.conv2 = MNASNet.layers._modules["10"]

        final_chs = 80

        self.inner1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_depths[1]),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(output_depths[1], final_chs, 1, bias=False),
        )
        self.inner2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_depths[2]),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(output_depths[2], final_chs, 1, bias=False),
        )

        self.out1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(final_chs),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(final_chs, output_depths[0], 1, bias=False),
        )
        self.out2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(final_chs),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(final_chs, output_depths[1], 3, bias=False, padding=1),
        )
        self.out3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(final_chs),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(final_chs, output_depths[2], 3, bias=False, padding=1),
        )

        torch.nn.init.kaiming_normal_(self.inner1[2].weight)
        torch.nn.init.kaiming_normal_(self.inner2[2].weight)
        torch.nn.init.kaiming_normal_(self.out1[2].weight)
        torch.nn.init.kaiming_normal_(self.out2[2].weight)
        torch.nn.init.kaiming_normal_(self.out3[2].weight)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["coarse"] = out

        intra_feat = torch.nn.functional.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=False
        ) + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["medium"] = out

        intra_feat = torch.nn.functional.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=False
        ) + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["fine"] = out

        return outputs
