import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor, disc_out = 512):
        super(Discriminator, self).__init__()
        discriminator = nn.ModuleDict()
        discriminator["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.utils.weight_norm(nn.Conv1d(1, ndf, kernel_size=15, stride=1)),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, disc_out)

            discriminator["layer_%d" % n] = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride
                    padding=stride * 5,
                    groups=nf_prev // 4,
                )),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, disc_out)
        discriminator["layer_%d" % (n_layers + 1)] = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d((nf_prev, nf, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )

        discriminator["layer_%d" % (n_layers + 2)] = nn.utils.weight_norm(nn.Conv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        ))
        self.discriminator = discriminator

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for key, module in self.discriminator.items():
            x = module(x)
            features.append(x)
        return featuress[:-1], features[-1]


if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    features, score = model(x)
    for feat in features:
        print(feat.shape)
    print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)