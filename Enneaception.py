import torch
import torch.nn as nn
import torch.nn.functional as F
import inception_short

def enneaception(pretrained_inception_dict=None, **kwargs):
    if pretrained_inception_dict is None:
        return Enneaception(**kwargs)
    else:
        net = Enneaception(**kwargs)
        net.inception_xs.load_state_dict(pretrained_inception_dict, strict=False)
        return net

class Enneaception(nn.Module):

    def __init__(self, num_classes=1000, transform_input=False):
        super().__init__()
        self.inception_xs = inception_short.Inception3XS(num_classes=num_classes, transform_input=transform_input)
        self.fc = nn.Linear(9*num_classes, num_classes, bias=True)

    def forward(self, x):
        # n x 3 x 897 x 897
        img0 = self.inception_xs(x[:,:,:299,:299])
        img1 = self.inception_xs(x[:,:,299:598,:299])
        img2 = self.inception_xs(x[:,:,598:,:299])
        img3 = self.inception_xs(x[:,:,:299,299:598])
        img4 = self.inception_xs(x[:,:,299:598,299:598])
        img5 = self.inception_xs(x[:,:,598:,299:598])
        img6 = self.inception_xs(x[:,:,:299,598:])
        img7 = self.inception_xs(x[:,:,299:598,598:])
        img8 = self.inception_xs(x[:,:,598:,598:])
        # 9 of n x 1000
        out = torch.cat([img0, img1, img2, img3, img4, img5, img6, img7, img8], dim=1)
        # n x 9000
        out = self.fc(out)
        return out