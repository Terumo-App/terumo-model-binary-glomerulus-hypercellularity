from torch import nn
from efficientnet_pytorch import EfficientNet
from config import settings

class Net(nn.Module):
    def __init__(self, net_version, num_classes):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Sequential(
            nn.Linear(settings.fcLayer, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)