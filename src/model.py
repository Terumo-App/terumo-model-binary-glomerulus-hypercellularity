from torch import nn
from efficientnet_pytorch import EfficientNet

from config import settings

class Net(nn.Module):
    def __init__(self, net_version, num_classes, freeze: bool = False):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Sequential(
            nn.Linear(settings.model.fcLayer, num_classes),
        )
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                if not name.startswith("_fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)