from torch import Tensor, nn
from efficientnet_pytorch import EfficientNet

from config import settings

class Net(nn.Module):
    def __init__(self, net_version, num_classes, freeze: bool = False, p_dropout: float = 0):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(settings.model.fcLayer, num_classes),
        )
        self.p = p_dropout
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                if not name.startswith("_fc"):
                    param.requires_grad = False

    def forward(self, x: Tensor):
        embedding = self.backbone(x)
        if self.p > 0:
            y = nn.functional.dropout(embedding, self.p)  # forces dropout even in testing
        y = self.fc(y)
        return y