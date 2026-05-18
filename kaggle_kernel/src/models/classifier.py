import torch.nn as nn
import timm

class SnakeClassifier(nn.Module):
    def __init__(self, num_classes=44, backbone="efficientnet_b3",
                 dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top(self, n=3):
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x))
