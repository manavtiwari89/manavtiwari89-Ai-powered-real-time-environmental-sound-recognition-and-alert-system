import torch
import torch.nn as nn


class SoundCNN(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = x.flatten(1)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = SoundCNN(num_classes=9)

    x = torch.randn(32, 1, 64, 63)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
