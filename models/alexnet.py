from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=0,
                dilation=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(12544, 6400),
            nn.ReLU(),
            nn.Dropout1d(p=0.5),
            nn.Linear(6400, 3200),
            nn.ReLU(),
            nn.Linear(3200, 101),
        )

    def forward(self, x):
        return self.stack(x)
