import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):

    def __init__(self, channels=3, num_classes=9):
        super(ConvModel, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=(4, 4)),    # image size -> 30
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=(4, 4)),         # image size -> 13
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),        # image size -> 5
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(256, 512, kernel_size=(2, 2)),        # image size -> 4
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc_1 = nn.Linear(512 * 4 * 4, 512)
        self.fc_2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x
