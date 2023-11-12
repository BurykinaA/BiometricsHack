import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"


class CustomModel2(nn.Module):
    def __init__(self):
        super(CustomModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(128 * 8 * 8, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = x

        # First input
        x1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x1 = self.pool2(F.relu(self.bn2(self.conv2(x1))))
        x1 = x1.view(x.size(0), -1)
        x1 = self.dropout(self.bn3(self.dense(x1)))
        x = self.fc(x1)

        return x


model = CustomModel2()
model.load_state_dict(
    torch.load(
        r"D:\sber_hack\backend\models_weights\model_conv.pth",
        map_location=torch.device("cpu"),
    )
)
model.eval()


def get_score_conv(X):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)

    return 0 if predicted else 1
