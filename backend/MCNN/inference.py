import torch
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cpu"


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(128 * 8 * 8, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(512 + 512, 2)

    def forward(self, x, mtcnn, resnet):
        x = x
        x1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x1 = self.pool2(F.relu(self.bn2(self.conv2(x1))))
        x1 = x1.view(x.size(0), -1)
        x1 = self.dropout(self.bn3(self.dense(x1)))

        # Second input
        x2_list = torch.zeros((x.size(0), 512)).to(device)
        for i in range(x.size(0)):
            img = x[i].permute(1, 2, 0).cpu()  # Перемещение изображения на CPU
            detected_faces = mtcnn(img)
            if detected_faces is not None:
                detected_faces = detected_faces.unsqueeze(0).to(
                    device
                )  # add batch dimension
                resnet.eval()
                detected_faces = resnet(detected_faces)
            else:
                detected_faces = torch.zeros((1, 512)).to(
                    device
                )  # assuming the output of resnet is of shape (batch_size, 512)
            x2_list[i] = detected_faces

        x = torch.cat((x1, x2_list), dim=1)

        # Classification
        x = self.fc(x)

        return x


mtcnn_model = MTCNN().eval().cpu()  # MTCNN на CPU
resnet_model = InceptionResnetV1(pretrained="vggface2").eval().to("cpu")
model = CustomModel()
model.load_state_dict(
    torch.load(
        r"D:\sber_hack\backend\models_weights\model_mtcnn.pth",
        map_location=torch.device("cpu"),
    )
)
model.eval()


def get_score(X):
    with torch.no_grad():
        outputs = model(X, mtcnn_model, resnet_model)
        _, predicted = torch.max(outputs.data, 1)

    return 0 if predicted else 1
