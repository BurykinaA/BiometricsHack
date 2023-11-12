import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

device = "cpu"

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.to(device)
model.load_state_dict(
    torch.load(
        r"D:\sber_hack\backend\models_weights\model_mobileNet.pth", map_location="cpu"
    )
)
model.eval()


def get_score_mobil(X):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)

    return 0 if predicted else 1
