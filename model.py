import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cpu")

def load_model():

    model = models.densenet121(weights=None)

    model.classifier = nn.Sequential(
        nn.Linear(1024,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128,2)
    )

    checkpoint = torch.load("pneumonia_model.pth", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model