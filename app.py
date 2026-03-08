from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import transforms

from model import load_model

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = ["NORMAL","PNEUMONIA"]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(outputs,dim=1)

        confidence, pred = torch.max(probs,1)

    result = classes[pred.item()]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result,
            "confidence": round(float(confidence)*100,2)
        }
    )