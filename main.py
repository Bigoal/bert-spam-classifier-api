import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("Bigoal1/bert-spam-classifier-api")
model = AutoModelForSequenceClassification.from_pretrained("Bigoal1/bert-spam-classifier-api",ignore_mismatched_sizes=True)
model.to(device)
model.eval()
class TextRequest(BaseModel):
    text: str

def predict_text(text:str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)
        prob = torch.softmax(output.logits, dim=1)

    pred = prob.argmax(dim=1).item()
    probability = prob[0][pred].detach().cpu().item()
    return {"label": "spam" if pred == 1 else "ham", "probability": round(probability,4)}
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Spam Classifier Welcome Page</title>
        </head>
        <body style="text-align:center; font-family:Arial; padding:40px;">
            <h1>Welcome to the Spam Classifier</h1>
            <img src="/static/photo.jpg" alt="Photo" style="width:800px; border-radius:12px; margin:20px 0;" />
            <br>
            <a href="/predict">
                <button style="padding:12px 24px; font-size:16px; cursor:pointer;">Go to Predict</button>
            </a>
        </body>
    </html>
    """
@app.get("/predict", response_class=HTMLResponse)
def predict_page():
    return """
    <html>
        <head>
            <title>Input page</title>
        </head>
        <body style="text-align:left; font-family:Arial; padding:40px;">
            <h1>Enter text to classify something...</h1>
            <h2>Something like a reminder for a meeting or a fake discount or anything else...</h2>
            
            <p>Example: "Don't forget our meeting at 3pm tomorrow!"</p>
            <p>Example: "Congratulations! You've won a free cruise. Click here to claim."</p>

            <form action="/predict" method="post">
                <textarea name="text" rows="5" style="width:400px;"></textarea>
                <br><br>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form("")):
    if not text.strip():
        return """
        <html>
            <body style="font-family:Arial; padding:40px;">
                <h2>Text input cannot be empty.</h2>
                <a href="/predict">Try again</a>
            </body>
        </html>
        """
    result = predict_text(text)
    return f"""
    <html>
        <body style="font-family:Arial; padding:40px;">
            <h2>Prediction: {result['label']} ({result['probability']*100}%)</h2>
            <a href="/predict">Try again</a>
        </body>
    </html>
    """