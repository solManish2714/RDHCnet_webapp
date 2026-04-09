from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from preventions import prevention_tips
from class_labels import class_names
import os
from PIL import UnidentifiedImageError

app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Model once
model = load_model(
    "best_cropnet_model",
    custom_objects={"top3_acc": TopKCategoricalAccuracy(k=3)}
)

# ----------------- Prediction Function -----------------
def predict(img_path, top_k=3):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except UnidentifiedImageError:
        return "Invalid Image", 0, ["Please upload a valid image."], []

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    top_indices = prediction.argsort()[-top_k:][::-1]
    top_predictions = [(class_names[i], float(prediction[i]*100)) for i in top_indices]

    predicted_class, confidence = top_predictions[0]
    preventions = prevention_tips.get(predicted_class, ["No data available."])

    return predicted_class, confidence, preventions, top_predictions


# ----------------- Home Page -----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "diseases": class_names
    })


# ----------------- Prediction Route -----------------
@app.post("/predict", response_class=HTMLResponse)
async def predict_disease(request: Request, file: UploadFile = File(...)):

    # ✅ Validate file type
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please upload a valid image file (JPG, JPEG, PNG).",
            "diseases": class_names
        })

    file_location = f"temp_{file.filename}"

    try:
        # ✅ Save file properly
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # ✅ Predict
        disease, confidence, tips, top_preds = predict(file_location)

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}",
            "diseases": class_names
        })

    finally:
        # ✅ Always delete temp file
        if os.path.exists(file_location):
            os.remove(file_location)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "disease": disease,
        "confidence": round(confidence, 2),
        "tips": tips,
        "top_preds": top_preds,
        "diseases": class_names
    })