from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import shutil
import io
import base64

# Initialize the FastAPI app
app = FastAPI()

# Initialize Jinja2Templates for rendering HTML files
templates = Jinja2Templates(directory="templates")

# Initialize the Inference client with your API URL and API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="nmoxDf7wfxJEIYLkLFDA"
)

# Route to serve the index.html template
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle the image upload and perform inference
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open(file.filename, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)

    # Perform inference on both models
    damage_result = CLIENT.infer(file.filename, model_id="car-damage-detection-u0q4r/1")
    severity_result = CLIENT.infer(file.filename, model_id="car-damage-severity-detection-cardd/1")

    # Extract predictions from the results
    damage_predictions = damage_result.get("predictions", [])
    severity_predictions = severity_result.get("predictions", [])

    # Initialize a dictionary to store the final predictions, avoiding duplication
    final_predictions = {}

    # Process damage predictions and add them to the final predictions
    for prediction in damage_predictions:
        if "x" in prediction and "y" in prediction and "width" in prediction and "height" in prediction:
            cls = prediction["class"]
            if cls not in final_predictions:
                final_predictions[cls] = []
            final_predictions[cls].append({
                "x": prediction["x"],
                "y": prediction["y"],
                "width": prediction["width"],
                "height": prediction["height"]
            })

    # Process severity predictions and merge with damage predictions
    for prediction in severity_predictions:
        if "x" in prediction and "y" in prediction and "width" in prediction and "height" in prediction:
            cls = prediction["class"]
            if cls not in final_predictions:
                final_predictions[cls] = []
            final_predictions[cls].append({
                "x": prediction["x"],
                "y": prediction["y"],
                "width": prediction["width"],
                "height": prediction["height"]
            })

    # Handle conflicts: if "moderate" and "severe" predictions exist for the same class, prioritize "moderate"
    for cls in final_predictions:
        if "moderate" in cls and "severe" in cls:
            moderate_predictions = [pred for pred in final_predictions[cls] if "moderate" in pred["class"]]
            severe_predictions = [pred for pred in final_predictions[cls] if "severe" in pred["class"]]
            final_predictions[cls] = moderate_predictions + severe_predictions  # Prioritize moderate

    # Open the image using PIL
    image = Image.open(file.filename)
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes for all predictions (damage and severity)
    for cls, predictions in final_predictions.items():
        for prediction in predictions:
            if "x" in prediction and "y" in prediction and "width" in prediction and "height" in prediction:
                x = prediction["x"]
                y = prediction["y"]
                width = prediction["width"]
                height = prediction["height"]
                draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

    # Convert the image to a byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Convert the byte stream to Base64
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    # Get unique class names from the final predictions
    classes = list(final_predictions.keys())

    # Return the class names and the annotated image as a Base64 string
    return {"detected_results": final_predictions, "annotated_image": img_base64}


