import os
import json
import requests
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Roboflow REST API details
MODEL_ID = "cattle-buffalo-breeds-q5sgq-79kyd-lijeg/1"
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Gemini API call
def get_breed_info(breed_name: str) -> dict:
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "API key not configured."}

        genai.configure(api_key=api_key)
        # prompt = f"""
        # Provide information about the "{breed_name}" cow breed.
        # Return the information ONLY as a valid JSON object with keys:
        # "name", "height_range", "weight_range", "life_span", "price".
        # """
        prompt = f"""
        Provide detailed information about the "{breed_name}" {breed_name.lower()} breed.
        Return the information ONLY as a valid JSON object with the following keys:
        - "name": Breed name
        - "height_range": Typical height range at the shoulder
        - "weight_range": Typical weight range
        - "life_span": Average lifespan
        - "price": Typical market price for a quality animal
        - "milk_yield": Average milk production per year (if applicable, otherwise "N/A")
        - "milk_fat": Average milk fat content (percentage, if applicable, otherwise "N/A")
        - "temperament": General temperament (e.g., docile, aggressive)
        - "coat_color": Common coat colors
        - "uses": Primary uses (e.g., dairy, beef, dual-purpose)
        - "origin": Country or region of origin
        - "adaptability": How well the breed adapts to different climates
        """

        model_gen = genai.GenerativeModel('gemini-2.5-pro')
        response = model_gen.generate_content(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_text)
    except Exception as e:
        return {"error": str(e)}

# Roboflow detection using REST API
def detect_breed_from_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}",
                files={"file": f}
            )
        result = response.json()
        predictions = result.get("predictions", [])
        if predictions:
            top_pred = max(predictions, key=lambda x: x["confidence"])
            return top_pred["class"].lower()
        else:
            return "jersey"  # default if nothing detected
    except Exception as e:
        print(f"Detection error: {e}")
        return "jersey"

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    breed_info = None
    uploaded_image_url = None
    default_used = False

    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file.filename != "":
                filename = secure_filename(image_file.filename.lower())
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                
                # Set URL for template
                uploaded_image_url = f"/uploads/{filename}"

                # Detect breed
                detected_breed = detect_breed_from_image(image_path)
                if detected_breed == "jersey":
                    default_used = True

                # Get breed info from Gemini
                breed_info = get_breed_info(detected_breed)
                breed_info["default_used"] = default_used

    return render_template("index.html", breed_info=breed_info, uploaded_image_url=uploaded_image_url)

if __name__ == "__main__":
    app.run(debug=True)
