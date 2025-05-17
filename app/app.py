# Importing essential libraries and modules
from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import requests
import torch
from torchvision import transforms
from PIL import Image
from markupsafe import Markup
import warnings
import joblib
import pickle
import io  # Added missing import
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9
import openai

import config  # For weather API key and other configurations
from openai import OpenAI
from flask import jsonify

client = OpenAI(api_key="sk-eLNufXkRYaYKuaDQbxrP1-OXwvr2z3SmGsGVZo9S2NT3BlbkFJF1FDv0HRhZRvgrPNghvaavmi_3poJgAVbYtyCcXg4A")
# ==============================================================================================


# Loading plant disease classification model
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                   'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                   'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Load crop recommendation model
crop_recommendation_model_path = 'C:\\Users\\91891\\Desktop\\Harvestify\\app\\models\\RandomForest.pkl'

try:
    crop_recommendation_model = joblib.load(crop_recommendation_model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading crop recommendation model: {e}")
    crop_recommendation_model = None  # Set to None to prevent further errors


# =========================================================================================

# Custom functions for calculations
def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] == 200:
        main_data = data["main"]
        temperature = round((main_data["temp"] - 273.15), 2)  # Kelvin to Celsius
        humidity = main_data["humidity"]
        return temperature, humidity
    return None
# ✅ Predict Disease from Image
def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label.
    :param img: image file
    :param model: PyTorch model for prediction
    :return: prediction (str)
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image = Image.open(io.BytesIO(img)).convert('RGB')  # Ensuring image is RGB
        img_t = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(img_t)
        _, preds = torch.max(output, dim=1)
        return disease_classes[preds[0].item()]
    except Exception as e:
        return str(e)

# ===============================================================================================

# ------------------------------------ FLASK APP -------------------------------------------------
# Flask App Initialization
app = Flask(__name__)


# Home page route
@app.route('/')
def home():
    return render_template('index.html', title='PrefectCrop - Home')

# Crop recommendation page route
@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='PrefectCrop - Crop Recommendation')

# Fertilizer recommendation page route
@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='PrefectCrop - Fertilizer Suggestion')

# ===============================================================================================

# ✅ Crop Prediction
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    
    title = 'PrefectCrop - Crop Recommendation'

    if request.method == 'POST':
        if crop_recommendation_model is None:
            return render_template('error.html', title=title, error="Crop recommendation model is not loaded. Please check the model file.")

        # Get input from form
        try:
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            city = request.form.get("city")

            weather_data = weather_fetch(city)
            if weather_data:
                temperature, humidity = weather_data
                data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                prediction = crop_recommendation_model.predict(data)
                final_prediction = prediction[0]
                return render_template('crop-result.html', prediction=final_prediction, title=title)
            else:
                return render_template('try_again.html', title=title)
        except Exception as e:
            return render_template('error.html', title=title, error=f"Error during prediction: {e}")

    # Fertilizer recommendation result page route
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'PrefectCrop - Fertilizer Suggestion'
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    
    # Load fertilizer data from CSV
    df = pd.read_csv(r'C:\Users\Subhan D\OneDrive\Desktop\Harvestify\Data-raw\FertilizerData.csv')

    try:
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        # Calculate deficiency
        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]

        if max_value == "N":
            key = 'NHigh' if n < 0 else "Nlow"
        elif max_value == "P":
            key = 'PHigh' if p < 0 else "Plow"
        else:
            key = 'KHigh' if k < 0 else "Klow"

        response = Markup(str(fertilizer_dic[key]))
        return render_template('fertilizer-result.html', recommendation=response, title=title)
    except Exception as e:
        return render_template('error.html', title=title, error="Error with fertilizer data: " + str(e))


# Disease prediction result page route
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'PrefectCrop - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic.get(prediction, "No recommendation found.")))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            return render_template('error.html', title=title, error="Error during prediction: " + str(e))

    return render_template('disease.html', title=title)

# ===============================================================================================
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_message = request.json["message"]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"reply": "Sorry, I couldn't process that request."})
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    app.run(debug=True)