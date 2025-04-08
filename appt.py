from flask import Flask, request, render_template, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from utils import apply_cityscapes_palette

app = Flask(__name__)

# Charger le modèle .tflite
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

# Obtenir les détails du modèle
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/', methods=['GET'])
def home():
    return '✅ API Flask Render fonctionne !'
    
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Lire dynamiquement la taille attendue
    input_shape = input_details[0]['shape']  # (1, height, width, 3)
    height, width = input_shape[1], input_shape[2]

    # Ouvrir et redimensionner l'image
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((width, height))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Masque prédit (argmax)
    pred_mask = np.argmax(output_data[0], axis=-1).astype(np.uint8)

    # Appliquer la palette colorée
    mask_img = apply_cityscapes_palette(pred_mask)

    # Sauvegarder le masque colorisé
    os.makedirs("static/outputs", exist_ok=True)
    mask_path = "static/outputs/mask.png"
    mask_img.save(mask_path)

    return render_template_string(f"""
        <h2>Masque généré</h2>
        <img src='/{mask_path}' width=512><br><br>
        <a href='/'>Retour</a>
    """)

# Lancer l'API Flask
if __name__ == "__main__":
    app.run(debug=True)
 