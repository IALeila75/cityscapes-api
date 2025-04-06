from flask import Flask, request, jsonify, render_template, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uuid
#import segmentation_models_tf as sm

app = Flask(__name__)
os.makedirs("static/outputs", exist_ok=True)

# Charger le modèle
MODEL_PATH = os.path.join("model", "best_fpn_mobilenet_categorical_crossentropy.keras")
#import segmentation_models as sm

# Spécifier les custom objects au moment du chargement


model = tf.keras.models.load_model(MODEL_PATH , compile=False)

# Paramètres
IMAGE_SIZE = (256, 128)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction
        #pred = model.predict(image_array)
        #mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
        
        from utils import apply_cityscapes_palette
        # ... dans la route /predict :
        pred = np.argmax(output_data[0], axis=-1).astype(np.uint8)
        mask = apply_cityscapes_palette(pred_mask)
        mask.save("static/outputs/mask.png")

        # Conversion en image
        mask_img = Image.fromarray(mask * (255 // 18))  # simple grayscale scaling
        mask_id = str(uuid.uuid4())
        output_path = f"static/outputs/mask_{mask_id}.png"
        mask_img.save(output_path)

        return render_template_string(f"""
        <h2>Masque généré ✅</h2>
        <img src="/{output_path}" style="max-width:500px;">
        <br><a href="/">Retour</a>
        """)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
