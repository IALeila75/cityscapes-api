# app.py
import os
import io
from flask import Flask, request, send_file, render_template_string
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)

# Load TFLite model
TFLITE_MODEL_PATH = "model/model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cityscapes palette
CITYSCAPES_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]
 
def predict_mask_tflite(image, input_size=(128, 256)):
    image = image.resize(input_size[::-1])
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    mask = np.argmax(output[0], axis=-1).astype(np.uint8)

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(CITYSCAPES_PALETTE):
        color_mask[mask == i] = color
    return Image.fromarray(color_mask)

@app.route("/")
def home():
    return "✅ Cityscapes API Flask fonctionne sur Render"

@app.route("/form", methods=["GET"])
def form():
    return '''
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Cityscapes Segmentation</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #f4f4f4;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            h2 {
                color: #333;
                margin-bottom: 20px;
            }
            form {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            input[type="file"] {
                margin-bottom: 20px;
                font-size: 16px;
            }
            input[type="submit"] {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h2>Uploader une image Cityscapes (.png)</h2>
        <form action="/dashboard" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/png" required><br>
            <input type="submit" value="Prédire le masque">
        </form>
    </body>
    </html>
    '''


@app.route("/dashboard", methods=["POST"])
def dashboard():
    if 'image' not in request.files:
        return "Aucune image envoyée", 400

    import time
    start_time = time.time()

    image_file = request.files['image']
    original = Image.open(image_file).convert("RGB")
    mask = predict_mask_tflite(original)
    mask = mask.resize(original.size)
    
    # Créer la superposition
    overlay = Image.blend(original.convert("RGBA"), mask.convert("RGBA"), alpha=0.5)
    overlay_path = "static/latest_overlay.png"
    os.makedirs("static", exist_ok=True)
    overlay.save(overlay_path)

    elapsed_time = round(time.time() - start_time, 2)

    original_io = io.BytesIO()
    mask_io = io.BytesIO()
    overlay_io = io.BytesIO()

    original.save(original_io, format='PNG')
    mask.save(mask_io, format='PNG')
    os.makedirs("static", exist_ok=True)
    overlay.save(overlay_io, format='PNG')

    original_io.seek(0)
    mask_io.seek(0)
    overlay_io.seek(0)

    html = f'''
    <html>
    <head>
        <title>Dashboard Prédiction</title>
        <style>
            img {{ max-width: 45%; height: auto; margin: 10px; }}
            .container {{ display: flex; flex-direction: row; justify-content: center; align-items: center; flex-wrap: wrap; }}
        </style>
      </head>
      <body>
          <h2 style="text-align:center;">Résultat de la segmentation</h2>
          <p style="text-align:center;">⏱️ Temps de prédiction : {elapsed_time} secondes</p>
          <div class="container">
              <div>
                  <h4>Image originale</h4>
                  <img src="data:image/png;base64,{base64_img(original_io)}">
              </div>
              <div>
                  <h4>Masque prédit</h4>
                  <img src="data:image/png;base64,{base64_img(mask_io)}">
              </div>
              <div>
                  <h4>Superposition</h4>
                  <img src="data:image/png;base64,{base64_img(overlay_io)}">
              </div>
           </div>
           <div style="text-align:center;">
               <a href="/form">↩ Retour</a><br>
               <a href="/download_mask"><button>📥 Télécharger le masque</button></a><br>
               <a href="/download_overlay"><button>📥 Télécharger la superposition</button></a>
           </div>
      </body>
      </html>
      '''
    return render_template_string(html)
    

def base64_img(io_buffer):
    import base64
    return base64.b64encode(io_buffer.read()).decode('utf-8')

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return "❌ Aucun fichier image reçu", 400
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    mask_image = predict_mask_tflite(image)
    mask_image = mask_image.resize(image.size)

    # Sauvegarder temporairement le masque sur disque
    output_path = "static/latest_mask.png"
    os.makedirs("static", exist_ok=True)
    mask_image.save(output_path)

    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")

@app.route("/download_mask")
def download_mask():
    mask_path = "static/latest_mask.png"
    if os.path.exists(mask_path):
        return send_file(mask_path, as_attachment=True)
    else:
        return "❌ Aucun masque disponible pour le téléchargement.", 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
