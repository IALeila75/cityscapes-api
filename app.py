# app.py
import os
import io
import numpy as np
from flask import Flask, request, send_file, render_template_string
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
TFLITE_MODEL_PATH = "model/model.tflite"

# Chargement du modèle TFLite
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
    return "✅ API Cityscapes prête à l’emploi"

@app.route("/form")
def form():
    return '''
    <html><body>
        <h2>Uploader une image Cityscapes (.png)</h2>
        <form action="/dashboard" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/png"><br><br>
            <input type="submit" value="Prédire">
        </form>
    </body></html>
    '''

@app.route("/dashboard", methods=["POST"])
def dashboard():
    if 'image' not in request.files:
        return "❌ Aucune image envoyée", 400

    image_file = request.files['image']
    original = Image.open(image_file).convert("RGB")
    mask = predict_mask_tflite(original)
    mask = mask.resize(original.size)

    # Superposition (blend)
    overlay = Image.blend(original, mask, alpha=0.5)

    # Sauvegarde
    os.makedirs("static", exist_ok=True)
    mask_path = "static/latest_mask.png"
    overlay_path = "static/latest_overlay.png"
    mask.save(mask_path)
    overlay.save(overlay_path)

    def encode_image(img):
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        import base64
        return base64.b64encode(buffer.read()).decode('utf-8')

    html = f'''
    <html>
    <head><title>Dashboard</title></head>
    <body style="text-align:center;font-family:sans-serif;">
        <h2>Résultat de la prédiction</h2>
        <div style="display:flex; justify-content:center; gap:20px;">
            <div><h4>Image originale</h4><img src="data:image/png;base64,{encode_image(original)}" width="250"></div>
            <div><h4>Masque prédit</h4><img src="data:image/png;base64,{encode_image(mask)}" width="250"></div>
            <div><h4>Superposition</h4><img src="data:image/png;base64,{encode_image(overlay)}" width="250"></div>
        </div>
        <br><a href="/form">↩ Nouvelle prédiction</a><br>
        <a href="/download_mask">📥 Télécharger le masque</a>
    </body></html>
    '''
    return render_template_string(html)

@app.route("/download_mask")
def download_mask():

    mask_path = "static/latest_mask.png"
    if os.path.exists(mask_path):
        return send_file(mask_path, as_attachment=True)
    else:
        return "❌ Aucun masque disponible à télécharger.", 404
    #return send_file(mask_path, as_attachment=True) if os.path.exists(mask_path) else "❌ Masque non trouvé", 404

if __name__ == '__main__':
     port = int(os.environ.get("PORT", 5000))
     app.run(host="0.0.0.0", port=port)
