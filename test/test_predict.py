import requests

# Assure-toi que ce fichier existe dans le même dossier que ce script
#image_path = r"C:\Users\CE PC\Documents\IA_P8\flaskproject\berlin_000019_000019_leftImg8bit.png"

with open(berlin_000019_000019_leftImg8bit.png, "rb") as img_file:
    response = requests.post("http://127.0.0.1:5000/predict", files={"file": img_file})
    print(response.json())
