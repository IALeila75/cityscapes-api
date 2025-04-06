import requests

file_path = "berlin_000019_000019_leftImg8bit.jpg"

with open(file_path, "rb") as f:
    response = requests.post("http://127.0.0.1:5000/predict", files={"file": f})
    print(response.text)  # HTML ou JSON selon la réponse
