from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import uvicorn

app = FastAPI()

model = load_model("models/hotdog/keras_model.h5", compile=False)
class_names = open("models/hotdog/labels.txt", "r", encoding="utf-8").readlines()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    if " " in class_name:
        class_name = class_name.split(" ", 1)[1]
    confidence = float(prediction[0][index])

    return {
        "class_name": class_name,
        "confidence": round(confidence * 100, 2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
