from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64

app = FastAPI()

@app.post("/annotate")
async def annotate(
    defective_image: UploadFile = File(...),
    good_json: str = Form(...),
    defective_json: str = Form(...)
):
    # Read image
    img_bytes = await defective_image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # --- Example: draw a red box (you will replace with IoU logic) ---
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 3)
    cv2.putText(img, "Missing Part", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Encode back
    _, buffer = cv2.imencode(".jpg", img)
    encoded = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={"image_base64": encoded})
