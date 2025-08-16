from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import json
from io import BytesIO

app = FastAPI()

@app.post("/annotate/")
async def annotate(
    file: UploadFile = File(...),
    good_json: str = Form(...),
    defective_json: str = Form(...)
):
    # Load image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    good = json.loads(good_json)
    defective = json.loads(defective_json)

    # --- Example: draw all GOOD boxes in green ---
    for part in good:
        x, y, w, h = part["x"], part["y"], part["width"], part["height"]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    # --- Example: draw all DEFECTIVE boxes in red ---
    for part in defective:
        x, y, w, h = part["x"], part["y"], part["width"], part["height"]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

    # Return annotated image
    _, img_encoded = cv2.imencode('.jpg', img)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
