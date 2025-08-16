# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import json

app = FastAPI(title="OpenCV Annotate (IoU missing only)")

def decode_image_bytes(bytes_data: bytes):
    arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def encode_image_to_base64(img, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def center_to_xyxy(box):
    # box: dict with x,y,width,height where x,y are CENTERS
    x, y, w, h = float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])
    x0 = x - w / 2.0
    y0 = y - h / 2.0
    x1 = x + w / 2.0
    y1 = y + h / 2.0
    return [x0, y0, x1, y1]

def clip_box_to_image(box_xyxy, img_w, img_h):
    x0, y0, x1, y1 = box_xyxy
    x0 = max(0, min(img_w-1, int(round(x0))))
    y0 = max(0, min(img_h-1, int(round(y0))))
    x1 = max(0, min(img_w-1, int(round(x1))))
    y1 = max(0, min(img_h-1, int(round(y1))))
    return [x0, y0, x1, y1]

def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0, inter_x1 - inter_x0)
    ih = max(0, inter_y1 - inter_y0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

@app.post("/annotate")
async def annotate(
    file: UploadFile = File(...),
    good_json: str = Form(...),
    defective_json: str = Form(...),
    iou_threshold: float = Form(0.5)
):
    """
    Expects:
      - file: image upload (defective image)
      - good_json: stringified JSON (or JSON object) with "predictions": [...]
      - defective_json: stringified JSON (or JSON object) from Roboflow with "predictions": [...]
      - iou_threshold: float (default 0.5)
    Returns JSON:
      { "annotated": "<base64 string>", "stats": {...}, "missing_labels": [...] }
    """

    # 1) read image
    contents = await file.read()
    try:
        img = decode_image_bytes(contents)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"image decode failed: {str(e)}"})

    img_h, img_w = img.shape[:2]

    # 2) parse incoming JSON strings (accept either string or object)
    try:
        good_obj = json.loads(good_json) if isinstance(good_json, str) else good_json
    except Exception:
        # try to accept as already JSON object string inside request
        try:
            good_obj = json.loads(str(good_json))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"good_json parse failed: {str(e)}"})

    try:
        defective_obj = json.loads(defective_json) if isinstance(defective_json, str) else defective_json
    except Exception:
        try:
            defective_obj = json.loads(str(defective_json))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"defective_json parse failed: {str(e)}"})

    # 3) extract predictions arrays (support both {"predictions": [...]} and bare list)
    good_preds = good_obj.get("predictions") if isinstance(good_obj, dict) and "predictions" in good_obj else good_obj
    defective_preds = defective_obj.get("predictions") if isinstance(defective_obj, dict) and "predictions" in defective_obj else defective_obj

    if not isinstance(good_preds, list):
        return JSONResponse(status_code=400, content={"error": "good_json.predictions must be a list"})
    if not isinstance(defective_preds, list):
        return JSONResponse(status_code=400, content={"error": "defective_json.predictions must be a list"})

    # 4) index defective by class for faster search
    defective_by_class = {}
    for d in defective_preds:
        cls = str(d.get("class"))
        defective_by_class.setdefault(cls, []).append(d)

    # 5) For each GOOD box, find best IoU among same-class defective; if below threshold -> missing
    missing_labels = []
    detected_count = 0

    for g in good_preds:
        g_class = str(g.get("class"))
        try:
            g_xyxy = center_to_xyxy(g)  # assumes center format (Roboflow style)
        except Exception:
            # fallback: try treat as top-left width/height
            g_xyxy = [g.get("x"), g.get("y"), g.get("x") + g.get("width"), g.get("y") + g.get("height")]

        # Clip coords
        g_xyxy_clipped = clip_box_to_image(g_xyxy, img_w, img_h)

        candidates = defective_by_class.get(g_class, [])
        best_iou = 0.0
        for d in candidates:
            try:
                d_xyxy = center_to_xyxy(d)
            except Exception:
                d_xyxy = [d.get("x"), d.get("y"), d.get("x") + d.get("width"), d.get("y") + d.get("height")]
            d_xyxy_clipped = clip_box_to_image(d_xyxy, img_w, img_h)
            v = iou_xyxy(g_xyxy_clipped, d_xyxy_clipped)
            if v > best_iou:
                best_iou = v

        if best_iou >= float(iou_threshold):
            detected_count += 1
        else:
            # missing -> draw red box and label
            x0, y0, x1, y1 = g_xyxy_clipped
            # thickness and font scale scalable by image size (optional)
            thickness = max(1, int(round(min(img_w, img_h) / 400)))
            font_scale = max(0.4, min(1.0, min(img_w, img_h) / 1000))
            color = (0, 0, 255)  # BGR - red
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
            label = g_class
            # place text slightly above the box if possible
            ty = max(0, y0 - 6)
            cv2.putText(img, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, thickness-1), cv2.LINE_AA)
            missing_labels.append(label)

    # 6) encode and return base64 + stats + missing list
    try:
        annotated_b64 = encode_image_to_base64(img, ext=".jpg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"encode failed: {str(e)}"})

    stats = {
        "total_expected": len(good_preds),
        "detected": detected_count,
        "missing": len(missing_labels),
        "iou_threshold": float(iou_threshold)
    }

    return JSONResponse(content={
        "annotated": annotated_b64,
        "stats": stats,
        "missing_labels": missing_labels
    })


