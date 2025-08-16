from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import json
import io

app = FastAPI()

# -------------------------------
# HARD-CODED GOOD JSON (your reference bonnet)
# -------------------------------
GOOD_JSON = {
  "predictions": [
    {
      "x": 118,
      "y": 579,
      "width": 22,
      "height": 24,
      "confidence": 0.924,
      "class": "spot",
      "class_id": 3,
      "detection_id": "b03ffb90-e49f-4e59-9a7d-9c3970de3490"
    },
    {
      "x": 351.5,
      "y": 285.5,
      "width": 27,
      "height": 29,
      "confidence": 0.921,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "70fb074e-d5fd-4cc9-9951-b9126269c205"
    },
    {
      "x": 176,
      "y": 770.5,
      "width": 16,
      "height": 21,
      "confidence": 0.913,
      "class": "spot",
      "class_id": 3,
      "detection_id": "24c11eeb-9e02-4bc6-be87-f96e97d03ee6"
    },
    {
      "x": 394,
      "y": 428.5,
      "width": 22,
      "height": 21,
      "confidence": 0.911,
      "class": "spot",
      "class_id": 3,
      "detection_id": "80284a48-2a4d-4863-b62f-7d0542d5780c"
    },
    {
      "x": 164,
      "y": 738,
      "width": 22,
      "height": 26,
      "confidence": 0.908,
      "class": "spot",
      "class_id": 3,
      "detection_id": "366fba04-7b4e-4be0-8d5b-8f1b5e2ae9f9"
    },
    {
      "x": 290.5,
      "y": 402.5,
      "width": 27,
      "height": 23,
      "confidence": 0.907,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "4a7b1e40-0a00-4740-980e-e48f39b5bdf3"
    },
    {
      "x": 196.5,
      "y": 1064.5,
      "width": 17,
      "height": 19,
      "confidence": 0.898,
      "class": "spot",
      "class_id": 3,
      "detection_id": "81d047a0-abea-4c17-8791-099530e79560"
    },
    {
      "x": 118.5,
      "y": 626.5,
      "width": 23,
      "height": 19,
      "confidence": 0.897,
      "class": "spot",
      "class_id": 3,
      "detection_id": "35a09413-1b4f-4d81-beb8-85cf74fffa9e"
    },
    {
      "x": 408,
      "y": 489.5,
      "width": 26,
      "height": 27,
      "confidence": 0.894,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "6381fe45-94ad-49aa-8d68-021154f81fcf"
    },
    {
      "x": 71.5,
      "y": 439.5,
      "width": 113,
      "height": 95,
      "confidence": 0.893,
      "class": "Bracket",
      "class_id": 0,
      "detection_id": "82b0ea4c-1167-4af0-ac53-337393ee8773"
    },
    {
      "x": 456,
      "y": 277.5,
      "width": 22,
      "height": 23,
      "confidence": 0.888,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "b0c3ef2f-c8c9-47dd-a1b6-319d078a4f98"
    },
    {
      "x": 119,
      "y": 545,
      "width": 22,
      "height": 24,
      "confidence": 0.874,
      "class": "spot",
      "class_id": 3,
      "detection_id": "2f7124d8-b4d3-4f8d-b11b-2bf62341c07b"
    },
    {
      "x": 229.5,
      "y": 754.5,
      "width": 17,
      "height": 21,
      "confidence": 0.871,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "cd4ef863-c9ef-4b14-8b2e-9d8df7cd5087"
    },
    {
      "x": 261,
      "y": 1017.5,
      "width": 22,
      "height": 15,
      "confidence": 0.871,
      "class": "spot",
      "class_id": 3,
      "detection_id": "737ee08a-30ec-4f70-b7d5-4a4342a63155"
    },
    {
      "x": 236.5,
      "y": 1188.5,
      "width": 21,
      "height": 15,
      "confidence": 0.867,
      "class": "spot",
      "class_id": 3,
      "detection_id": "4d0a9d6c-e2b5-4cac-9383-5173ab8decd0"
    },
    {
      "x": 391.5,
      "y": 412.5,
      "width": 15,
      "height": 15,
      "confidence": 0.867,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "dd6f93c1-77da-445b-9f87-164c3047ae57"
    },
    {
      "x": 169,
      "y": 430.5,
      "width": 24,
      "height": 27,
      "confidence": 0.863,
      "class": "spot",
      "class_id": 3,
      "detection_id": "185a52f2-c505-4575-bab9-b1b0e53af97e"
    },
    {
      "x": 193,
      "y": 851.5,
      "width": 22,
      "height": 21,
      "confidence": 0.863,
      "class": "spot",
      "class_id": 3,
      "detection_id": "c184548e-3548-4bf1-a873-3eed761b350f"
    },
    {
      "x": 298.5,
      "y": 1018,
      "width": 23,
      "height": 14,
      "confidence": 0.862,
      "class": "spot",
      "class_id": 3,
      "detection_id": "f7bcef44-f096-4eef-8188-1c24ca9bffa3"
    },
    {
      "x": 313.5,
      "y": 508,
      "width": 19,
      "height": 22,
      "confidence": 0.86,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "f9d9aa95-3998-4d06-a8f3-48414fa8bf14"
    },
    {
      "x": 192.5,
      "y": 980,
      "width": 17,
      "height": 16,
      "confidence": 0.86,
      "class": "spot",
      "class_id": 3,
      "detection_id": "d357f0bd-b983-4459-b7de-35fac120cdec"
    },
    {
      "x": 447.5,
      "y": 360,
      "width": 13,
      "height": 16,
      "confidence": 0.859,
      "class": "spot",
      "class_id": 3,
      "detection_id": "6d1706c6-8500-4d68-9080-8fdfcc54cca5"
    },
    {
      "x": 292.5,
      "y": 1065.5,
      "width": 23,
      "height": 11,
      "confidence": 0.852,
      "class": "spot",
      "class_id": 3,
      "detection_id": "c63bea75-e065-4da8-a07f-547275e889e8"
    },
    {
      "x": 442.5,
      "y": 418,
      "width": 13,
      "height": 20,
      "confidence": 0.849,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "1e289def-2c90-4867-b7bf-6b678dfff9c3"
    },
    {
      "x": 388,
      "y": 1041.5,
      "width": 26,
      "height": 9,
      "confidence": 0.848,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "a541e2ba-c6c0-49f7-91e5-fde5ab006068"
    },
    {
      "x": 247.5,
      "y": 327,
      "width": 25,
      "height": 20,
      "confidence": 0.847,
      "class": "spot",
      "class_id": 3,
      "detection_id": "d387fa21-1d92-4403-a298-30fd89efa117"
    },
    {
      "x": 334.5,
      "y": 1077.5,
      "width": 21,
      "height": 11,
      "confidence": 0.845,
      "class": "spot",
      "class_id": 3,
      "detection_id": "18608e66-885a-46f1-89f9-78dc80bd012a"
    },
    {
      "x": 288.5,
      "y": 470,
      "width": 17,
      "height": 18,
      "confidence": 0.845,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "ea16adf6-846a-4a4f-a398-12fa138b5d18"
    },
    {
      "x": 121,
      "y": 498.5,
      "width": 24,
      "height": 23,
      "confidence": 0.84,
      "class": "spot",
      "class_id": 3,
      "detection_id": "46387122-d8db-4ff1-a619-e4eb91a7b151"
    },
    {
      "x": 292.5,
      "y": 162,
      "width": 21,
      "height": 22,
      "confidence": 0.838,
      "class": "spot",
      "class_id": 3,
      "detection_id": "b47ae876-a1b5-4692-bd00-f832088da6f0"
    },
    {
      "x": 299.5,
      "y": 228.5,
      "width": 21,
      "height": 23,
      "confidence": 0.834,
      "class": "spot",
      "class_id": 3,
      "detection_id": "2c798226-1202-4141-aa73-215615ee621d"
    },
    {
      "x": 416.5,
      "y": 392.5,
      "width": 61,
      "height": 185,
      "confidence": 0.833,
      "class": "Bracket",
      "class_id": 0,
      "detection_id": "b48e0eba-1e74-43cc-8f7c-41b87c69777d"
    },
    {
      "x": 184.5,
      "y": 799.5,
      "width": 19,
      "height": 19,
      "confidence": 0.832,
      "class": "spot",
      "class_id": 3,
      "detection_id": "403bc34f-e9d5-4f6f-89cf-9f16a0b5c548"
    },
    {
      "x": 186.5,
      "y": 384.5,
      "width": 21,
      "height": 23,
      "confidence": 0.831,
      "class": "spot",
      "class_id": 3,
      "detection_id": "b4e52722-8979-4e92-a5ff-4c51806e4626"
    },
    {
      "x": 141,
      "y": 689,
      "width": 20,
      "height": 16,
      "confidence": 0.829,
      "class": "spot",
      "class_id": 3,
      "detection_id": "240d9c3d-d36f-4618-b6ea-d1f438e04394"
    },
    {
      "x": 284,
      "y": 258.5,
      "width": 24,
      "height": 19,
      "confidence": 0.828,
      "class": "spot",
      "class_id": 3,
      "detection_id": "176f0efd-cc9d-4955-b767-9e66b389b9a7"
    },
    {
      "x": 417,
      "y": 1029,
      "width": 20,
      "height": 18,
      "confidence": 0.826,
      "class": "spot",
      "class_id": 3,
      "detection_id": "b8612ab7-b799-45f6-b32a-2b6da386cd80"
    },
    {
      "x": 382.5,
      "y": 673.5,
      "width": 15,
      "height": 19,
      "confidence": 0.826,
      "class": "spot",
      "class_id": 3,
      "detection_id": "df99c544-6a50-4fdf-acd2-f13583775150"
    },
    {
      "x": 298.5,
      "y": 906,
      "width": 23,
      "height": 14,
      "confidence": 0.819,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "be3ba0b8-f05c-4b9e-b9cd-83054ecf763a"
    },
    {
      "x": 265,
      "y": 305.5,
      "width": 24,
      "height": 21,
      "confidence": 0.817,
      "class": "spot",
      "class_id": 3,
      "detection_id": "a78256ad-b3bc-482a-acce-b68c99fe1a61"
    },
    {
      "x": 346,
      "y": 1038,
      "width": 30,
      "height": 12,
      "confidence": 0.814,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "dceb0797-65b0-493a-bba4-ef19cb7146b6"
    },
    {
      "x": 202,
      "y": 945.5,
      "width": 20,
      "height": 19,
      "confidence": 0.811,
      "class": "spot",
      "class_id": 3,
      "detection_id": "05aedf03-c4b3-48ce-af62-f9de216f1373"
    },
    {
      "x": 577,
      "y": 40.5,
      "width": 18,
      "height": 17,
      "confidence": 0.804,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "9b4bc18c-ffc3-49fa-a1bf-43c8ef73fbd3"
    },
    {
      "x": 30,
      "y": 411.5,
      "width": 24,
      "height": 19,
      "confidence": 0.804,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "33bdbea7-ef21-473a-a6ed-f272f6f9c422"
    },
    {
      "x": 483.5,
      "y": 158,
      "width": 23,
      "height": 20,
      "confidence": 0.802,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "cd08a064-aaeb-4901-87ce-53a1858a5fee"
    },
    {
      "x": 186.5,
      "y": 1189,
      "width": 23,
      "height": 14,
      "confidence": 0.799,
      "class": "spot",
      "class_id": 3,
      "detection_id": "252023a6-677b-4dcc-9448-cc9efe40d460"
    },
    {
      "x": 403,
      "y": 354.5,
      "width": 18,
      "height": 15,
      "confidence": 0.799,
      "class": "spot",
      "class_id": 3,
      "detection_id": "46cf6451-b75f-4732-872e-34c07d9e171c"
    },
    {
      "x": 23.5,
      "y": 508,
      "width": 25,
      "height": 26,
      "confidence": 0.799,
      "class": "spot",
      "class_id": 3,
      "detection_id": "296f2c71-ed27-4956-ba49-3dfec0851fb2"
    },
    {
      "x": 217.5,
      "y": 1022,
      "width": 15,
      "height": 14,
      "confidence": 0.797,
      "class": "spot",
      "class_id": 3,
      "detection_id": "7fb96561-bdba-4b63-bd3a-c86db1039006"
    },
    {
      "x": 369.5,
      "y": 1078.5,
      "width": 25,
      "height": 13,
      "confidence": 0.795,
      "class": "spot",
      "class_id": 3,
      "detection_id": "24760dcb-0067-40d8-9c2b-018c97fee0d8"
    },
    {
      "x": 398.5,
      "y": 388,
      "width": 17,
      "height": 16,
      "confidence": 0.79,
      "class": "spot",
      "class_id": 3,
      "detection_id": "6c3062ae-cbac-49c2-b39e-24d2a4f48548"
    },
    {
      "x": 452,
      "y": 102,
      "width": 18,
      "height": 22,
      "confidence": 0.79,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "c9d8757f-0c5c-4d6f-8fe5-eabe3dadc7f4"
    },
    {
      "x": 421.5,
      "y": 1092,
      "width": 19,
      "height": 12,
      "confidence": 0.788,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "50b4822c-f67c-470a-b260-20156e8d74aa"
    },
    {
      "x": 331.5,
      "y": 690,
      "width": 15,
      "height": 14,
      "confidence": 0.785,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "208e3dc7-210c-462c-85ea-7cc57b2d5da5"
    },
    {
      "x": 299,
      "y": 1031,
      "width": 28,
      "height": 8,
      "confidence": 0.78,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "be2441a8-8d5c-4fa2-9690-b6abaa28b187"
    },
    {
      "x": 231,
      "y": 547,
      "width": 24,
      "height": 14,
      "confidence": 0.771,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "7a4b14cc-31f4-4f1b-9cc4-4c35f7180e37"
    },
    {
      "x": 196,
      "y": 1145,
      "width": 18,
      "height": 14,
      "confidence": 0.76,
      "class": "spot",
      "class_id": 3,
      "detection_id": "deeef66b-7f3c-41a3-bc39-ada5b069626a"
    },
    {
      "x": 378,
      "y": 1025,
      "width": 18,
      "height": 16,
      "confidence": 0.751,
      "class": "spot",
      "class_id": 3,
      "detection_id": "f7002322-0c2f-402d-ae6a-8b23cbbcc5e7"
    },
    {
      "x": 224.5,
      "y": 349.5,
      "width": 21,
      "height": 19,
      "confidence": 0.738,
      "class": "spot",
      "class_id": 3,
      "detection_id": "08829454-7e3c-4f5a-b433-664f62518466"
    },
    {
      "x": 300.5,
      "y": 753,
      "width": 19,
      "height": 16,
      "confidence": 0.727,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "0ba6e363-5f85-4aef-9837-46f7817c295b"
    },
    {
      "x": 289.5,
      "y": 1139,
      "width": 353,
      "height": 124,
      "confidence": 0.719,
      "class": "Bracket",
      "class_id": 0,
      "detection_id": "b7069c67-e050-4c01-a957-7a49f9313aba"
    },
    {
      "x": 590.5,
      "y": 74.5,
      "width": 19,
      "height": 17,
      "confidence": 0.715,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "e18e5dfa-2009-483b-88f4-6fef166b3b3c"
    },
    {
      "x": 209,
      "y": 370,
      "width": 20,
      "height": 18,
      "confidence": 0.703,
      "class": "spot",
      "class_id": 3,
      "detection_id": "c560bc29-0999-45f5-9446-093ad05de313"
    },
    {
      "x": 545,
      "y": 98,
      "width": 22,
      "height": 20,
      "confidence": 0.695,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "c70997d2-9f39-4cc3-a4be-15afc7bf3da4"
    },
    {
      "x": 363,
      "y": 987,
      "width": 20,
      "height": 12,
      "confidence": 0.692,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "75f42b29-f176-4465-a279-c2e2c41d6a50"
    },
    {
      "x": 343,
      "y": 1021.5,
      "width": 18,
      "height": 13,
      "confidence": 0.689,
      "class": "spot",
      "class_id": 3,
      "detection_id": "25e2b487-76b8-435a-9542-c2f01318e4fa"
    },
    {
      "x": 199.5,
      "y": 894,
      "width": 17,
      "height": 14,
      "confidence": 0.684,
      "class": "spot",
      "class_id": 3,
      "detection_id": "988fca45-f26d-4beb-9aa9-b0c224e1a23a"
    },
    {
      "x": 194,
      "y": 1118.5,
      "width": 20,
      "height": 17,
      "confidence": 0.675,
      "class": "spot",
      "class_id": 3,
      "detection_id": "047df0c5-822e-4d41-9999-415279461710"
    },
    {
      "x": 292.5,
      "y": 1051,
      "width": 15,
      "height": 10,
      "confidence": 0.609,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "6515b725-4db8-46cb-8317-dddeaba77e54"
    },
    {
      "x": 276,
      "y": 801.5,
      "width": 24,
      "height": 15,
      "confidence": 0.478,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "1f187f4e-e36f-481c-b11a-46511cf3e54c"
    },
    {
      "x": 195.5,
      "y": 1022,
      "width": 15,
      "height": 14,
      "confidence": 0.476,
      "class": "spot",
      "class_id": 3,
      "detection_id": "c11a96e7-e847-4953-b438-e82ed8cd3096"
    },
    {
      "x": 194,
      "y": 1089,
      "width": 20,
      "height": 16,
      "confidence": 0.422,
      "class": "spot",
      "class_id": 3,
      "detection_id": "5a6bd770-7990-4328-85b0-ac4171e4147f"
    }
  ]
}

# -------------------------------
# Utility: IOU for bounding box overlap
# -------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/annotate")
async def annotate(
    file: UploadFile = File(...),
    defective_json: str = Form(...)
):
    # Read defective image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Parse defective JSON
    defective = json.loads(defective_json)

    # Compare GOOD vs DEFECTIVE
    for good_pred in GOOD_JSON["predictions"]:
        gx, gy, gw, gh = good_pred["x"], good_pred["y"], good_pred["width"], good_pred["height"]
        gclass = good_pred["class"]

        good_box = [gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2]

        found = False
        for defect_pred in defective.get("predictions", []):
            if defect_pred["class"] == gclass:
                dx, dy, dw, dh = defect_pred["x"], defect_pred["y"], defect_pred["width"], defect_pred["height"]
                defect_box = [dx - dw/2, dy - dh/2, dx + dw/2, dy + dh/2]

                if iou(good_box, defect_box) > 0.5:  # bounding box overlap
                    found = True
                    break

        if not found:
            # Draw red box at GOOD location (missing part)
            pt1 = (int(good_box[0]), int(good_box[1]))
            pt2 = (int(good_box[2]), int(good_box[3]))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255), 4)

    # Encode image to return
    _, img_encoded = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
