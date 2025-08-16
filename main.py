# main.py
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
import io
import os

app = FastAPI(title="Annotate with alignment (ORB+Homography)")

# -------------------------------
# Hard-coded GOOD JSON (exactly as provided by you)
# -------------------------------
GOOD_JSON = {
  "predictions": [
    {
      "x": 407.017,
      "y": 489.096,
      "width": 25.561,
      "height": 24.705,
      "confidence": 0.889,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "ca57fa5e-f047-444b-9289-0f3abdb6d86d"
    },
    {
      "x": 75.616,
      "y": 439.42,
      "width": 121.232,
      "height": 100.024,
      "confidence": 0.883,
      "class": "Bracket",
      "class_id": 0,
      "detection_id": "4522abaf-f393-4610-90de-c3e02f5d52b1"
    },
    {
      "x": 21.921,
      "y": 508.105,
      "width": 25.646,
      "height": 26.946,
      "confidence": 0.87,
      "class": "spot",
      "class_id": 3,
      "detection_id": "4f1749b4-52c0-4016-b224-28536b011515"
    },
    {
      "x": 229.531,
      "y": 755.791,
      "width": 16.78,
      "height": 21.116,
      "confidence": 0.867,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "3cec766b-3a24-4257-b436-bcee7311c43a"
    },
    {
      "x": 163.624,
      "y": 738.545,
      "width": 20.887,
      "height": 23.819,
      "confidence": 0.864,
      "class": "spot",
      "class_id": 3,
      "detection_id": "ad823eaf-9f49-492a-9cda-3ea60bf6dfb8"
    },
    {
      "x": 118.464,
      "y": 625.721,
      "width": 24.158,
      "height": 18.363,
      "confidence": 0.863,
      "class": "spot",
      "class_id": 3,
      "detection_id": "14b25cea-bdaf-40cf-ab55-834f05075a05"
    },
    {
      "x": 393.188,
      "y": 429.006,
      "width": 21.178,
      "height": 20.248,
      "confidence": 0.861,
      "class": "spot",
      "class_id": 3,
      "detection_id": "a552d786-ae20-4182-a914-10e197aa176e"
    },
    {
      "x": 117.631,
      "y": 578.353,
      "width": 23.908,
      "height": 23.46,
      "confidence": 0.86,
      "class": "spot",
      "class_id": 3,
      "detection_id": "553330c6-6faa-4504-99a8-0d0a5403525f"
    },
    {
      "x": 119.432,
      "y": 546.569,
      "width": 23.201,
      "height": 23.669,
      "confidence": 0.86,
      "class": "spot",
      "class_id": 3,
      "detection_id": "a4d93397-781c-42e5-98fc-f80ddbaecb4a"
    },
    {
      "x": 290.004,
      "y": 402.607,
      "width": 26.195,
      "height": 23.243,
      "confidence": 0.859,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "b87fa453-cb6d-4f92-9f69-e35df8d8e3e5"
    },
    {
      "x": 352.207,
      "y": 285.635,
      "width": 27.713,
      "height": 28.65,
      "confidence": 0.858,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "db55c265-e1df-411a-aae3-9322971cb170"
    },
    {
      "x": 274.898,
      "y": 801.919,
      "width": 22.057,
      "height": 14.3,
      "confidence": 0.857,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "a09e645a-bb78-4a65-b078-b126c0e05539"
    },
    {
      "x": 414.17,
      "y": 391.103,
      "width": 59.522,
      "height": 184.135,
      "confidence": 0.854,
      "class": "Bracket",
      "class_id": 0,
      "detection_id": "ae7a7b9d-d7da-4535-bff6-535870c34743"
    },
    {
      "x": 169.525,
      "y": 430.064,
      "width": 23.674,
      "height": 25.874,
      "confidence": 0.854,
      "class": "spot",
      "class_id": 3,
      "detection_id": "7426d223-1b2f-4840-8e5c-366d996516bb"
    },
    {
      "x": 141.267,
      "y": 689.415,
      "width": 19.942,
      "height": 15.639,
      "confidence": 0.853,
      "class": "spot",
      "class_id": 3,
      "detection_id": "cf3ca419-0a91-44d0-8f1f-d0b0090e9e21"
    },
    {
      "x": 455.55,
      "y": 276.765,
      "width": 22.985,
      "height": 23.813,
      "confidence": 0.85,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "e6a47c67-6aac-452c-b1df-451d46e97e81"
    },
    {
      "x": 264.641,
      "y": 304.32,
      "width": 23.552,
      "height": 20.235,
      "confidence": 0.849,
      "class": "spot",
      "class_id": 3,
      "detection_id": "aeeab558-f505-4ed5-96b2-377b14c269f1"
    },
    {
      "x": 27.946,
      "y": 411.563,
      "width": 21.944,
      "height": 18.707,
      "confidence": 0.847,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "ff1f9b9c-042a-4a31-88a5-4bf827440d3e"
    },
    {
      "x": 289.248,
      "y": 1139.145,
      "width": 355.944,
      "height": 128.647,
      "confidence": 0.847,
      "class": "Bracket",
      "class_id": 0,
      "detection_id": "5d9b0eea-5153-42fd-9d4d-354a4a61d579"
    },
    {
      "x": 331.31,
      "y": 690.854,
      "width": 14.904,
      "height": 14.501,
      "confidence": 0.842,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "b82f095a-2261-4fa5-985f-54d0af9be9b2"
    },
    {
      "x": 121.245,
      "y": 498.846,
      "width": 24.119,
      "height": 23.35,
      "confidence": 0.841,
      "class": "spot",
      "class_id": 3,
      "detection_id": "649aff3b-bf93-4ffa-ac03-e476bcbffca9"
    },
    {
      "x": 231.602,
      "y": 547.935,
      "width": 23.537,
      "height": 14.824,
      "confidence": 0.841,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "d395f14d-b218-4765-ae3c-07a7fc38912e"
    },
    {
      "x": 184.917,
      "y": 384.341,
      "width": 20.59,
      "height": 21.777,
      "confidence": 0.841,
      "class": "spot",
      "class_id": 3,
      "detection_id": "db7f4530-a47e-4af6-b8f3-f282069f0e67"
    },
    {
      "x": 389.704,
      "y": 1043.384,
      "width": 24.936,
      "height": 10.819,
      "confidence": 0.841,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "9d582457-029e-4284-bb23-8e4bcd7b1891"
    },
    {
      "x": 291.795,
      "y": 161.183,
      "width": 20.702,
      "height": 21.272,
      "confidence": 0.839,
      "class": "spot",
      "class_id": 3,
      "detection_id": "71a51bf3-d23f-44c0-a466-8ad6e14a288d"
    },
    {
      "x": 482.326,
      "y": 158.954,
      "width": 21.576,
      "height": 18.666,
      "confidence": 0.839,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "01de1302-6ca9-461f-a41c-e8b5d07bf838"
    },
    {
      "x": 202.052,
      "y": 947.165,
      "width": 20.879,
      "height": 18.808,
      "confidence": 0.838,
      "class": "spot",
      "class_id": 3,
      "detection_id": "6be0faca-25aa-4049-8670-3b3fe00f97fb"
    },
    {
      "x": 299.057,
      "y": 227.782,
      "width": 21.271,
      "height": 23.115,
      "confidence": 0.838,
      "class": "spot",
      "class_id": 3,
      "detection_id": "cc9f01cc-aefb-4dd1-ac95-534789f5168b"
    },
    {
      "x": 312.492,
      "y": 506.673,
      "width": 17.883,
      "height": 22.043,
      "confidence": 0.834,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "249cf442-258d-4d94-853d-28edbb4eeecd"
    },
    {
      "x": 237.196,
      "y": 1189.694,
      "width": 22.552,
      "height": 14.462,
      "confidence": 0.834,
      "class": "spot",
      "class_id": 3,
      "detection_id": "8b05e0f6-6b1e-4023-ac6d-0ffc9f100934"
    },
    {
      "x": 377.679,
      "y": 1026.837,
      "width": 17.7,
      "height": 16.786,
      "confidence": 0.834,
      "class": "spot",
      "class_id": 3,
      "detection_id": "2449aef1-29a3-4e5f-a145-faad800748ac"
    },
    {
      "x": 362.914,
      "y": 987.505,
      "width": 18.557,
      "height": 13.857,
      "confidence": 0.833,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "4dc59757-3e00-433c-b185-6ff1f9b295ac"
    },
    {
      "x": 369.996,
      "y": 1080.353,
      "width": 23.779,
      "height": 12.407,
      "confidence": 0.83,
      "class": "spot",
      "class_id": 3,
      "detection_id": "580eb641-d636-4a49-b4b1-8bf7527fac7a"
    },
    {
      "x": 287.193,
      "y": 470.724,
      "width": 16.522,
      "height": 18.43,
      "confidence": 0.83,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "2692adbb-8ea2-4a53-8888-483ee3dbe32c"
    },
    {
      "x": 416.666,
      "y": 1030.324,
      "width": 19.821,
      "height": 17.694,
      "confidence": 0.829,
      "class": "spot",
      "class_id": 3,
      "detection_id": "c506d774-d5de-481b-a6aa-83e3e6e88e9d"
    },
    {
      "x": 198.193,
      "y": 894.562,
      "width": 17.718,
      "height": 14.416,
      "confidence": 0.829,
      "class": "spot",
      "class_id": 3,
      "detection_id": "ea2c5e46-190f-40ce-897a-f0b0a06503f2"
    },
    {
      "x": 300.292,
      "y": 1020.885,
      "width": 22.55,
      "height": 13.348,
      "confidence": 0.829,
      "class": "spot",
      "class_id": 3,
      "detection_id": "b0f334e3-60ad-45bd-9140-ad31bd71a82e"
    },
    {
      "x": 441.329,
      "y": 418.62,
      "width": 13.313,
      "height": 22.195,
      "confidence": 0.829,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "44feb1b9-1726-4c8d-a564-4a7f9d9eca04"
    },
    {
      "x": 422.205,
      "y": 1093.287,
      "width": 19.339,
      "height": 11.713,
      "confidence": 0.828,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "56575fb9-5dbc-4d81-b38a-9ed45cc4b9e1"
    },
    {
      "x": 261.52,
      "y": 1019.005,
      "width": 21.17,
      "height": 13.977,
      "confidence": 0.827,
      "class": "spot",
      "class_id": 3,
      "detection_id": "80f090fa-d38b-4b94-961b-428888753cd4"
    },
    {
      "x": 384.316,
      "y": 673.658,
      "width": 15.553,
      "height": 17.398,
      "confidence": 0.826,
      "class": "spot",
      "class_id": 3,
      "detection_id": "fbe5b6c7-e91e-48c6-93a7-09ee6873252d"
    },
    {
      "x": 298.837,
      "y": 907.257,
      "width": 21.404,
      "height": 14.308,
      "confidence": 0.826,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "8b1531c9-b728-4b52-9947-7354ced4d31a"
    },
    {
      "x": 343.169,
      "y": 1023.146,
      "width": 18.605,
      "height": 14.743,
      "confidence": 0.823,
      "class": "spot",
      "class_id": 3,
      "detection_id": "07be658e-90e8-49b3-a462-9635792aa9c3"
    },
    {
      "x": 589.811,
      "y": 73.314,
      "width": 18.029,
      "height": 16.738,
      "confidence": 0.823,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "44a8e0d8-9206-43b6-af38-682380cf720d"
    },
    {
      "x": 336.236,
      "y": 1078.727,
      "width": 22.84,
      "height": 12.222,
      "confidence": 0.822,
      "class": "spot",
      "class_id": 3,
      "detection_id": "85c71021-5ec6-447e-858c-9299fb94e5ac"
    },
    {
      "x": 191.866,
      "y": 850.169,
      "width": 20.484,
      "height": 21.541,
      "confidence": 0.822,
      "class": "spot",
      "class_id": 3,
      "detection_id": "f40b31b1-30a3-46c2-8a5d-1a66d18464c5"
    },
    {
      "x": 451.655,
      "y": 104.229,
      "width": 19.864,
      "height": 21.944,
      "confidence": 0.819,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "66388875-d51f-434d-85c4-a19fa6f06aaf"
    },
    {
      "x": 193.645,
      "y": 982.308,
      "width": 16.82,
      "height": 16.777,
      "confidence": 0.816,
      "class": "spot",
      "class_id": 3,
      "detection_id": "e52982ba-aa86-4f88-b216-e20eb8158cc5"
    },
    {
      "x": 399.192,
      "y": 386.497,
      "width": 16.871,
      "height": 14.898,
      "confidence": 0.811,
      "class": "spot",
      "class_id": 3,
      "detection_id": "598d7935-5d33-49dc-b53c-327116f2baad"
    },
    {
      "x": 401.818,
      "y": 354.959,
      "width": 18.297,
      "height": 15.081,
      "confidence": 0.811,
      "class": "spot",
      "class_id": 3,
      "detection_id": "898df6f6-e1fb-4d18-986b-34e0579f3d2c"
    },
    {
      "x": 247.422,
      "y": 325.807,
      "width": 26.413,
      "height": 19.955,
      "confidence": 0.811,
      "class": "spot",
      "class_id": 3,
      "detection_id": "aff20d35-9f22-4d66-b05c-2aa2f7dbc4b3"
    },
    {
      "x": 294.456,
      "y": 1065.697,
      "width": 24.688,
      "height": 10.759,
      "confidence": 0.81,
      "class": "spot",
      "class_id": 3,
      "detection_id": "a89387a2-fbcb-43b2-ace7-4ccd5fbb46ed"
    },
    {
      "x": 285.545,
      "y": 258.256,
      "width": 23.729,
      "height": 19.446,
      "confidence": 0.809,
      "class": "spot",
      "class_id": 3,
      "detection_id": "ea2f3ddf-ff27-454a-9347-cd9849d32208"
    },
    {
      "x": 301.312,
      "y": 754.065,
      "width": 17.117,
      "height": 13.101,
      "confidence": 0.807,
      "class": "Screw",
      "class_id": 2,
      "detection_id": "c4f4da73-15c5-45ae-af6f-c7e73ad04926"
    },
    {
      "x": 186.989,
      "y": 1189.272,
      "width": 19.902,
      "height": 13.779,
      "confidence": 0.806,
      "class": "spot",
      "class_id": 3,
      "detection_id": "1e558ff9-e671-4fea-be2a-f2aff5abfc62"
    },
    {
      "x": 217.446,
      "y": 1023.258,
      "width": 16.41,
      "height": 15.198,
      "confidence": 0.805,
      "class": "spot",
      "class_id": 3,
      "detection_id": "8d19ba90-2c38-4e2b-916d-e7856ba8ac8f"
    },
    {
      "x": 196.757,
      "y": 1120.459,
      "width": 20.069,
      "height": 16.508,
      "confidence": 0.8,
      "class": "spot",
      "class_id": 3,
      "detection_id": "8bc60d73-57ef-4802-b57b-88e9645dfb12"
    },
    {
      "x": 544.325,
      "y": 96.516,
      "width": 21.821,
      "height": 20.981,
      "confidence": 0.8,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "9e7d81f3-69d2-4aaf-9c77-910551b78d47"
    },
    {
      "x": 185.689,
      "y": 800.984,
      "width": 18.667,
      "height": 18.997,
      "confidence": 0.798,
      "class": "spot",
      "class_id": 3,
      "detection_id": "f2146054-45d0-42ce-8080-8a70612c3e56"
    },
    {
      "x": 331.744,
      "y": 505.304,
      "width": 8.985,
      "height": 16.321,
      "confidence": 0.797,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "93a705ab-fa0d-43a3-b015-3915bf908ab3"
    },
    {
      "x": 194.904,
      "y": 1023.619,
      "width": 16.737,
      "height": 15.013,
      "confidence": 0.795,
      "class": "spot",
      "class_id": 3,
      "detection_id": "56193add-4919-41b3-8159-456c712a81fb"
    },
    {
      "x": 576.497,
      "y": 39.219,
      "width": 17.206,
      "height": 16.876,
      "confidence": 0.79,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "6515398c-1d84-4c20-8158-1067bf4f2f39"
    },
    {
      "x": 346.943,
      "y": 1037.7,
      "width": 28.245,
      "height": 10.911,
      "confidence": 0.777,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "2776c633-47c4-4580-bc06-e617a656850a"
    },
    {
      "x": 299.264,
      "y": 1033.117,
      "width": 26.574,
      "height": 9.212,
      "confidence": 0.763,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "59d43ded-ed01-4909-8367-a6d88f8fbd1f"
    },
    {
      "x": 390.636,
      "y": 411.329,
      "width": 14.547,
      "height": 13.569,
      "confidence": 0.76,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "1924483d-97f3-48ad-b8a1-940fee6b00bd"
    },
    {
      "x": 195.742,
      "y": 1092.614,
      "width": 19.963,
      "height": 13.97,
      "confidence": 0.755,
      "class": "spot",
      "class_id": 3,
      "detection_id": "4bd252e2-be32-4f50-8221-c55b082fd0b1"
    },
    {
      "x": 224.373,
      "y": 349.019,
      "width": 20.042,
      "height": 19.112,
      "confidence": 0.748,
      "class": "spot",
      "class_id": 3,
      "detection_id": "2fb9fdec-fcbe-4386-b73b-cf91c20f551f"
    },
    {
      "x": 394.958,
      "y": 371.993,
      "width": 11.353,
      "height": 10.955,
      "confidence": 0.744,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "29d2bf1d-73f9-4598-a045-dd19e195a363"
    },
    {
      "x": 207.202,
      "y": 367.931,
      "width": 21.993,
      "height": 17.93,
      "confidence": 0.737,
      "class": "spot",
      "class_id": 3,
      "detection_id": "116d4e1c-81cb-44e1-b0d1-6503909010ab"
    },
    {
      "x": 176.063,
      "y": 771.219,
      "width": 17.124,
      "height": 21.461,
      "confidence": 0.736,
      "class": "spot",
      "class_id": 3,
      "detection_id": "0ffc1cef-e0c7-4ae0-a03a-df79bb8b7faa"
    },
    {
      "x": 448.471,
      "y": 358.827,
      "width": 12.341,
      "height": 13.96,
      "confidence": 0.733,
      "class": "spot",
      "class_id": 3,
      "detection_id": "a008d4e0-a7ed-4762-bc85-51e00b636d20"
    },
    {
      "x": 195.959,
      "y": 1064.871,
      "width": 17.764,
      "height": 17.47,
      "confidence": 0.73,
      "class": "spot",
      "class_id": 3,
      "detection_id": "6afa240f-4a06-424c-9782-7e3c18048ee4"
    },
    {
      "x": 306,
      "y": 466.201,
      "width": 11.071,
      "height": 8.303,
      "confidence": 0.704,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "c887889c-8c3c-4ba6-b756-a529343deb82"
    },
    {
      "x": 197.709,
      "y": 1145.611,
      "width": 18.492,
      "height": 13.45,
      "confidence": 0.703,
      "class": "spot",
      "class_id": 3,
      "detection_id": "23326d3b-e1d1-4ac8-a2b8-7921f2c3f3ff"
    },
    {
      "x": 309.532,
      "y": 766.808,
      "width": 6.815,
      "height": 9.29,
      "confidence": 0.658,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "0d944fca-b7da-4f0d-b3fa-c4667a6e6491"
    },
    {
      "x": 295.749,
      "y": 1053.147,
      "width": 15.154,
      "height": 11.035,
      "confidence": 0.619,
      "class": "Hole",
      "class_id": 1,
      "detection_id": "a87c91d0-c658-4592-b57b-aa0667b64ed3"
    }
  ]
}

# -------------------------------
# Configuration: reference image filename (put this file in repo root)
# -------------------------------
REF_IMAGE_PATH = "good_ref.jpg"  # <-- add your GOOD reference image to repo with this name

# -------------------------------
# Utilities
# -------------------------------
def center_to_xyxy_centerformat(box):
    # box has center-format x,y,width,height -> convert to [x0,y0,x1,y1]
    x, y, w, h = float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])
    return [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0]

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

def encode_img_to_base64(img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# Load reference image once at startup
if not os.path.exists(REF_IMAGE_PATH):
    REF_IMG = None
else:
    REF_IMG = cv2.imread(REF_IMAGE_PATH)
    if REF_IMG is None:
        REF_IMG = None

# -------------------------------
# /annotate endpoint
# -------------------------------
@app.post("/annotate")
async def annotate(
    file: UploadFile = File(...),
    defective_json: str = Form(...),
    iou_threshold: float = Form(0.5)
):
    """
    - file: defective image (binary)
    - defective_json: JSON string { "predictions": [...] } (Roboflow output for defective image)
    - iou_threshold: float for deciding presence (default 0.5)
    Returns:
      { "annotated": "<base64>", "stats": {...}, "debug": {...} }
    """
    # read incoming image
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    defect_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if defect_img is None:
        return JSONResponse(status_code=400, content={"error":"Could not decode uploaded image."})

    # parse defective_json
    try:
        defective_obj = json.loads(defective_json) if isinstance(defective_json, str) else defective_json
        defective_preds = defective_obj.get("predictions", [])
    except Exception as e:
        return JSONResponse(status_code=400, content={"error":"defective_json parse failed", "detail": str(e)})

    # prepare GOOD preds (center->xyxy in reference image coords)
    good_preds = GOOD_JSON.get("predictions", [])
    good_xyxy = []
    for g in good_preds:
        box = center_to_xyxy_centerformat(g)
        good_xyxy.append({"class": str(g.get("class")), "xyxy": box, "raw": g})

    # Attempt alignment if we have a reference image
    method_used = "none"
    matches_count = 0
    transformed_defective_preds = []  # will be list of dicts with class and xyxy in ref coords

    if REF_IMG is not None:
        # 1. compute ORB features and match
        try:
            ref_gray = cv2.cvtColor(REF_IMG, cv2.COLOR_BGR2GRAY)
            def_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(2000)
            kp1, des1 = orb.detectAndCompute(ref_gray, None)
            kp2, des2 = orb.detectAndCompute(def_gray, None)

            if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
                bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
                knn = bf.knnMatch(des1, des2, k=2)
                good_matches = []
                for m,n in knn:
                    # Lowe ratio test
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                matches_count = len(good_matches)

                if matches_count >= 6:
                    # build source/target pts for homography
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # map defective -> reference
                    if H is not None:
                        method_used = "homography"
                        # transform each defective bounding box (four corners) into reference coordinate space
                        for d in defective_preds:
                            try:
                                cx, cy, w, h = float(d["x"]), float(d["y"]), float(d["width"]), float(d["height"])
                                x0 = cx - w/2.0
                                y0 = cy - h/2.0
                                x1 = cx + w/2.0
                                y1 = cy + h/2.0
                                corners = np.array([ [x0,y0],[x1,y0],[x1,y1],[x0,y1] ], dtype=np.float32).reshape(-1,1,2)
                                warped = cv2.perspectiveTransform(corners, H)  # map to ref coords
                                xs = warped[:,0,0]
                                ys = warped[:,0,1]
                                tx0, ty0, tx1, ty1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                                transformed_defective_preds.append({
                                    "class": str(d.get("class")),
                                    "xyxy": [tx0, ty0, tx1, ty1],
                                    "raw": d
                                })
                            except Exception:
                                continue
                else:
                    method_used = "not_enough_matches"
            else:
                method_used = "no_descriptors"
        except Exception as e:
            method_used = "orb_error"
    else:
        method_used = "no_ref_image"

    # If homography failed, fallback to simple scale (map coordinates by image sizes)
    if method_used not in ("homography",):
        # fallback scale
        ref_h, ref_w = REF_IMG.shape[:2] if REF_IMG is not None else (None, None)
        def_h, def_w = defect_img.shape[:2]
        if (ref_w is not None) and (def_w is not None) and def_w>0 and def_h>0:
            sx = ref_w / float(def_w)
            sy = ref_h / float(def_h)
            method_used = "scale_fallback"
            for d in defective_preds:
                cx, cy, w, h = float(d["x"]), float(d["y"]), float(d["width"]), float(d["height"])
                x0 = cx - w/2.0
                y0 = cy - h/2.0
                x1 = cx + w/2.0
                y1 = cy + h/2.0
                tx0, ty0, tx1, ty1 = x0*sx, y0*sy, x1*sx, y1*sy
                transformed_defective_preds.append({
                    "class": str(d.get("class")),
                    "xyxy": [tx0, ty0, tx1, ty1],
                    "raw": d
                })
        else:
            method_used = "no_mapping_possible"

    # Now compare GOOD boxes vs transformed defective boxes (only red boxes drawn for missing)
    missing_labels = []
    annotated_img = REF_IMG.copy() if REF_IMG is not None else defect_img.copy()  # show boxes on reference if available; else defective
    total_expected = len(good_xyxy)
    for g in good_xyxy:
        gclass = g["class"]
        gbox = g["xyxy"]  # [x0,y0,x1,y1] in reference coords
        matched = False
        # find any defective prediction of same class with IOU >= threshold
        for td in transformed_defective_preds:
            if td["class"] == gclass:
                i = iou_xyxy(gbox, td["xyxy"])
                if i >= float(iou_threshold):
                    matched = True
                    break
        if not matched:
            # draw red box and label at GOOD location (convert floats to ints)
            x0, y0, x1, y1 = int(round(gbox[0])), int(round(gbox[1])), int(round(gbox[2])), int(round(gbox[3]))
            # clamp
            h, w = annotated_img.shape[:2]
            x0 = max(0, min(w-1, x0)); x1 = max(0, min(w-1, x1))
            y0 = max(0, min(h-1, y0)); y1 = max(0, min(h-1, y1))
            cv2.rectangle(annotated_img, (x0, y0), (x1, y1), (0,0,255), 3)
            label = str(gclass)
            ty = y0 - 8 if y0-8>8 else y0 + 12
            font_scale = max(0.4, min(1.0, min(w,h)/1000))
            cv2.putText(annotated_img, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), 2, cv2.LINE_AA)
            missing_labels.append(label)

    # prepare response
    try:
        annotated_b64 = encode_img_to_base64(annotated_img)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":"encode_failed", "detail": str(e)})

    stats = {
        "total_expected": total_expected,
        "missing": len(missing_labels),
        "iou_threshold": float(iou_threshold)
    }
    debug = {
        "method_used": method_used,
        "matches_count": matches_count,
        "transformed_def_count": len(transformed_defective_preds)
    }

    ok, buf = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
if not ok:
    return JSONResponse(status_code=500, content={"error": "encode_failed"})

return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

    })
