
import cv2
import numpy as np
import os
from flask import jsonify
from werkzeug.utils import secure_filename



def visible_watermark_endpoint(data):
    watermark = data.get('watermark')
    img_name = secure_filename(data.get('name'))

    if not watermark or not img_name:
        return jsonify({"message":"fill all feilds"}),400


    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invalid file type"}),400


    img_path="./images/"+img_name
    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    img = cv2.imread(img_path)

    result = add_watermark(img,watermark)
    output_path = "./encoded_images/"+img_name
    cv2.imwrite(output_path,result)

    return jsonify({"message":"visible watermark added"}),201


def get_pos(image, left=100, bottom=150):
    img_h = image.shape[0]
    img_w = image.shape[1]

    x = left
    y = img_h - bottom
    return(x,y)


def add_watermark(image,watermark):
    if len(watermark) > 20:
        print("max 20 char")
        watermark = watermark[:20]
    
    watermark = watermark.upper()
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1
    thickness = 2
    pos = get_pos(image)
    wm_colour = (200, 200, 200)
    alpha = 0.7
    
    overlay = image.copy()

    cv2.putText(overlay, watermark, pos, font, scale, wm_colour, thickness)
    wm_image = cv2.addWeighted(overlay, alpha, image, 1-alpha,0)
    return wm_image