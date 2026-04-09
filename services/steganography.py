import cv2
import numpy as np
import os
from flask import jsonify
from werkzeug.utils import secure_filename

def get_bin_char(character):
    bits = [int(x) for x in bin(ord(character))[2:]] # convert to ascii, binary, int and get rid of Ob at start
    bits = [0]*(7-len(bits))+bits # to make sure it equals a length of 7 add zero to front / 7 minus length + x array
    return bits

def get_bin_str(s):
    wm = [] # array to store watermark bits in int
    for character in s: # loop through string call fucntion on each char
        wm += get_bin_char(character) # add to string 
    wm += [0]*7 # add null at end \0 
    return wm



def encode_endpoint(data):
    watermark = data.get('watermark')
    img_name=data.get('name')

    if not watermark or not img_name:
        return jsonify({"message":"fill all feilds"}),400

    img_name = secure_filename(img_name)

    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400

    if len(watermark) > 100:
        return jsonify({"message":"max 100 char"}),400

    img_path="./images/"+img_name

    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    image = cv2.imread(img_path)
    encoded_image=encode(image,watermark)

    output_path = "./encoded_images/"+img_name
    cv2.imwrite(output_path, encoded_image)
    return jsonify({'message': 'encode image sucessfully'}), 201
     

def encode(image, s): # encode watermark bits into lsb of image 
    bits = get_bin_str(s) # get bits in string
    bits = np.array(bits, dtype=np.uint8) # chnage to np array
    total_pixels = image.size # all pixels
    if len(bits) > total_pixels: # if watermark bits bigger than image
        raise ValueError("message is too long to fit inside image")
    image_rounded = image - (image%2) # round values so even and removes lsb
    image_flat = image_rounded.flatten() # flatten to 1D 
    image_flat[0:len(bits)] += bits # put bits in image
    encoded_img = np.reshape(image_flat, image.shape) # put back into 3d image
    return encoded_img



def decode_endpoint(data):
    img_name=secure_filename(data.get('name'))

    if not img_name:
        return jsonify({"message": "fill all fields"}), 400

    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400

    img_path="./encoded_images/"+img_name

    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    image = cv2.imread(img_path)
    decoded_wm = decode(image)
    return jsonify({"decoded_watermark" : decoded_wm}), 200

def decode(encoded_img):
    weights = np.array([2 ** (6-i) for i in range(7)]) # weights to help convert for 7 bits to char
    flat_img = encoded_img.flatten() # flattend image into 1d
    message = "" 
    i = 0 # start position in the array
    reading = True #reading  util null
    while reading:
        char_bits = flat_img[i:i +7]%2 # extract 7 pixels and get LSB
        ascii_num = np.sum(weights*char_bits) # convert binary to ascii
        if ascii_num == 0: # if equal to the null val
            break # stop if null value \0
        
        message += chr(ascii_num) # add ascii num to msg
        i += 7 # move to next 7 pixels 

        if i >= len(flat_img): # stop if end of image
            break
    return message

