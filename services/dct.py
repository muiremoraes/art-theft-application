import cv2
import numpy as np
import os
from flask import jsonify
from werkzeug.utils import secure_filename


    
ENCODED_DIR = "./encoded_images"
IMG_DIR = "./images"




def dct_encode_endpoint(data):
    watermark = data.get('watermark')
    img_name = secure_filename(data.get('name'))

    if not watermark or not img_name:
        return jsonify({"message":"fill all feilds"}),400


    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400


    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    image = cv2.imread(img_path)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    # check divisible by 8
    h, w = y.shape
    y = y[:h - h % 8, :w - w % 8] # remove pixels remaining
    cr = cr[:h - h % 8, :w - w % 8]
    cb = cb[:h - h % 8, :w - w % 8]

    y_with_wm = encode_watermark_into_image(y.copy(), watermark)

    ycrcb_with_wm = cv2.merge([y_with_wm, cr, cb])
    final_img = cv2.cvtColor(ycrcb_with_wm, cv2.COLOR_YCrCb2BGR)

    output_path = os.path.join(ENCODED_DIR, img_name)
    cv2.imwrite(output_path,final_img)

    return jsonify({"message":"dct watermark added"}),201



def encode_into_block(block, arr):
    block = np.float32(block) / 255.0  # convert pixels from (0-255) to a float number 0-1
    block = cv2.dct(block) * 300  #do dct to the block and * 300 a stronger effect

    for i in range(0, len(arr)): #add 4 bits across block in low frequency
        x = int(1 + (i % 2))
        y = int(1 + np.floor(i / 2))

        coeff = block[x, y] #get value at that position

        coeff = coeff / 8 #make it smaller and easier to manage
        coeff = np.floor(coeff)

        if arr[i]:  # check if even
            if coeff % 2 != 0: #bit=1 it should be even
                coeff += 1
        else:  # else odd
            if coeff % 2 == 0: #bit=0 it should be odd
                coeff += 1

        coeff = coeff * 8 # chnage number back to normal range
        block[x, y] = coeff

    block = cv2.idct(block / 300) # inverse dct to bring back to normal img in that block
    block[block > 255] = 255 #check pixels bettween 0-255
    block[block < 0] = 0
    # turn back into a normal image format
    block = np.uint8(block * 255.0)

    return block



def get_4_bits_from_bytearray(byteArray, blockIndex):
    byteIndex = blockIndex // 2 #every bytes is 2 blocks

    if byteIndex >= len(byteArray): #if block index is bigger than message repeat message for watermark all over image
        byteIndex = byteIndex % len(byteArray)

    byte = byteArray[byteIndex] #get the bytes for the block

    if blockIndex % 2 == 0: #checks and takes high bits
        offset = 4
    else:
        offset = 0 #checks and takes low bits
  
    bits = []
    for i in range(3, -1, -1):# get 4 bits msb to lsb
        #check if each bit is 1  or 0 convert to True/False
        bits.append(bool((byte >> (offset + i)) & 1)) # convert to boolean

    return bits



def encode_watermark_into_image(image, wm):
    b = wm.encode('ascii') #convert to ascii bytes
    blockIndex = 0 #start block

    #loop through image block by block
    for start_h in range(0, image.shape[0], 8): #get height/width by 8
        for start_w in range(0, image.shape[1], 8):

            end_h = start_h + 8
            end_w = start_w + 8
            # block is start height to end height and start width to end width
            block = image[start_h:end_h, start_w:end_w]
            # change the block
            block = encode_into_block(block, get_4_bits_from_bytearray(b, blockIndex)) #enocde block with next 4 bits
            #save the block back into the image
            image[start_h:end_h, start_w:end_w] = block

            blockIndex += 1

    return image #return watermarked image




####



def dct_decode_endpoint(data):
    img_name = secure_filename(data.get('name'))

    if not img_name:
        return jsonify({"message":"fill all feilds"}),400


    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400


    img_path=os.path.join(ENCODED_DIR, img_name)
    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    #print(img_path)
    image = cv2.imread(img_path)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)


    #print(image)
    # check divisible by 8
    h, w = y.shape
    y = y[:h - h % 8, :w - w % 8] # remove pixels remaining

    decoded_wm = decode_watermark_from_image(y)

    return jsonify({"decoded_watermark" : decoded_wm}), 200





def decode_from_block(block):
    block = np.float32(block) / 255.0 #convert pixels to float between 0-1
    block = cv2.dct(block) * 300 # value big enough for even/odd changes

    returnList = []

    for i in range(0, 4): #store 4 bits in 4 low frequency DCT positions
        x = int(1 + (i % 2)) #gets 1 or 2
        y = int(1 + np.floor(i / 2))

        coeff = block[x, y] #change DCT coeff
        coeff = coeff / 8 #undo encoding scaling
        coeff = np.round(coeff) #rounding because floor in encode

        if coeff % 2 == 0: #even bit was 1 and odd bit was 0
            returnList.append(True)
        else:
            returnList.append(False)

    return returnList


def bools_to_bytearray(bits):
    out = bytearray()

    for i in range(0, len(bits), 8): #loop through 8 bits
        group = bits[i:i+8]
        if len(group) < 8: #if no more bits break
            break

        value = 0
        for b in group: #make byte from bits
            value = (value << 1) | (1 if b else 0)

        out.append(value)

    return out


def decode_watermark_from_image(image):
    all_bits = []

    for start_h in range(0, image.shape[0], 8): # loop through image 8*8
        for start_w in range(0, image.shape[1], 8):

            block = image[start_h:start_h + 8,start_w:start_w + 8] # get 8*8 block

            arr = decode_from_block(block) #decode 4 bits from block
            all_bits.extend(arr) # add bits to list

    data = bools_to_bytearray(all_bits) # convert bite back to characters at end

    text = data.decode('ascii', errors='ignore')#decode into text

    if len(text) > 30: #get first 30 char
        text = text[:30]

    return text



