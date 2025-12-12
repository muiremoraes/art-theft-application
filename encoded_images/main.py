
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from models.user_model import db, User
import cv2
import numpy as np
import os
import secrets
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/*": {"origins":["http://localhost:5173"]}}) #TODO:set website

# source ./venv/bin/activate
app.config['SECRET_KEY'] = 'strong_secret_key' # TODO: dont have secret displayed like this 
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key' # TODO: dont have secret displayed like this 
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)



# @app.route('/get_user', methods=['GET'])
# @jwt_required()
# def get_user():
#     user_id = get_jwt_identity()
#     user = User.query.filter_by(id=user_id).first()

#     # if not username or not email or not password:
#     #     return jsonify({'message': 'Missing required fields'}), 400

#     if user:
#         return jsonify({'message': 'Sucess', 'name': user.username}), 200
#     else:
#         return jsonify({'message': 'Failed'}), 404



@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'Missing required fields'}), 400

    if "@" not in email:
        return jsonify({"message":"please enter a valid email"}),400
    
    if User.query.filter_by(email=email).first():
        return jsonify({"message":"one email per account"}),400

    # password check
    if len(password) < 8 or len(password) > 12:
        return jsonify({"message":"Password must be between 8 and 12 characters."}),400
    if not any(c.isupper() for c in password):
        return jsonify({"message":"password must have at least one uppercase letter."}),400
    if not any(c.islower() for c in password):
        return jsonify({"message":"password must have at least one lowercase letter."}),400

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message':'User registered'}), 201



@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message":"Please fill in all fields"}),400

    user = User.query.filter_by(email=email).first()

    if user and bcrypt.check_password_hash(user.password, password):
        token = create_access_token(identity=str(user.id))
        return jsonify({'token': token}), 200

    return jsonify({'message': 'Invalid credentials'}), 401




@app.route('/change_user_info', methods=['PUT'])
@jwt_required()
def change_user_info():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    if not user:
        return jsonify({"message":"user cant be found"}),404

    data = request.get_json()

    new_email = data.get('email')
    new_password = data.get('new_password')
    current_password = data.get("current_password")
 
    if (new_email or new_password) and not current_password:
        return jsonify({"message":"please enter current password"}),400
    
    if current_password:
        if not bcrypt.check_password_hash(user.password,current_password):
            return jsonify({"message":"incorrect password"}),401

    if new_password:
        if len(new_password) < 8 or len(new_password) > 12:
            return jsonify({"message":"Password must be between 8 and 12 characters."}),400
        if not any(c.isupper() for c in new_password):
            return jsonify({"message":"password must have at least one uppercase letter."}),400
        if not any(c.islower() for c in new_password):
            return jsonify({"message":"password must have at least one lowercase letter."}),400

        new_pass = bcrypt.generate_password_hash(new_password).decode('utf-8')
        user.password = new_pass

    if new_email:
        if "@" not in new_email:
            return jsonify({"message":"please use a valid email"}),400
        exists = User.query.filter(User.email == new_email, User.id != user_id).first()
        if exists:
            return jsonify({"message":"one email per account"}),400
        user.email = new_email
    
    db.session.commit()

    return jsonify({"message":"updated user info"}),200





    




#img_path = r"c:/home/marym/art_detector/images/image1.png"

#watermark = "digital design blah"


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

@app.route('/encode_steganography', methods=['POST'])
@jwt_required()
def encode_endpoint():
    data = request.get_json()
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


@app.route('/steganography_decode', methods=['POST'])
@jwt_required()
def decode_endpoint():
    data = request.get_json()
    img_name=secure_filename(data.get('name'))

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



@app.route('/visible_encode',methods=['POST'])
@jwt_required()
def visible_watermark_endpoint():
    data = request.get_json()
    watermark = data.get('watermark')
    img_name = secure_filename(data.get('name'))

    if not watermark or not img_name:
        return jsonify({"message":"fill al feilds"}),400


    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400


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
    
ENCODED_DIR = "./encoded_images"

@app.route("/download_image", methods=["POST"])
def download_image():
    data = request.get_json()

    if not data or "name" not in data:
        return jsonify({"error": "Missing 'name'"}), 400

    filename = data["name"]
    file_path = os.path.join(ENCODED_DIR, filename)

    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    # send_from_directory handles headers and streaming safely
    return send_from_directory(
        ENCODED_DIR,
        filename,
        as_attachment=True
    )

@app.route('/dct_encode',methods=['POST'])
@jwt_required()
def dct_encode_endpoint():
    data = request.get_json()
    watermark = data.get('watermark')
    img_name = secure_filename(data.get('name'))

    if not watermark or not img_name:
        return jsonify({"message":"fill al feilds"}),400


    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400


    img_path="./images/"+img_name
    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    # check divisible by 8
    h, w = image.shape
    image = image[:h - h % 8, :w - w % 8] # remove pixels remaining

    image_with_wm = encode_watermark_into_image(image.copy(), watermark)

    output_path = "./encoded_images/"+img_name
    cv2.imwrite(output_path,image_with_wm)

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






@app.route('/dct_decode',methods=['POST'])
@jwt_required()
def dct_decode_endpoint():
    data = request.get_json()
    img_name = secure_filename(data.get('name'))

    if not img_name:
        return jsonify({"message":"fill all feilds"}),400


    allowed={'png','jpg','jpeg'}
    if img_name.split(".")[-1].lower() not in allowed:
        return jsonify({"message":"invlaid file"}),400


    img_path="./encoded_images/"+img_name
    if not os.path.exists(img_path):
        return jsonify({"message":"error please try again"}),404

    print(img_path)
    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    print(image)
    # check divisible by 8
    h, w = image.shape
    image = image[:h - h % 8, :w - w % 8] # remove pixels remaining

    decoded_wm = decode_watermark_from_image(image)

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







UPLOAD_FOLDER = "./images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)#create upload folder if not exists

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS #check if file ends in one of the okay types



#upload image route
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename) #gets image from file and display on frontend

@app.route("/upload-image", methods=["POST"]) #check file exists
def upload_image():
    if "file" not in request.files: #check if contain file
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # remove bad char
    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()

    # gens random filename hash
    file_hash = secrets.token_hex(16)
    filename = f"{file_hash}.{ext}"

    # reads file into memory once
    file_bytes = file.read()

    os.makedirs(UPLOAD_FOLDER, exist_ok=True) # check is is written to both folders
    encoded_folder = "./encoded_images"
    os.makedirs(encoded_folder, exist_ok=True)

    with open(os.path.join(UPLOAD_FOLDER, filename), "wb") as f: #save uploaded file to images for displaying
        f.write(file_bytes)

    with open(os.path.join(encoded_folder, filename), "wb") as f: #save second in encoded images
        f.write(file_bytes)

    # URL returned for image box
    file_url = f"/images/{filename}"

    return jsonify({
        "hash": file_hash, #frontend knows filename
        "url": file_url #frontend can show image
    }), 201




if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run(debug=True)