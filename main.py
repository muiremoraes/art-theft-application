
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from models.user_model import db, User
import cv2
import numpy as np

app = Flask(__name__)

app.config['SECRET_KEY'] = 'strong_secret_key' # TODO: dont have secret displayed like this 
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)



@app.route('/get_user', methods=['GET'])
@jwt_required()
def get_user():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=user_id).first()

    # if not username or not email or not password:
    #     return jsonify({'message': 'Missing required fields'}), 400

    if user:
        return jsonify({'message': 'Sucess', 'name': user.username}), 200
    else:
        return jsonify({'message': 'Failed'}), 404



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
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"message":"Please fill in all 3 fields"}),400

    user = User.query.filter_by(username=username, email=email).first()

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

@app.route('/encode', methods=['POST'])´
@jwt_required()
def encode_endpoint():
    data = request.get_json()
    watermark = data.get('watermark')
    img_name=data.get('name')
    img_path="./images/"+img_name
    image = cv2.imread(img_path)
    encoded_image=encode(image,watermark)

    output_path = "./lsb_watermark_img/"+img_name
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

@app.route('/decode', methods=['POST'])
@jwt_required()
def decode_endpoint():
    data = request.get_json()
    img_name=data.get('name')
    img_path="./lsb_watermark_img/"+img_name
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
            reading = False # stop if null value \0
        else:
            message += chr(ascii_num) # add ascii num to msg
            i += 7 # move to next 7 pixels 
            if i >= len(flat_img): # stop if end of image
                break
    return message


    






if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run(debug=True)