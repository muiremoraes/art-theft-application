
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from models.user_model import db, User
from services.auth import register, login, change_user_info
from services.file import serve_image, upload_image, download_image
from services.steganography import encode_endpoint, decode_endpoint
from services.visible_wm import visible_watermark_endpoint
from services.dct import dct_encode_endpoint, dct_decode_endpoint
import os
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




@app.route('/register', methods=['POST'])
def register_route():
    return register(request.get_json())



@app.route('/login', methods=['POST'])
def login_route():
    return login(request.get_json())



@app.route('/change_user_info', methods=['PUT'])
@jwt_required()
def change_user_info_route():
    return change_user_info(request.get_json())



@app.route('/encode_steganography', methods=['POST'])
@jwt_required()
def encode_steganography_route():
    return encode_endpoint(request.get_json())


@app.route('/steganography_decode', methods=['POST'])
@jwt_required()
def decode_steganography_route():
    return decode_endpoint(request.get_json())



@app.route('/visible_encode',methods=['POST'])
@jwt_required()
def visible_watermark_route():
    return visible_watermark_endpoint(request.get_json())


@app.route("/download_image", methods=["POST"])
def download_image_route():
    return download_image()


@app.route('/dct_encode',methods=['POST'])
@jwt_required()
def dct_encode_route():
    return dct_encode_endpoint(request.get_json())




@app.route('/dct_decode',methods=['POST'])
@jwt_required()
def dct_decode_route():
    return dct_decode_endpoint(request.get_json())




#upload image route
@app.route("/images/<filename>")
def serve_image_route(filename):
    return serve_image(filename)

@app.route("/upload-image", methods=["POST"]) #check file exists
def upload_image_route():
    return upload_image()



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run(debug=True)

