
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from models.user_model import db, User
from models.image_model import Image
from services.auth import register, login, change_user_info
from services.file import serve_image, upload_image, download_image
from services.steganography import encode_endpoint, decode_endpoint
from services.visible_wm import visible_watermark_endpoint
from services.dct import dct_encode_endpoint, dct_decode_endpoint
from services.compare import upload_to_compare, compare_all, serve_compare_result
from services.image_for_scan import add_image, display_images, delete_image
from services.reverse_search import start_scan, get_results
from services.scheduler import start_scheduler
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

# #upload image route
# @app.route("/uploaded_images/<filename>")
# def serve_uploaded_images_route(filename):
#     return send_from_directory(REVERSE_IMAGE_IMGS, filename)

@app.route("/upload-image", methods=["POST"]) #check file exists
@jwt_required()
def upload_image_route():
    return upload_image()
    


@app.route("/upload-to_compare", methods=["POST"])
@jwt_required()
def upload_to_compare_route():
    return upload_to_compare()

@app.route("/compare", methods=["POST"])
@jwt_required()
def compare_route():
    return compare_all()

@app.route("/compare_results/<filename>")
def serve_compare_result_route(filename):
    return serve_compare_result(filename)


@app.route("/upload", methods=["POST"])
@jwt_required()
def add_image_route():
    user_id = get_jwt_identity()
    file = request.files.get("file")
    return add_image(user_id, file)

@app.route("/user-images", methods=["GET"])
@jwt_required()
def display_images_route():
    user_id = get_jwt_identity()
    return display_images(user_id)

@app.route("/delete/<int:image_id>", methods=["DELETE"])
@jwt_required()
def delete_image_route(image_id):
    user_id = get_jwt_identity()
    return delete_image(user_id, image_id)



@app.route("/scan", methods=["POST"])
@jwt_required()
def scan():
    user_id = get_jwt_identity()
    return start_scan(user_id)


@app.route("/scan/results", methods=["GET"])
@jwt_required()
def scan_results():
    user_id = get_jwt_identity()
    return get_results(user_id)








if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    start_scheduler(app)
    app.run(debug=True)


