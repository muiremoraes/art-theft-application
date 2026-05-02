
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from models.user_model import db, User
from models.image_model import Image
from services.auth import register, login, change_user_info, check_otp
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
from flask_mail import Mail, Message
from api_config import GMAIL_ID, GMAIL_PASSWORD, SECRET_KEY, JWT_SECRET_KEY
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import timedelta
from flask_talisman import Talisman
from services.fgsm import  fgsm_endpoint

app = Flask(__name__)
CORS(app)

Talisman(
    app,
    force_https=False,
    content_security_policy=None
)

# source ./venv/bin/activate
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES']=timedelta(minutes=30)
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)


app.config["MAIL_SERVER"]='smtp.gmail.com'
app.config["MAIL_PORT"]=587
app.config["MAIL_USERNAME"]=GMAIL_ID
app.config["MAIL_PASSWORD"]=GMAIL_PASSWORD
app.config["MAIL_USE_TLS"]=True
app.config["MAIL_USE_SSL"]=False
app.config["MAIL_DEFAULT_SENDER"]=GMAIL_ID

mail = Mail(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
)

@app.route('/register', methods=['POST'])
@limiter.limit("5 per minute")
@limiter.limit("10 per hour")
def register_route():
    return register(request.get_json())
 

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
@limiter.limit("10 per hour")
def login_route():
    return login(request.get_json())

@app.route('/check_otp', methods=['POST'])
@limiter.limit("2 per minute")
def check_otp_route():
    return check_otp(request.get_json())



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
@limiter.limit("10 per minute")
@jwt_required()
def compare_route():
    return compare_all()

@app.route("/compare_results/<filename>")
def serve_compare_result_route(filename):
    return serve_compare_result(filename)


@app.route("/upload", methods=["POST"])
@limiter.limit("6 per minute")
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
@limiter.limit("4 per minute")
@jwt_required()
def scan():
    user_id = get_jwt_identity()
    return start_scan(user_id)


@app.route("/scan/results", methods=["GET"])
@jwt_required()
def scan_results():
    user_id = get_jwt_identity()
    return get_results(user_id)


def is_admin_user():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    return user and user.is_admin


@app.route("/admin/users", methods=["GET"])
@jwt_required()
def admin_view_users_route():
    if not is_admin_user():
        return jsonify({"message":"admin access required"}),403
    users = User.query.all()
    results = []
    for u in users:
        results.append({
        "id":u.id,
        "username":u.username,
        "email":u.email,
        "password_hash":u.password,
        "num_images":u.num_images,
        "is_admin":u.is_admin
        })
    return jsonify(results),200

    
@app.route("/check-perms", methods=["GET"])
@jwt_required()
def check_perms_route():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    return jsonify({"is_admin":user.is_admin}),200

@app.route("/add-fgsm", methods=["POST"])
@jwt_required()
@limiter.limit("5 per minute")
def fgsm_route():
    return fgsm_endpoint(request)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    start_scheduler(app)
    app.run(debug=True)


