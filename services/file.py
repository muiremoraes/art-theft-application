
from flask import send_from_directory, jsonify,request
import os
import secrets
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "./images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
encoded_folder = "./encoded_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)#create upload folder if not exists

def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename) #gets image from file and display on frontend



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



def download_image():
    data = request.get_json()

    if not data or "name" not in data:
        return jsonify({"error": "Missing 'name'"}), 400

    filename = data["name"]
    file_path = os.path.join(encoded_folder, filename)

    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    # send_from_directory handles headers and streaming safely
    return send_from_directory(
        encoded_folder,
        filename,
        as_attachment=True
    )
