import os
from uuid import uuid4
from models.user_model import User, db
from models.image_model import Image
from flask import jsonify

REVERSE_IMAGE_IMGS = "./uploaded_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(REVERSE_IMAGE_IMGS, exist_ok=True)

def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def add_image(user_id, file):
    if not file or not file.filename:
        return jsonify({"error": "no file given"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file type"}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "error with user please login again"}), 400

    if user.num_images >= 10:
        return jsonify({"error": "10 images max at a time"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid4().hex}.{ext}"
    path = os.path.join(REVERSE_IMAGE_IMGS, filename)

    file.save(path)

    image = Image(user_id=user.id, file_path=filename)
    db.session.add(image)
    user.num_images += 1

    db.session.commit()

    return jsonify({"message": "image added"}), 201




def display_images(user_id):
    user = User.query.get(user_id)

    if not user:
        return jsonify({"error": "User not found"}), 400

    images = Image.query.filter_by(user_id = user_id).all()

    results = []
    for img in images:
        results.append({"id":img.id, "path": f"/uploaded_images/{img.file_path}"})

    return jsonify({"images": results}),200



def delete_image(image_id):
    image = Image.query.get(image_id)

    if not image:
        return jsonify({"error": "image not found"}), 404

    if image.user_id != int(user_id):
        return jsonify({"error":"unathorized"}), 403

    user = User.query.get(image.user_id)

    file_path = os.path.join(REVERSE_IMAGE_IMGS, image.file_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    db.session.delete(image)

    if user and user.num_images > 0:
        user.num_images -= 1

    db.session.commit()
    return jsonify({"message": "image deleted"}), 200





# REVERSE_IMAGE_IMGS = ./path

# def add_image():
#     get the user id from jwt
#     check how images the user has 
#     add image to folder with id to db

# REVERSE_IMAGE_IMGS = ./path

# file random string for secuirty 

# add file id to db

# def display_images()
#     get user id from jwt

#     get images with id from db 

#     make list {} for paths and send it to the frontend








