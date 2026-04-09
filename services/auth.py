
from flask import jsonify, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from models.user_model import db, User

bcrypt = Bcrypt()



def register(data):
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
    if len(password) < 8 or len(password) > 40:
        return jsonify({"message":"Password must be at least 8"}),400
    if not any(c.isupper() for c in password):
        return jsonify({"message":"password must have at least one uppercase letter."}),400
    if not any(c.islower() for c in password):
        return jsonify({"message":"password must have at least one lowercase letter."}),400

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message':'User registered'}), 201




def login(data):
    
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message":"Please fill in all fields"}),400

    user = User.query.filter_by(email=email).first()

    if user and bcrypt.check_password_hash(user.password, password):
        token = create_access_token(identity=str(user.id))
        return jsonify({'token': token}), 200

    return jsonify({'message': 'Invalid credentials'}), 401





def change_user_info(data):
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    if not user:
        return jsonify({"message":"user cant be found"}),404


    new_email = data.get('email')
    new_password = data.get('new_password')
    current_password = data.get("current_password")
 
    if (new_email or new_password) and not current_password:
        return jsonify({"message":"please enter current password"}),400
    
    if current_password:
        if not bcrypt.check_password_hash(user.password,current_password):
            return jsonify({"message":"incorrect password"}),401

    if new_password:
        if len(new_password) < 8 or len(new_password) > 40:
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





    