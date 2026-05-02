import tensorflow as tf
import numpy as np
from flask import jsonify
from werkzeug.utils import secure_filename
import cv2
import base64

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
guesses = tf.keras.applications.mobilenet_v2.decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()
eps=0.1

def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb,(224,224))
    image = tf.cast(resized, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    img = img[None,...]
    return img


def top_prediction(image):
    predictions = pretrained_model(image)
    i, label, confidence = guesses(predictions.numpy(), top=1)[0][0]
    return label, round(float(confidence)*100,2)


def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss,input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad


def add_fgsm(image, eps):
    predictions = pretrained_model(image)
    top_guess = tf.argmax(predictions, axis=1).numpy()[0]
    label = tf.one_hot(top_guess, predictions.shape[-1])
    label = tf.reshape(label, (1, predictions.shape[-1]))
    perturbation = create_adversarial_pattern(image, label)
    adv_image = image + eps * perturbation
    adv_image = tf.clip_by_value(adv_image, -1,1)
    return adv_image, perturbation



def saveable(tensor):
    img = (tensor[0].numpy() * 0.5 + 0.5)*255
    img = img.clip(0,255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def fgsm_endpoint(request):
    if "file" not in request.files:
        return jsonify({"message":"no file uploaded"}),400
    
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "no file uploaded"}), 400
    
    ext = {"png", "jpg", "jpeg"}
    if file.filename.split(".")[-1].lower() not in ext:
        return jsonify({"message":"invalid file type"}),400
    

    file_bytes = file.read()
    list_num = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(list_num, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"message":"could not read image"}), 400
    
    img_t = preprocess(img_bgr)
    og_label,og_conf = top_prediction(img_t)

    adv_image, preturbations = add_fgsm(img_t,eps)
    adv_label,adv_conf = top_prediction(adv_image)
    bgr_img = saveable(adv_image)

    result, encoded_img = cv2.imencode(".png", bgr_img)
    if not result:
        return jsonify({"message":"error encoding"}),500
    
    image_base64 = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

    return jsonify({
        "orignal":{"label":og_label, "confidence":og_conf},
        "after FGSM": {"label":adv_label, "confidence":adv_conf},
        "image_base64": image_base64
    }), 201


