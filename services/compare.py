import cv2 as cv
import matplotlib.pyplot as plt
from flask import request, jsonify, send_from_directory
import os

COMPARE_FOLDER = "./compare_folder"
RESULTS_FOLDER = "./compare_results"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def upload_to_compare():
    first_img = request.args.get("first_img")
    if first_img is None:
        return jsonify({"error": "missing image"}), 400

    first_img = first_img.lower() == "true"

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file type"}), 400

    file_format = file.filename.rsplit(".", 1)[1].lower()

    if first_img:
        number = "1"
    else:
        number = "2"

    os.makedirs(COMPARE_FOLDER, exist_ok=True)

    for f in os.listdir(COMPARE_FOLDER):
        if f.startswith(number + "."):
            os.remove(os.path.join(COMPARE_FOLDER, f))

    filepath = os.path.join(COMPARE_FOLDER, f"{number}.{file_format}")
    file.save(filepath)

    return jsonify({"message": f"saved {number}.{file_format}"}), 201


def _load_images():
    img1_path = None
    img2_path = None

    if os.path.isdir(COMPARE_FOLDER):
        for f in os.listdir(COMPARE_FOLDER):
            if f.startswith("1."):
                img1_path = os.path.join(COMPARE_FOLDER, f)
            if f.startswith("2."):
                img2_path = os.path.join(COMPARE_FOLDER, f)

    if img1_path:
        img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    else:
        img1 = None

    if img2_path:
        img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
    else:
        img2 = None

    return img1, img2


def sift(img1, img2):
    img1 = cv.resize(img1, (1000, 1000))
    img2 = cv.resize(img2, (1000, 1000))

    detector = cv.SIFT_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    a = len(good)
    b = len(kp2)
    if b == 0:
        b = 1
    percent = (a * 100) / b

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.title("sift")
    plt.imshow(img3)
    plt.savefig(os.path.join(RESULTS_FOLDER, "sift.png"))
    plt.close()

    return {"score": round(percent, 2), "match": percent >= 40.00, "image_url": "/compare_results/sift.png"}


def sift_blur(img1, img2):
    img1 = cv.resize(img1, (200, 200))
    img2 = cv.resize(img2, (200, 200))
    img1 = cv.blur(img1, (10, 10))
    img2 = cv.blur(img2, (10, 10))

    detector = cv.SIFT_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good.append([m])

    a = len(good)
    b = len(kp2)
    if b == 0:
        b = 1
    percent = (a * 100) / b

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.title("sift_blur")
    plt.imshow(img3)
    plt.savefig(os.path.join(RESULTS_FOLDER, "sift_blur.png"))
    plt.close()

    return {"score": round(percent, 2), "match": percent >= 40.00, "image_url": "/compare_results/sift_blur.png"}


def orb(img1, img2):
    img1 = cv.resize(img1, (400, 400))
    img2 = cv.resize(img2, (400, 400))

    detector = cv.ORB_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    a = len(good)
    b = len(kp2)
    if b == 0:
        b = 1
    percent = (a * 100) / b

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.title("orb")
    plt.imshow(img3)
    plt.savefig(os.path.join(RESULTS_FOLDER, "orb.png"))
    plt.close()

    return {"score": round(percent, 2), "match": percent >= 40.00, "image_url": "/compare_results/orb.png"}


def orb_blur(img1, img2):
    img1 = cv.resize(img1, (400, 400))
    img2 = cv.resize(img2, (400, 400))
    img1 = cv.blur(img1, (8, 8))
    img2 = cv.blur(img2, (8, 8))

    detector = cv.ORB_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    a = len(good)
    b = len(kp2)
    if b == 0:
        b = 1
    percent = (a * 100) / b

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.title("orb_blur")
    plt.imshow(img3)
    plt.savefig(os.path.join(RESULTS_FOLDER, "orb_blur.png"))
    plt.close()

    return {"score": round(percent, 2), "match": percent >= 40.00, "image_url": "/compare_results/orb_blur.png"}



def compare_all():
    img1, img2 = _load_images()

    if img1 is None or img2 is None:
        return jsonify({"error": "Both images must be uploaded first"}), 400

    results = {
        "sift":      sift(img1.copy(), img2.copy()),
        "sift_blur": sift_blur(img1.copy(), img2.copy()),
        "orb":       orb(img1.copy(), img2.copy()),
        "orb_blur":  orb_blur(img1.copy(), img2.copy()),
    }

    for key, val in results.items():
        if val is None:
            results[key] = {"error": "Could not detect keypoints"}

    return jsonify(results), 200


def serve_compare_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)













