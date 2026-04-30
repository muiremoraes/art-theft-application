import pytest
import numpy as np
from services.steganography import encode, decode
import cv2 



def test_lsb_add_and_get_wm():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256, 3), dtype=np.uint8)
    wm = "test 123"
    encoded_img = encode(fake_img.copy(), wm)
    decoded = decode(encoded_img)
    print(f"test_lsb_add_and_get_wm() = lsb wm:{wm}, decode:{decoded}")
    assert decoded == wm



def test_lsb_add_and_no_wm():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256, 3), dtype=np.uint8)
    wm = ""
    encoded_img = encode(fake_img.copy(), wm)
    decoded = decode(encoded_img)
    print(f"test_lsb_add_and_no_wm() = lsb wm:{wm}, decode:{decoded}")
    assert decoded == wm


def test_lsb_add_and_34char_wm():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256, 3), dtype=np.uint8)
    wm = "a"*34
    encoded_img = encode(fake_img.copy(), wm)
    decoded = decode(encoded_img)
    print(f"test_lsb_add_and_34char_wm() = lsb wm:{wm}, decode:{decoded}")
    assert decoded == wm


def test_lsb_add_and_34char_wm_small_img():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(8,8, 3), dtype=np.uint8)
    wm = "a"*34
    with pytest.raises(ValueError):
        print(f"test_lsb_add_and_34char_wm_small_img()")
        encode(fake_img.copy(), wm)
    

def test_lsb_resize_and_get_wm():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256, 3), dtype=np.uint8)
    wm = "test 123"
    encoded_img = encode(fake_img.copy(), wm)
    resize_down = cv2.resize(encoded_img, None, fx=0.5, fy=0.5)
    resize_back = cv2.resize(resize_down,(encoded_img.shape[1],encoded_img.shape[0]))
    decoded = decode(resize_back)
    print(f"test_lsb_resize_and_get_wm() = lsb wm:{wm}, decode:{decoded}")
    assert decoded != wm


def test_lsb_compress_and_get_wm():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256, 3), dtype=np.uint8)
    wm = "test 123"
    encoded_img = encode(fake_img.copy(), wm)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encoded_img = cv2.imencode('.jpg', encoded_img, encode_param)
    jpeg = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    decoded = decode(jpeg)
    print(f"test_lsb_compress_and_get_wm() = lsb wm:{wm}, decode:{decoded}")
    assert decoded != wm


