import pytest
import numpy as np
from services.dct import encode_watermark_into_image, decode_watermark_from_image
import cv2 

def test_dct_add_and_recover_watermark():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "test 123"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_and_recover_watermark() = wm:{wm}, decode:{decode}")
    assert wm in decode


def test_dct_add_and_recover_diff_watermark():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "this is test 2"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_and_recover_diff_watermark() = wm:{wm}, decode:{decode}")
    assert wm in decode 


def test_dct_add_and_30_char_to_watermark():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "a"*30
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_and_30_char_to_watermark() = {len(decode)}, first 30 char added")
    assert len(decode) <= 30


def test_dct_add_and_more_char_to_watermark():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "a"*33
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_and_more_char_to_watermark() = {len(decode)}, first 30 char added")
    assert len(decode) <= 30


def test_dct_add_and_empty_watermark():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = ""
    with pytest.raises(ZeroDivisionError):
        encode_watermark_into_image(fake_img.copy(),wm)
        print("test_dct_add_and_empty_watermark()")


def test_dct_add_on_flat_img():
    fake_img = np.full((256,256),200,dtype=np.uint8)
    wm = "test123"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_on_flat_img() = wm:{wm}, decode:{decode}")
    assert wm in decode


def test_dct_add_min_size_img():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(8,8), dtype=np.uint8)
    wm = "a"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_min_size_img() = wm:{wm}, decode:{decode}")
    assert wm not in decode


def test_dct_add_min_size_img_30char():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(8,8), dtype=np.uint8)
    wm = "a"*28
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    decode = decode_watermark_from_image(watermark_img)
    print(f"test_dct_add_min_size_img_30char() = wm:{wm}, decode:{decode}")
    assert wm not in decode

def test_dct_add_and_resize_img():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "resize test"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    resize_down = cv2.resize(watermark_img, None, fx=0.5, fy=0.5)
    resize_back = cv2.resize(resize_down,(watermark_img.shape[1],watermark_img.shape[0]))
    decode = decode_watermark_from_image(resize_back)
    print(f"test_dct_add_and_resize_img() = wm:{wm}, decode:{decode}")
    assert wm not in decode


def test_dct_add_and_compress_img():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "compress img"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encoded_img = cv2.imencode('.jpg', watermark_img, encode_param)
    jpeg = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    decode = decode_watermark_from_image(jpeg)
    print(f"test_dct_add_and_compress_img = wm:{wm}, decode:{decode}")
    assert wm in decode


def test_dct_add_and_20_compress_img():
    np.random.seed(42)
    fake_img = np.random.randint(0,256,(256,256), dtype=np.uint8)
    wm = "compress img"
    watermark_img = encode_watermark_into_image(fake_img.copy(),wm)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    result, encoded_img = cv2.imencode('.jpg', watermark_img, encode_param)
    jpeg = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    decode = decode_watermark_from_image(jpeg)
    print(f"test_dct_add_and_20_compress_img() = wm:{wm}, decode:{decode}")
    assert wm not in decode


    