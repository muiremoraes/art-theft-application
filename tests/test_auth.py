import pytest


def test_register(client):
    response=client.post("/register", json={
        "username":"testuser",
        "email":"missrandom370@gmail.com",
        "password": "StrongPass@123"
    })
    print("test_register(client) = user created")
    assert response.status_code == 201


def test_register_empty_fields(client):
    response=client.post("/register", json={
        "username":"",
        "email":"",
        "password": ""
    })
    print("test_register_empty_fields(client) = bad request")
    assert response.status_code == 400

def test_register_invalid_email(client):
    response=client.post("/register", json={
        "username":"testuser",
        "email":"notgoodemail",
        "password": "StrongPass@123"
    })
    print("test_register_invalid_email(client) = bad request")
    assert response.status_code == 400


def test_register_short_password(client):
    response=client.post("/register", json={
        "username":"testuser",
        "email":"missrandom370@gmail.com",
        "password": "abc"
    })
    print("test_register_short_password(client) = bad request")
    assert response.status_code == 400

def test_register_no_caps_password(client):
    response=client.post("/register", json={
        "username":"testuser",
        "email":"missrandom370@gmail.com",
        "password": "testuser@123"
    })
    print("test_register_no_caps_password(client) = bad request")
    assert response.status_code == 400


def test_login_empty_fields(client):
    response=client.post("/login", json={
        "email":"",
        "password": ""
    })
    print("test_login_empty_fields(client) = bad request")
    assert response.status_code == 400

def test_login_no_valid_email(client):
    response=client.post("/login", json={
        "email":"missrandom467@gmail.com",
        "password": "randompass"
    })
    print("test_login_no_valid_email(client) = no enumeration")
    assert response.status_code == 200


def test_login(client):
    client.post("/register", json={
        "username":"testuser",
        "email":"missrandom467@gmail.com",
        "password": "StrongPass@123"
    })
    response=client.post("/login", json={
        "email":"missrandom467@gmail.com",
        "password": "StrongPass@123"
    })
    print("test_login(client) = login okay")
    assert response.status_code == 200