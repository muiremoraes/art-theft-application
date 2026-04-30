import pytest


@pytest.fixture
def app():
    from main import app as flask_app
    from models.user_model import db

    flask_app.config["TESTING"]=True
    flask_app.config["SQLALCHEMY_DATABASE_URI"]= "sqlite:///:memory:"
    flask_app.config['RATELIMIT_ENABLED'] = True
    flask_app.config["MAIL_SUPPRESS_SEND"]=True

    from main import limiter
    limiter.reset()

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()
    

@pytest.fixture
def client(app):
    return app.test_client()


def test_ratelimit_register(client):
    for i in range(5):
        response = client.post("/register", json={})
        assert response.status_code != 429,"rate limit test = test_ratelimit_register(client)"

    response = client.post("/register", json={})
    print("rate limit test more than limit = test_ratelimit_register(client)")
    assert response.status_code == 429
    


def test_ratelimit_login(client):
    for i in range(5):
        response = client.post("/login", json={})
        assert response.status_code != 429,"rate limit test = test_ratelimit_login(client)"

    response = client.post("/login", json={})
    print("rate limit test more than limit = test_ratelimit_login(client)")
    assert response.status_code == 429


def test_ratelimit_compare(client):
    for i in range(10):
        response = client.post("/compare", json={})
        assert response.status_code != 429, "rate limit test = test_ratelimit_compare(client)"

    response = client.post("/compare", json={})
    print("rate limit test more than limit = test_ratelimit_compare(client)")
    assert response.status_code == 429


def test_ratelimit_upload(client):
    for i in range(6):
        response = client.post("/upload", json={})
        assert response.status_code != 429, "rate limit test = test_ratelimit_upload(client)"

    response = client.post("/upload", json={})
    print("rate limit test more than limit = test_ratelimit_upload(client)")
    assert response.status_code == 429