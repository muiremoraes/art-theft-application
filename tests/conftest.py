import pytest
import os

os.environ["FLASK_TESTING"]="1"

@pytest.fixture
def app():
    from main import app as flask_app
    from models.user_model import db

    flask_app.config["TESTING"]=True
    flask_app.config["SQLALCHEMY_DATABASE_URI"]= "sqlite:///:memory:"
    flask_app.config['RATELIMIT_ENABLED'] = False
    flask_app.config["MAIL_SUPPRESS_SEND"]=True

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()
    

@pytest.fixture
def client(app):
    return app.test_client()
