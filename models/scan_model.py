
from datetime import datetime
from .user_model import db

class ScanJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(30), default="waiting")



class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey("scan_job.id"), nullable=False)
    website_url = db.Column(db.String(600), nullable=False)
    website_title = db.Column(db.String(600), nullable=True)
    thumbnail_url = db.Column(db.String(600), nullable=True)
    is_safe = db.Column(db.Boolean, nullable=True)

