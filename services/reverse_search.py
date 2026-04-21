
import os
from models.user_model import User, db
from models.scan_model import ScanJob, ScanResult
from models.image_model import Image
from flask import jsonify
from api_config import SERPAPI_KEY, IMAGE_BB_KEY
from datetime import datetime, timedelta
import requests

def start_scan(user_id):

    thirty_days = datetime.utcnow() - timedelta(days=30)
    last_scan = ScanJob.query.filter_by(user_id=user_id, status="done").filter(ScanJob.created >= thirty_days).first()

    if last_scan:
        return jsonify({"error": "one scan per 30 days"}), 429
  
    new_scan = ScanJob(user_id=user_id, status="waiting")
    db.session.add(new_scan)
    db.session.commit()

    return jsonify({"message": "scanning now, check back later"}),200


def process_scans(app):
    with app.app_context():
        oldest_job = ScanJob.query.filter_by(status="waiting").order_by(ScanJob.created.asc()).first()

        if oldest_job is None:
            print("no jobs")##
            return 

        oldest_job.status = "doing"
        print("doing")##
        db.session.commit()

        user_images = Image.query.filter_by(user_id=oldest_job.user_id).all()

        for image in user_images:
            print("searching")##
            matches = search_image(image)

            for match in matches[:5]:
                print("matches")##
                result = ScanResult(
                    user_id = oldest_job.user_id,
                    image_id = image.id,
                    job_id = oldest_job.id,
                    website_url = match.get("link"),
                    website_title = match.get("title"),
                    thumbnail_url = match.get("thumbnail"),
                )

                db.session.add(result)

        oldest_job.status = "done"
        db.session.commit()



def search_image(image):
    response = requests.get(
            "https://serpapi.com/search",
            params={"engine":"google_lens", "api_key":SERPAPI_KEY, "url":image.image_url},
            timeout=30
        )
    # image_path = os.path.join("uploaded_images", image.file_path)
    # print("image path",image_path)##

    # with open(image_path, "rb") as f:
    #     response = requests.post(
    #         "https://serpapi.com/search",
    #         params={"engine":"google_lens", "api_key":SERPAPI_KEY},
    #         files = {"image_file":f},
    #         timeout=30
    #     )

    print(response.status_code)
    print(response.text[:500])

    if response.status_code != 200:
        return []

    data = response.json()
    return data.get("visual_matches",[])

        

def get_results(user_id):
    job = ScanJob.query.filter_by(user_id=user_id, status="done").order_by(ScanJob.created.desc()).first()

    if job is None:
        return jsonify({"results":[]}),200

    results = ScanResult.query.filter_by(job_id=job.id).all()

    return jsonify({
        "results":[
            {
                "website_url": r.website_url,
                "website_title": r.website_title,
                "thumbnail_url": r.thumbnail_url,
            }
            for r in results
        ]
    }),200


