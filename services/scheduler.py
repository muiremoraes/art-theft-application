
from apscheduler.schedulers.background import BackgroundScheduler
from services.reverse_search import process_scans


def start_scheduler(app):
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=lambda: process_scans(app),
        trigger="interval",
        minutes=10,
    )

    scheduler.start()
    print("started")


