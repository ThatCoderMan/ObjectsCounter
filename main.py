from handler.track import Tracker
from handler.predict import Predictor
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':
    tracker = Tracker(show=True, conf=0.1, imgsz=(1920, 1080))
    tracker.process_video()
    # predictor = Predictor(show=True, conf=0.15, imgsz=(1920, 1056))
    # predictor.process_video()
