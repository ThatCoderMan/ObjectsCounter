from handler.track import Tracker
from handler.predict import Predictor
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':
    tracker = Tracker(show=False, video_path='data/videos/hay_v1_fhd.mp4', conf=0.1, imgsz=(3840, 2144))
    # tracker.process_video()
    tracker.save_video()
    # predictor = Predictor(show=True, conf=0.1, imgsz=(3840, 2144))
    # predictor.process_video()
