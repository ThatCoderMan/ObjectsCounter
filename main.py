import os

from handler.track import Tracker
from handler.predict import Predictor
from handler.constants import Size


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    tracker = Tracker(show=True, video_path='data/videos/video.mp4', conf=0.1, imgsz=Size.sd)
    tracker.process_video()
    # predictor = Predictor(show=True, conf=0.1, imgsz=(3840, 2144))
    # predictor.process_video()
