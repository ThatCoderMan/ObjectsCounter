import os

from handler.track import Tracker
from handler.predict import Predictor
from handler.constants import Size


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    tracker = Tracker(show=True, video_path='data/videos/hay_v1_fhd.mp4', conf=0.1, imgsz=Size.s4K)
    tracker.process_video()
    # tracker.save_video()
    # predictor = Predictor(show=True, video_path='data/videos/hay_v1_fhd.mp4', conf=0.1, imgsz=Size.s1K)
    # predictor.process_video(framerate=1, skip_frames=29)
