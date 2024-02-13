import os

from handler.constants import Size
from handler.predict import Predictor
from handler.track import Tracker

import torch
import torchvision
print(f'{torch.cuda.is_available()=}')
print(f'{torch.__version__=}')
print(f'{torchvision.__version__=}')


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    tracker = Tracker(
        show=False,
        model=r'C:\Users\rokeb\PycharmProjects\Hay_test_task\runs\detect\train32\weights\best.pt',
        video_path=r'C:\Users\rokeb\PycharmProjects\Hay_test_task\data\videos\eubankgateandtrafficvideosar.mp4',
        conf=0.01,
        imgsz=Size.s1K,
        draw_lines=True,
        lines_history=50,
        debug=True,
        init_model='YOLO8L'
    )
    tracker.process_video(start_frame=150)
    # tracker.save_video()
    # predictor = Predictor(show=True, video_path='data/videos/hay_v1_fhd.mp4', conf=0.1, imgsz=Size.s1K)
    # predictor.process_video(framerate=1, skip_frames=29)
