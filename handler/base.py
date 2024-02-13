import glob
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from collections import defaultdict

import yaml
import cv2
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO

logging.disable(logging.CRITICAL)
logger = logging.getLogger('ultralytics')
logger.setLevel(logging.CRITICAL)


class HandlerBase(ABC):
    def __init__(
            self,
            model: Union[str, Path] = 'models/best.pt',
            video_path: Union[str, Path] = 'data/videos/hay_v1_fhd.mp4',
            save_path: Union[str, Path] = 'data/results/',
            show: bool = False,
            hide_labels: bool = True,
            draw_lines: bool = True,
            lines_history: int = 0,
            debug=False,
            init_model='l',
            **kwargs,
    ):
        self.model = YOLO(model)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.counter = set()
        self.show = show
        self.save_path = save_path
        self.hide_labels = hide_labels
        self.history = defaultdict(list)
        self.draw_lines = draw_lines
        self.lines_history = lines_history
        self.frame_cnt = 0
        self.debug = debug
        self.init_model = init_model
        for key, value in kwargs.items():
            setattr(self, key, value)
        with open('models/custom_tracker.yaml', 'r') as f:
            self.tracker_options = yaml.safe_load(f)

    def process_video(self, framerate: int = 30, skip_frames: int = 0, start_frame: int = 0):
        self.prepare_model()
        print('start processing video with')
        frame_cnt = 0
        with tqdm(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // max(skip_frames, 1)) as pbar:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if (skip_frames and frame_cnt % skip_frames != 0) or frame_cnt < start_frame:
                    frame_cnt += 1
                    continue
                if success:
                    annotated_frame, frame_hay_cnt = self.annotate_frame(frame)
                    annotated_frame = self.counter_box(annotated_frame, frame_hay_cnt)
                    if self.debug:
                        annotated_frame = self.info_box(annotated_frame)
                    if self.show:
                        cv2.imshow('YOLOv8 Tracking', annotated_frame)
                    else:
                        cv2.imwrite(self.save_path + f'frame_{frame_cnt:0>4}.jpg', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    frame_cnt += 1
                    pbar.update(1)
                else:
                    break
        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()
        print('Total objects count:', len(self.counter))
        if not self.show:
            self.save_video(framerate)

    @abstractmethod
    def annotate_frame(self, frame: cv2.typing.MatLike) -> (cv2.typing.MatLike, int):
        pass

    @abstractmethod
    def draw_history(self, results: ultralytics.engine.results.Results,
                     frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        pass

    def prepare_model(self):
        pass

    def custom_box(self, results: ultralytics.engine.results.Results, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        self.frame_cnt += 1
        if self.draw_lines:
            frame = self.draw_history(results, frame)
        boxes = results[0].boxes.xywh.cpu()
        annotated_frame = frame.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                          (0, 255, 0), 2)
        return annotated_frame

    def counter_box(self, frame: cv2.typing.MatLike, frame_hay_cnt: int) -> cv2.typing.MatLike:
        cv2.rectangle(frame, (10, 10), (235, 80), (100, 100, 100, 100), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Total: {len(self.counter)}'
        text_x, text_y = 20, 40
        text_size = 1
        cv2.putText(frame, text, (text_x, text_y), font, text_size, (255, 255, 255))
        text = f'Current: {frame_hay_cnt}'
        text_y = 70
        cv2.putText(frame, text, (text_x, text_y), font, text_size, (255, 255, 255))
        return frame

    def info_box(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        cv2.rectangle(frame, (980, 10), (1270, 290), (100, 100, 100, 100), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = []
        text.append(f'frame: {self.frame_cnt}')
        text.append(f'{self.init_model=}')
        imgsz = getattr(self, "imgsz", 640)
        conf = getattr(self, "conf", 0.2)
        text.append(f'{imgsz=}')
        text.append(f'{conf=}')
        text.append(f'{self.draw_lines=}')
        text.append(f'{self.lines_history=}')
        text.append('')
        for key, val in self.tracker_options.items():
            text.append(f'{key}={val}')
        text_x, text_y = 990, 40
        for text_y, text_line in enumerate(text, start=1):
            cv2.putText(frame, text_line, (text_x, 20 * text_y + 15), font, 0.6, (255, 255, 255))
        return frame

    def save_video(self, framerate: int = 30):

        images = sorted(glob.glob(self.save_path + '/*.jpg'))
        if images:
            print('Saving video')
            height, width, _ = cv2.imread(images[0]).shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('data/result.mp4', fourcc, framerate, (width, height))
            for filename in tqdm(images):
                img = cv2.imread(filename)
                out.write(img)
            out.release()
            print('Video saved')
            return
        print('No images to save video')
