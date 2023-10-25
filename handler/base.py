from abc import ABC, abstractmethod

import cv2
import ultralytics
from ultralytics import YOLO
from pathlib import Path
from typing import Union
from tqdm import tqdm
import cv2
import numpy as np
import glob

class HandlerBase(ABC):

    def __init__(
            self,
            model: Union[str, Path] = 'models/best.pt',
            video_path: Union[str, Path] = 'data/videos/hay_v1_fhd.mp4',
            save_path: Union[str, Path] = 'data/results/',
            show: bool = False,
            hide_labels: bool = True,
            **kwargs
    ):
        self.model = YOLO(model)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.counter = set()
        self.show = show
        self.save_path = save_path
        self.hide_labels = hide_labels
        for key, value in kwargs.items():
            setattr(self, key, value)

    def process_video(self):
        self.prepare_model()
        if not self.show:
            frame_cnt = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                annotated_frame = self.annotate_frame(frame)
                annotated_frame = self.counter_box(annotated_frame)
                print('Hay counted:', len(self.counter))
                if self.show:
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                else:
                    cv2.imwrite(self.save_path + f'frame_{frame_cnt:0>4}.jpg', annotated_frame)
                    frame_cnt += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()

        print("Total objects count:", len(self.counter))

    @abstractmethod
    def annotate_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        pass

    def prepare_model(self):
        pass

    def custom_box(self, results: ultralytics.engine.results.Results, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        boxes = results[0].boxes.xywh.cpu()
        annotated_frame = frame.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(
                annotated_frame,
                (int(x - w / 2), int(y - h / 2)),
                (int(x + w / 2), int(y + h / 2)),
                (0, 255, 0),
                2
            )
        return annotated_frame

    def counter_box(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Objects found: {len(self.counter)}"
        text_x, text_y = 20, 40
        text_size = 0.7
        cv2.putText(frame, text, (text_x, text_y), font, text_size, (255, 255, 255))
        return frame

    def save_video(self, framerate: int = 30):
        images = sorted(glob.glob(self.save_path+'/*.jpg'))
        if images:
            print('Saving video')
            height, width, _ = cv2.imread(images[0]).shape
            out = cv2.VideoWriter('data/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), framerate, (width, height))
            for filename in tqdm(images):
                img = cv2.imread(filename)
                out.write(img)
            out.release()
            print('Video saved')
        print('No images to save video')
