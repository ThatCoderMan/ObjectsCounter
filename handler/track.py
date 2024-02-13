import cv2
import torch
from ultralytics.engine.results import Results
import numpy as np
from .base import HandlerBase

device = 'cpu' if not torch.cuda.is_available() else '0'


class Tracker(HandlerBase):
    def annotate_frame(self, frame: cv2.typing.MatLike) -> (cv2.typing.MatLike, int):
        results = self.model.track(
            frame,
            persist=True,
            conf=getattr(self, 'conf', 0.2),
            imgsz=getattr(self, 'imgsz', 640),
            device=device,
            tracker='models/custom_tracker.yaml'
        )
        try:
            tracked_hays = results[0].boxes.id.int().cpu().tolist()
            self.counter.update(tracked_hays)
            if self.hide_labels:
                return self.custom_box(results, frame), len(tracked_hays)
            return results[0].plot(), tracked_hays
        except Exception:
            return frame, 0

    def draw_history(self, results: Results, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = frame.copy()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = self.history[track_id]
            track.append((float(x), float(y), self.frame_cnt))
            if len(track) > self.lines_history:
                track.pop(0)
        for track in self.history.values():
            if track[-1][-1] == self.frame_cnt:
                points = np.hstack([(x_, y_) for x_, y_, _ in track]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)
            elif track[-1][-1] + self.lines_history > self.frame_cnt:
                if len(track) > 2:
                    track.pop(0)
                points = np.hstack([(x_, y_) for x_, y_, _ in track]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(100, 100, 100), thickness=1)
        return annotated_frame

