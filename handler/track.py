import cv2

from .base import HandlerBase


class Tracker(HandlerBase):
    def annotate_frame(self, frame: cv2.typing.MatLike) -> (cv2.typing.MatLike, int):
        results = self.model.track(frame, persist=True, conf=getattr(self, 'conf', 0.2), imgsz=getattr(self, 'imgsz', 640))
        try:
            tracked_hays = results[0].boxes.id.int().cpu().tolist()
            self.counter.update(tracked_hays)
            if self.hide_labels:
                return self.custom_box(results, frame), len(tracked_hays)
            return results[0].plot(), tracked_hays
        except Exception:
            return frame, 0
