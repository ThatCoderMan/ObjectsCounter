from .base import HandlerBase

import cv2


class Tracker(HandlerBase):

    def annotate_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        results = self.model.track(frame, persist=True, conf=getattr(self, 'conf', 0.2),
                                   imgsz=getattr(self, 'imgsz', 640))
        try:
            self.counter.update(results[0].boxes.id.int().cpu().tolist())
            if self.hide_labels:
                return self.custom_box(results, frame)
            return results[0].plot()
        except Exception as e:
            print(e)
            return frame
