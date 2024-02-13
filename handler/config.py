import argparse
from handler.constants import SIZE


def configure_argument_parser():
    parser = argparse.ArgumentParser(description='Objects detection, tacking and counting with YOLO8 model')

    parser.add_argument('mode', choices=('tracking', 'predict'), help='Choose mode of detection objects')

    parser.add_argument('-m', '--model', type=str, default='models/best.pt', help='Path to the YOLOv8 model file.')
    parser.add_argument('-v', '--video', type=str, default='data/videos/SAR.mp4', help='Path to the video file for analysis.')
    parser.add_argument('-s', '--save', type=str, default='data/result.mp4', help='Path to the save video.')
    parser.add_argument('-S', '--show', action='store_true', help='Display results on screen.')

    parser.add_argument('-c', '--conf', type=float, default=0.01, help='Confidence threshold for object detection. (from 0 to 1)')
    parser.add_argument('--imgsz', choices=SIZE.keys(), default='s1K', help='Image size for processing.')

    parser.add_argument('--draw_lines', action='store_true', help='Draw history lines on the video.')
    parser.add_argument('--lines_history', type=int, default=50, help='Number of frames to keep lines history.')

    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--init_model', type=str, default='YOLO8L', help='Initial model name. (for debug data)')

    parser.add_argument('-f', '--framerate', type=int, default=30, help='Result video framerate.')
    parser.add_argument('--skip_frames', type=int, default=0, help='Skip frames for better speed of analyzing.')
    parser.add_argument('--start_frame', type=int, default=0, help='First frame of video for better speed of analyzing.')

    return parser
