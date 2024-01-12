import time
import numpy as np
# import cv2


class Utils():
    WHITE = 0
    BLACK = 1
    GRAY = 2

    @staticmethod
    def get_current_time():
        return int(round(time.time()))

    @staticmethod
    def get_color(color):
        if color == Utils.WHITE:
            return (255, 255, 255)
        elif color == Utils.BLACK:
            return (0, 0, 0)
        elif color == Utils.GRAY:
            return (80, 80, 80)
