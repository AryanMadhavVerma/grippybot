"""
Pi Camera capture wrapper for Grippy Bot.

Uses picamera2 for image capture. Resolution: 640x480.
"""

from picamera2 import Picamera2


class Camera:
    def __init__(self, resolution=(640, 480)):
        self.cam = Picamera2()
        config = self.cam.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()

    def capture_frame(self):
        """Capture a frame as numpy array (H, W, 3) RGB."""
        return self.cam.capture_array()

    def save_frame(self, path):
        """Capture and save a frame to disk as JPEG."""
        self.cam.capture_file(path)

    def close(self):
        """Stop the camera."""
        self.cam.stop()
