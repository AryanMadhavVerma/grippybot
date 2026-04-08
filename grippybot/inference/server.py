"""
Inference server — runs on Raspberry Pi.
Captures frames + joint state, sends to Mac, receives predicted actions, executes.

Usage on Pi:
    grippybot-server --host 0.0.0.0 --port 5555
"""

import argparse
import io
import time
from PIL import Image

from grippybot.hardware import ServoDriver, Camera
from grippybot.inference.protocol import send_frame, recv_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    import socket

    # Init hardware
    driver = ServoDriver()
    camera = Camera()
    driver.home()
    time.sleep(2)

    # Wait for Mac to connect
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Waiting for Mac to connect on port {args.port}...")

    conn, addr = server.accept()
    print(f"Connected: {addr}")

    step = 0
    try:
        while True:
            t_start = time.time()

            # Capture frame as JPEG bytes
            frame = camera.capture_frame()  # numpy HWC RGB
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            jpeg_bytes = buf.getvalue()

            # Read current joint state
            angles = driver.get_all_angles()
            gripper = 1.0 if driver.is_gripper_open() else 0.0
            state = [
                angles.get("base", 0.0) or 0.0,
                angles.get("shoulder", 0.0) or 0.0,
                angles.get("elbow", 0.0) or 0.0,
                angles.get("wrist", 0.0) or 0.0,
            ]

            # Send to Mac
            send_frame(conn, jpeg_bytes, state, gripper)

            # Receive predicted action from Mac
            action = recv_msg(conn)
            if action is None:
                print("Connection closed by Mac.")
                break

            # Execute action
            driver.set_angle("base", float(action["base"]))
            driver.set_angle("shoulder", float(action["shoulder"]))
            driver.set_angle("elbow", float(action["elbow"]))
            driver.set_angle("wrist", float(action["wrist"]))
            if action["gripper"] > 0.5:
                driver.gripper_open()
            else:
                driver.gripper_close()

            if step % 5 == 0:
                print(f"step {step:3d} | action: base={action['base']:6.1f} "
                      f"sh={action['shoulder']:6.1f} el={action['elbow']:6.1f} "
                      f"wr={action['wrist']:6.1f} grip={'OPEN' if action['gripper'] > 0.5 else 'CLOSE'}")

            step += 1

            # Match FPS
            elapsed = time.time() - t_start
            time.sleep(max(0, 1.0 / args.fps - elapsed))

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        driver.home()
        time.sleep(1)
        driver.close()
        camera.close()
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
