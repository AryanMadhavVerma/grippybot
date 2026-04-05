from servo_driver import ServoDriver
import time

d = ServoDriver()
d.home()
time.sleep(1)

d.set_angle("base", 45)
print("base:", d.get_angle("base"))
time.sleep(1)

d.set_angle("shoulder", 45)
print("shoulder:", d.get_angle("shoulder"))
time.sleep(2)

d.close()
