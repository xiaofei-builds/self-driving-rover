from vilib import Vilib
from time import sleep

Vilib.camera_start(vflip=False, hflip=False)
Vilib.display(local=False, web=True)
print("Streaming. Open http://<PI-IP>:9000/mjpg in your laptop browser.")
print("Ctrl+C here to stop.")
try:
    while True:
        sleep(1)
except KeyboardInterrupt:
    Vilib.camera_close()
    print("Camera closed.")
