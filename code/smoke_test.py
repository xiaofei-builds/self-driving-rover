from picarx import Picarx
import time
px = Picarx()          # inits board, centers all servos
time.sleep(1)
print("Ultrasonic distance:", px.get_distance(), "cm")
print("Grayscale sensors  :", px.get_grayscale_data())
print("Smoke test OK - nothing should have driven.")
