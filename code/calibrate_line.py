from picarx import Picarx
import time

px = Picarx()

def avg_reading(n=30):
    tot = [0, 0, 0]
    for _ in range(n):
        g = px.get_grayscale_data()   # [left, middle, right]
        tot = [tot[i] + g[i] for i in range(3)]
        time.sleep(0.02)
    return [round(t / n, 1) for t in tot]

input("1) Put ALL THREE sensors over BARE FLOOR, then press Enter...")
floor = avg_reading()
print("   floor baseline :", floor)

input("2) Slide car so the LINE sits under the sensors, then press Enter...")
tape = avg_reading()
print("   tape reading   :", tape)

# tape reads HIGHER than floor on your surface -> threshold = midpoint per sensor
thresh = [round((floor[i] + tape[i]) / 2, 1) for i in range(3)]
delta  = [round(tape[i] - floor[i], 1) for i in range(3)]
print()
print("   per-sensor delta (signal):", delta)
print("   THRESHOLDS (on-line if reading > this):", thresh)
