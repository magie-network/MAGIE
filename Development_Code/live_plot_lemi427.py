import struct
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

FILE = "/home/simon/MARTAS/mqtt/LEMI417_3_0001/LEMI417_3_0001_2026-03-12.bin"

FMT = ">6hLfffhhiiii"
SIZE = struct.calcsize(FMT)

plt.ion()

fig, ax = plt.subplots()

while True:

    times = []
    bx = []
    by = []
    bz = []

    with open(FILE, "rb") as f:

        f.readline()

        while True:
            chunk = f.read(SIZE)
            if not chunk:
                break

            rec = struct.unpack(FMT, chunk)

            year,month,day,hour,minute,second,ms = rec[:7]

            dt = datetime(year,month,day,hour,minute,second) + timedelta(milliseconds=ms)

            times.append(dt)
            bx.append(rec[7])
            by.append(rec[8])
            bz.append(rec[9])

    ax.clear()

    # Lines connecting all points
    ax.plot(times, bx, '-', label='Bx')
    ax.plot(times, by, '-', label='By')
    ax.plot(times, bz, '-', label='Bz')

    # Dot for most recent value
    if times:
        ax.plot(times[-1], bx[-1], 'o')
        ax.plot(times[-1], by[-1], 'o')
        ax.plot(times[-1], bz[-1], 'o')

    ax.set_ylabel("nT")
    ax.set_title("LEMI417 Live Magnetometer")
    ax.legend()

    plt.pause(1)
