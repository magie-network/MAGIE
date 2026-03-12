from twisted.protocols.basic import LineReceiver
import struct
import datetime
import os

from martas.core import methods as mm

FRAME_SIGNATURE = b"L417"
FRAME_SIZE = 47


def bcd_to_int(v):
    return ((v >> 4) * 10) + (v & 0x0F)


def data2_to_double(z):
    tmp = z[1]*(2**8) + z[0]
    if z[1] > 128:
        tmp -= 2**16
    return tmp / 100.0


def data3_to_double(z):
    tmp = z[2]*(2**16) + z[1]*(2**8) + z[0]
    if z[2] > 128:
        tmp -= 2**24
    return tmp / 100.0


def data4_to_double(z):
    tmp = z[3]*(2**24) + z[2]*(2**16) + z[1]*(2**8) + z[0]
    if z[3] > 128:
        tmp -= 2**32
    return tmp / 100.0


class Lemi417Protocol(LineReceiver):

    def __init__(self, client, sensordict, confdict):

        self.client = client
        self.sensordict = sensordict
        self.confdict = confdict

        self.sensor = sensordict.get("sensorid")
        self.station = confdict.get("station")
        self.qos = int(confdict.get("mqttqos", 0))

        self.buffer = b""
        self.datacnt = 0

        self._init_file()

    # --------------------------------------------------

    def _init_file(self):

        bufferpath = (
            self.confdict.get("bufferpath")
            or self.confdict.get("mqttbuffer")
            or self.confdict.get("bufferdirectory")
        )

        if not bufferpath:
            raise Exception("No MARTAS buffer path defined")

        self.sensorpath = os.path.join(bufferpath, self.sensor)
        os.makedirs(self.sensorpath, exist_ok=True)

        self.filename = os.path.join(
            self.sensorpath,
            self.sensor + "_" +
            datetime.datetime.utcnow().strftime("%Y-%m-%d") + ".bin"
        )

        if not os.path.exists(self.filename):

            header = (
                "# MagPyBin {} {} {} {} {} {} {}\n".format(
                    self.sensor,
                    "[x,y,z,t1,t2,var1,var2,var3,var4]",
                    "[X,Y,Z,Te,Tf,E1,E2,E3,E4]",
                    "[nT,nT,nT,deg,deg,V,V,V,V]",
                    "[1,1,1,1,1,100,100,100,100]",
                    "6hLfffhhiiii",
                    struct.calcsize(">6hLfffhhiiii")
                )
            )

            with open(self.filename, "wb") as f:
                f.write(header.encode("ascii"))

    # --------------------------------------------------

    def dataReceived(self, data):

        self.buffer += data

        while True:

            idx = self.buffer.find(FRAME_SIGNATURE)

            if idx < 0:
                self.buffer = self.buffer[-3:]
                return

            if len(self.buffer) < idx + FRAME_SIZE:
                return

            frame = self.buffer[idx:idx+FRAME_SIZE]
            self.buffer = self.buffer[idx+FRAME_SIZE:]

            payload = frame[4:]

            try:
                unpacked = struct.unpack(
                    "2B B B B B B B B 2B 3B 3B 3B 4B 4B 4B 4B 2B 2B 2B B",
                    payload
                )
            except:
                continue

            year = bcd_to_int(unpacked[3])
            month = bcd_to_int(unpacked[4])
            day = bcd_to_int(unpacked[5])
            hour = bcd_to_int(unpacked[6])
            minute = bcd_to_int(unpacked[7])
            second = bcd_to_int(unpacked[8])

            try:
                gpstime = datetime.datetime(
                    2000 + year,
                    month,
                    day,
                    hour,
                    minute,
                    second
                )
            except:
                continue

            bx = data3_to_double(unpacked[11:14])
            by = data3_to_double(unpacked[14:17])
            bz = data3_to_double(unpacked[17:20])

            e1 = data4_to_double(unpacked[20:24])
            e2 = data4_to_double(unpacked[24:28])
            e3 = data4_to_double(unpacked[28:32])
            e4 = data4_to_double(unpacked[32:36])

            tf = data2_to_double(unpacked[36:38])
            te = data2_to_double(unpacked[38:40])

            datalst = mm.time_to_array(
                gpstime.strftime("%Y-%m-%d %H:%M:%S.%f")
            )

            datalst += [
                bx, by, bz,
                int(te*100),
                int(tf*100),
                int(e1*100),
                int(e2*100),
                int(e3*100),
                int(e4*100)
            ]

            dataarray = ",".join(map(str, datalst))

            topic = self.station + "/" + self.sensor

            try:
                self.client.publish(topic + "/data", dataarray, qos=self.qos)
            except:
                pass

            sectime = int(gpstime.timestamp())

            rec = struct.pack(
                ">6hLfffhhiiii",
                gpstime.year,
                gpstime.month,
                gpstime.day,
                gpstime.hour,
                gpstime.minute,
                gpstime.second,
                0,
                float(bx),
                float(by),
                float(bz),
                int(round(te*100)),
                int(round(tf*100)),
                int(round(e1*100)),
                int(round(e2*100)),
                int(round(e3*100)),
                int(round(e4*100))
            )

            with open(self.filename, "ab") as f:
                f.write(rec)

            self.datacnt += 1