"""
Filename:               lemi018protocol
Part of package:        acquisition
Type:                   Part of data acquisition library

PURPOSE:
        This package initiates LEMI-018 data acquisition and streaming,
        and saves data in MARTAS/MagPy compatible binary buffer files.

SUPPORTED INPUT FORMATS:
        1) With GPS sync  (20 comma-separated fields)
        2) Without GPS sync (12 comma-separated fields)

FIELD MAP:
        1  year
        2  month
        3  day
        4  hour
        5  minute
        6  second
        7  Bx
        8  By
        9  Bz
        10 TE
        11 TF
        12 UIN
        13 altitude          (optional)
        14 latitude          (optional)
        15 lat hemisphere    (optional)
        16 longitude         (optional)
        17 lon hemisphere    (optional)
        18 satellites        (optional)
        19 gps_fix           (optional)
        20 time_diff         (optional)

OUTPUT CHANNEL MAP:
        x     <- Bx        [nT]
        y     <- By        [nT]
        z     <- Bz        [nT]
        t1    <- TE        [deg_C]
        t2    <- TF        [deg_C]
        var2  <- UIN       [V]

NOTES:
        - Primary time is taken from the instrument time fields.
        - Optional GPS quality/status information is published in the dict
          payload but not written as a data column in this first version.
        - The local buffer file is written as MagPyBin-compatible binary data.
"""

import os
import socket
import string
import struct
import sys
from datetime import datetime, timezone

from twisted.protocols.basic import LineReceiver
from twisted.python import log

from martas.core import methods as mm


class Lemi018Protocol(LineReceiver):
    """
    Protocol to read LEMI018 ASCII records.

    This protocol defines the sensor-specific parsing for newline-delimited
    comma-separated LEMI018 output.
    """

    delimiter = b"\n"

    def __init__(self, client, sensordict, confdict):
        print("Initializing LEMI018")
        self.client = client
        self.sensordict = sensordict
        self.confdict = confdict

        self.count = 0
        self.sensor = sensordict.get("sensorid")
        self.hostname = socket.gethostname()
        self.printable = set(string.printable)
        self.datalst = []
        self.datacnt = 0
        self.metacnt = 10

        self.last_gps_fix = ""
        self.last_satellites = ""
        self.last_time_diff = ""
        self.last_altitude = ""
        self.last_latitude = ""
        self.last_longitude = ""
        self.last_raw_line = ""

        self.qos = int(confdict.get("mqttqos", 0))
        if self.qos not in [0, 1, 2]:
            self.qos = 0
        log.msg("  -> setting QOS:", self.qos)

        self.pvers = sys.version_info[0]

        debugtest = confdict.get("debug")
        self.debug = False
        if debugtest == "True":
            log.msg("DEBUG - {}: Debug mode activated.".format(self.sensordict.get("protocol")))
            self.debug = True
        else:
            log.msg("  -> Debug mode = {}".format(debugtest))

        print("Initializing LEMI018 finished")

    def connectionMade(self):
        log.msg("  -> {} connected.".format(self.sensor))

    def connectionLost(self, reason):
        log.msg("  -> {} lost.".format(self.sensor))

    def _safe_float(self, value, default=None):
        try:
            if value is None:
                return default
            value = str(value).strip()
            if value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value, default=None):
        try:
            if value is None:
                return default
            value = str(value).strip()
            if value == "":
                return default
            return int(float(value))
        except Exception:
            return default

    def _buffer_path(self):
        path = os.path.join(self.confdict.get("bufferdirectory"), self.sensor)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _binary_header(self):
        packcode = "6hLffflll"
        header = "# MagPyBin {} {} {} {} {} {} {}\n".format(
            self.sensor,
            "[x,y,z,t1,t2,var2]",
            "[X,Y,Z,TE,TF,UIN]",
            "[nT,nT,nT,deg_C,deg_C,V]",
            "[0.001,0.001,0.001,100,100,10]",
            packcode,
            struct.calcsize("<" + packcode),
        )
        if self.pvers > 2:
            return header.encode("ascii")
        return header

    def _meta_header(self):
        sendpackcode = "6hLffflll"
        return "# MagPyBin {} {} {} {} {} {} {}".format(
            self.sensor,
            "[x,y,z,t1,t2,var2]",
            "[X,Y,Z,TE,TF,UIN]",
            "[nT,nT,nT,deg_C,deg_C,V]",
            "[0.001,0.001,0.001,100,100,10]",
            sendpackcode,
            struct.calcsize("<" + sendpackcode),
        )

    def _write_binary_record(self, path, date, datearray, bx, by, bz, te, tf, uin):
        """
        Write one MagPyBin record to the local MARTAS buffer file.
        """
        outpath = os.path.join(path, self.sensor + "_" + date + ".bin")
        header = self._binary_header()

        if not os.path.exists(outpath):
            with open(outpath, "ab") as fh:
                fh.write(header)

        try:
            rec = struct.pack(
                "<6hLffflll",
                datearray[0],
                datearray[1],
                datearray[2],
                datearray[3],
                datearray[4],
                datearray[5],
                datearray[6],
                float(bx),
                float(by),
                float(bz),
                int(round(float(te) * 100.0)),
                int(round(float(tf) * 100.0)),
                int(round(float(uin) * 10.0)),
            )
            with open(outpath, "ab") as fh:
                fh.write(rec)
        except Exception as exc:
            log.err("LEMI018 - Protocol: Could not write data to file: {}".format(exc))

    def _build_dict_payload(self):
        """
        Build the MARTAS /dict payload.
        """
        add = (
            "SensorID:{},"
            "StationID:{},"
            "DataPier:{},"
            "SensorModule:{},"
            "SensorGroup:{},"
            "SensorDescription:{},"
            "DataTimeProtocol:{},"
            "DataNTPTimeDelay:{},"
            "DataGPSFixQuality:{},"
            "DataSatellites:{},"
            "DataTimeDiff:{},"
            "DataAltitude:{},"
            "DataLatitude:{},"
            "DataLongitude:{}"
        ).format(
            self.sensordict.get("sensorid", ""),
            self.confdict.get("station", ""),
            self.sensordict.get("pierid", ""),
            self.sensordict.get("protocol", ""),
            self.sensordict.get("sensorgroup", ""),
            self.sensordict.get("sensordesc", "").rstrip(),
            self.sensordict.get("ptime", ""),
            "",  # no NTP delay estimation in this first version
            self.last_gps_fix,
            self.last_satellites,
            self.last_time_diff,
            self.last_altitude,
            self.last_latitude,
            self.last_longitude,
        )
        return add

    def _normalize_parts(self, parts):
        """
        Normalize parsed CSV fields.

        Accept:
            - 12 fields: no GPS sync
            - 20 fields: GPS sync
            - 13..19 fields: partial GPS block, pad missing fields
            - >20 fields: truncate to first 20
        """
        parts = [p.strip() for p in parts]

        if len(parts) < 12:
            return None

        if len(parts) > 20:
            parts = parts[:20]

        if 12 < len(parts) < 20:
            parts = parts + [""] * (20 - len(parts))

        return parts

    def processLemi018Line(self, line):
        """
        Convert one LEMI018 ASCII record into:
            - MARTAS/MQTT data string
            - MARTAS/MQTT meta header
        and write local binary buffer output.
        """
        self.last_raw_line = line

        # Decode and clean
        if isinstance(line, bytes):
            try:
                line = line.decode("ascii", errors="ignore")
            except Exception:
                line = str(line)

        line = line.strip().replace("\r", "")
        if not line:
            return "", self._meta_header()

        # Ignore comment-like or obviously bad lines
        if line.startswith("#"):
            return "", self._meta_header()
        parts = self._normalize_parts(line.strip().split())
        if parts is None:
            if self.debug:
                log.msg("LEMI018 - Protocol: too few fields ({}): {}".format(len(line.strip().split(",")), line))
            return "", self._meta_header()

        try:
            year = self._safe_int(parts[0])
            month = self._safe_int(parts[1])
            day = self._safe_int(parts[2])
            hour = self._safe_int(parts[3])
            minute = self._safe_int(parts[4])
            second = self._safe_int(parts[5])

            bx = self._safe_float(parts[6])
            by = self._safe_float(parts[7])
            bz = self._safe_float(parts[8])
            te = self._safe_float(parts[9])
            tf = self._safe_float(parts[10])
            uin = self._safe_float(parts[11])

            # Basic validation
            required = [year, month, day, hour, minute, second, bx, by, bz, te, tf, uin]
            if any(v is None for v in required):
                if self.debug:
                    log.msg("LEMI018 - Protocol: invalid core fields in line: {}".format(line))
                return "", self._meta_header()

            gpstime = datetime(year, month, day, hour, minute, second)
            gps_time = datetime.strftime(gpstime, "%Y-%m-%d %H:%M:%S.%f")
            date = datetime.strftime(gpstime, "%Y-%m-%d")
            datearray = mm.time_to_array(gps_time)

            # Optional GPS/location block
            if len(parts) >= 20:
                altitude = parts[12]
                latitude = parts[13]
                lat_hemi = parts[14]
                longitude = parts[15]
                lon_hemi = parts[16]
                satellites = parts[17]
                gps_fix = parts[18]
                time_diff = parts[19]

                self.last_altitude = altitude
                self.last_latitude = "{}{}".format(latitude, lat_hemi).strip()
                self.last_longitude = "{}{}".format(longitude, lon_hemi).strip()
                self.last_satellites = satellites
                self.last_gps_fix = gps_fix
                self.last_time_diff = time_diff
            else:
                self.last_altitude = ""
                self.last_latitude = ""
                self.last_longitude = ""
                self.last_satellites = ""
                self.last_gps_fix = ""
                self.last_time_diff = ""

            # Write local MARTAS buffer file
            self._write_binary_record(
                self._buffer_path(),
                date,
                datearray,
                bx,
                by,
                bz,
                te,
                tf,
                uin,
            )

            # Build MQTT /data line
            # Keep the same scalar convention MARTAS uses for MagPyBin rows
            datalst = mm.time_to_array(gps_time)
            datalst.append(float(bx))
            datalst.append(float(by))
            datalst.append(float(bz))
            datalst.append(int(round(float(te) * 100.0)))
            datalst.append(int(round(float(tf) * 100.0)))
            datalst.append(int(round(float(uin) * 10.0)))

            dataarray = ",".join(map(str, datalst))
            return dataarray, self._meta_header()

        except Exception as exc:
            log.err("LEMI018 - Protocol: parse error: {} | line={!r}".format(exc, line))
            return "", self._meta_header()

    def lineReceived(self, line):
        """
        Handle one ASCII line received from the serial stream.
        """
        topic = self.confdict.get("station") + "/" + self.sensordict.get("sensorid")

        try:
            dataarray, head = self.processLemi018Line(line)
        except Exception as exc:
            log.err("LEMI018 - Protocol: Error while parsing data: {}".format(exc))
            return

        if not dataarray:
            return

        # Stack handling identical in spirit to existing MARTAS protocols
        senddata = False
        coll = int(self.sensordict.get("stack"))

        if coll > 1:
            self.metacnt = 1
            if self.datacnt < coll:
                self.datalst.append(dataarray)
                self.datacnt += 1
            else:
                senddata = True
                dataarray = ";".join(self.datalst)
                self.datalst = []
                self.datacnt = 0
        else:
            senddata = True

        if senddata:
            self.client.publish(topic + "/data", dataarray, qos=self.qos)
            if self.count == 0:
                self.client.publish(topic + "/dict", self._build_dict_payload(), qos=self.qos)
                self.client.publish(topic + "/meta", head, qos=self.qos)
            self.count += 1
            if self.count >= self.metacnt:
                self.count = 0