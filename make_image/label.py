# label.py
from datetime import datetime, timedelta
import os

ATTACK_SCHEDULES = {
    "Monday-WorkingHours.pcap": [],

    "Tuesday-WorkingHours.pcap": [
        {"start": "09:20:00", "end": "10:20:00", "type": "FTP-Patator"},
        {"start": "14:00:00", "end": "15:00:00", "type": "SSH-Patator"},
    ],

    "Wednesday-workingHours.pcap": [
        {"start": "09:47:00", "end": "10:10:00", "type": "DoS_slowloris"},
        {"start": "10:14:00", "end": "10:35:00", "type": "DoS_Slowhttptest"},
        {"start": "10:43:00", "end": "11:00:00", "type": "DoS_Hulk"},
        {"start": "11:10:00", "end": "11:23:00", "type": "DoS_GoldenEye"},
        {"start": "15:12:00", "end": "15:32:00", "type": "Heartbleed"},
    ],

    "Thursday-WorkingHours.pcap": [
        {"start": "09:20:00", "end": "10:00:00", "type": "Web_Attack_Brute_Force"},
        {"start": "10:15:00", "end": "10:35:00", "type": "Web_Attack_XSS"},
        {"start": "10:40:00", "end": "10:42:00",
            "type": "Web_Attack_Sql_Injection"},
        {"start": "14:19:00", "end": "14:21:00", "type": "Infiltration"},
        {"start": "14:33:00", "end": "14:35:00", "type": "Infiltration"},
        {"start": "14:53:00", "end": "15:00:00", "type": "Infiltration"},
        {"start": "15:04:00", "end": "15:45:00", "type": "Infiltration"},
    ],

    "Friday-WorkingHours.pcap": [
        {"start": "10:02:00", "end": "11:02:00", "type": "Botnet_ARES"},
        {"start": "13:55:00", "end": "14:24:00", "type": "PortScan"},
        {"start": "14:33:00", "end": "14:35:00", "type": "PortScan"},
        {"start": "14:51:00", "end": "15:29:00", "type": "PortScan"},
        {"start": "15:56:00", "end": "16:16:00", "type": "DDoS_LOIT"},
    ],
}


def _str_to_timeobj(s):
    hh, mm, ss = s.split(":")
    return datetime.strptime(s, "%H:%M:%S").time()


# キャッシュ化（文字列→time オブジェクト）
_SCHEDULE_CACHE = {}
for fname, entries in ATTACK_SCHEDULES.items():
    lst = []
    for e in entries:
        lst.append({
            "start": _str_to_timeobj(e["start"]),
            "end": _str_to_timeobj(e["end"]),
            "type": e.get("type", "UNKNOWN")
        })
    _SCHEDULE_CACHE[fname] = lst


def get_label(timestamp, pcap_filename, tz_offset_hours: float = 0.0):
    """
    timestamp: float(UNIX秒) or datetime
    pcap_filename: 実ファイルパス or basename を渡してOK
    tz_offset_hours: pcap 時刻からの補正（例: -3）
    戻り値: ラベル名（該当無しなら "BENIGN"）
    """
    base = os.path.basename(pcap_filename)
    schedules = _SCHEDULE_CACHE.get(base, [])
    # normalize timestamp → datetime
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(float(timestamp))
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise TypeError("timestamp must be float or datetime")

    if tz_offset_hours:
        dt = dt + timedelta(hours=tz_offset_hours)

    t = dt.time()
    # 比較は start <= t < end を採用（end を含めない）
    for e in schedules:
        s = e["start"]
        en = e["end"]
        if s <= en:
            if s <= t < en:
                return e["type"]
        else:
            # 深夜跨ぎ（例 23:00-01:00）の場合
            if t >= s or t < en:
                return e["type"]
    return "BENIGN"


def sanitize_label(label_str):
    return label_str.replace(" ", "_").replace("-", "_").replace("/", "_")
