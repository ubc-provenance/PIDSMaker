"""Generates a small synthetic dataset in the format of the NetFlow v3 datasets (e.g.
NF-CSE-CIC-IDS2018-v3).

Used to test the NetFlow v3 preprocessing and the CIC_IDS2018 pipeline without the real data. The
flows mimic the structure of CSE-CIC-IDS2018: hosts of the testbed (172.31.0.0/16) query a DNS
server and web servers, a few ICMP, ICMPv6 and IGMP flows, and attacks from an external host on a
train day (SSH brute force on 2018-02-14) and on test days (DDoS on 2018-02-21, lateral movement
on 2018-02-28). Times are between 08:00 and 20:00, local time of the testbed (UTC-4).
"""

import os
import random
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd

# Days with flows (train, val and test dates of CIC_IDS2018)
DAYS = [
    "2018-02-14", "2018-02-15", "2018-02-16", "2018-02-20", "2018-02-21", "2018-02-22",
    "2018-02-23", "2018-02-27", "2018-02-28", "2018-03-01", "2018-03-02",
]  # fmt: skip
LOCAL_TIME = timezone(timedelta(hours=-4))
ATTACKER, VICTIM = "18.219.211.138", "172.31.69.25"
# Attacks: (day, protocol, destination port, number of flows, lateral movement)
ATTACKS = [
    ("2018-02-14", 6, 22, 40, False),
    ("2018-02-21", 17, 80, 60, False),
    ("2018-02-28", 6, 445, 30, True),
]


def _ms(day, rng):
    start = datetime.fromisoformat(day).replace(hour=8, tzinfo=LOCAL_TIME)
    return int(start.timestamp() * 1000) + rng.randint(0, 12 * 3600 * 1000 - 1)


def generate(flows_per_day=120, num_hosts=30, seed=0):
    """DataFrame of the flows, with the NetFlow v3 columns used by the loader and a few others."""
    rng = random.Random(seed)
    hosts = [f"172.31.64.{i}" for i in range(1, num_hosts + 1)]
    dns, web = "172.31.0.2", ["172.31.0.10", "172.31.0.11"]
    external = ["52.94.233.129", "104.16.52.111"]
    rows = []

    def flow(t, src, sport, dst, dport, proto, label=0, attack="Benign"):
        in_bytes, in_pkts = rng.randint(40, 5000), rng.randint(1, 20)
        rows.append(
            {
                "FLOW_START_MILLISECONDS": t,
                "FLOW_END_MILLISECONDS": t + rng.randint(0, 3000),
                "IPV4_SRC_ADDR": src,
                "L4_SRC_PORT": sport,
                "IPV4_DST_ADDR": dst,
                "L4_DST_PORT": dport,
                "PROTOCOL": proto,
                "L7_PROTO": 7.0,
                "IN_BYTES": in_bytes,
                "IN_PKTS": in_pkts,
                "OUT_BYTES": rng.randint(0, 20000),
                "OUT_PKTS": rng.randint(0, 20),
                "TCP_FLAGS": 27,
                "FLOW_DURATION_MILLISECONDS": rng.randint(0, 3000),
                "Label": label,
                "Attack": attack,
            }
        )

    for day in DAYS:
        for _ in range(flows_per_day):
            src, kind = rng.choice(hosts), rng.random()
            sport = rng.randint(49152, 65535)
            if kind < 0.4:
                flow(_ms(day, rng), src, sport, dns, 53, 17)
            elif kind < 0.8:
                flow(_ms(day, rng), src, sport, rng.choice(web), rng.choice([80, 443]), 6)
            elif kind < 0.9:
                # Registered and well-known ports without their own category
                sport, dport = rng.randint(1024, 49151), rng.choice([111, 8081])
                flow(_ms(day, rng), src, sport, rng.choice(external), dport, 6)
            else:
                proto = rng.choice([1, 58, 2])  # ICMP, ICMPv6, IGMP (OTHER)
                flow(_ms(day, rng), src, 0, rng.choice(hosts), 0, proto)
    for day, proto, port, num_flows, lateral in ATTACKS:
        for i in range(num_flows):
            src, dst = (VICTIM, hosts[i % len(hosts)]) if lateral else (ATTACKER, VICTIM)
            sport = rng.randint(49152, 65535)
            flow(_ms(day, rng), src, sport, dst, port, proto, 1, f"attack_{day}")

    df = pd.DataFrame(rows)
    return df.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)


def write(df, path):
    """Writes the flows as the loader reads them, by extension: .csv, .csv.gz, .zip (with the
    features description of the dataset) or .parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as archive:
            features = "Feature,Description\nIN_BYTES,Incoming bytes\n"
            archive.writestr("NetFlow_v3_Features.csv", features)
            archive.writestr("data/NF-CSE-CIC-IDS2018-v3.csv", df.to_csv(index=False))
    else:
        df.to_csv(path, index=False)  # gzipped from the extension
    return path


if __name__ == "__main__":
    import sys

    print(write(generate(), sys.argv[1]))
