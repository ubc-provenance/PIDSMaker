"""Generates a small synthetic dataset in the format of LANL's auth.txt and redteam.txt.

Used to test the LANL preprocessing and the network dataset pipeline without the real data
(which must be requested on the LANL website). The data mimics the structure of LANL:
users authenticating from their workstation to servers, computer accounts, self-loops, several
authentication types, and a red team campaign starting at t=150885 (the first red team event of
LANL) that moves laterally from one computer to new ones.
"""

import gzip
import os
import random

FIRST_REDTEAM_TIME = 150885


def generate(out_dir, num_days=15, num_hosts=120, num_users=80, events_per_day=4000, seed=0):
    """Writes auth.txt.gz and redteam.txt.gz in `out_dir`, and returns their paths."""
    rng = random.Random(seed)
    hosts = [f"C{i}" for i in range(1, num_hosts + 1)]
    servers = hosts[:8]
    users = [f"U{i}@DOM1" for i in range(1, num_users + 1)]
    workstation = {u: rng.choice(hosts[8:]) for u in users}
    attacker_host, attacker_user = "C17693", "U748@DOM1"

    events = []
    end = num_days * 86400
    for _ in range(num_days * events_per_day):
        t = rng.randint(1, end)
        kind = rng.random()
        if kind < 0.6:  # user logon from its workstation to a server
            user = rng.choice(users)
            src, dst = workstation[user], rng.choice(servers)
            dst_user = user
            logon = "Network"
        elif kind < 0.85:  # computer account authentication
            src = rng.choice(hosts)
            user = f"{src}$@DOM1"
            dst = rng.choice(servers)
            dst_user = user
            logon = "Network"
        else:  # local logon (self-loop)
            src = dst = rng.choice(hosts)
            user = f"SYSTEM@{src}"
            dst_user = user
            logon = rng.choice(["Service", "Batch", "Interactive"])
        auth_type = rng.choices(["NTLM", "Kerberos", "Negotiate"], weights=[5, 3, 2])[0]
        orientation = rng.choices(["LogOn", "LogOff", "TGS"], weights=[6, 3, 1])[0]
        success = "Success" if rng.random() < 0.98 else "Fail"
        events.append((t, user, dst_user, src, dst, auth_type, logon, orientation, success))

    # Red team: lateral movement from the attacker host to new computers
    redteam = []
    t = FIRST_REDTEAM_TIME
    for target in rng.sample(hosts[8:], 25):
        redteam.append((t, attacker_user, attacker_host, target))
        events.append((t, attacker_user, attacker_user, attacker_host, target, "NTLM", "Network", "LogOn", "Success"))
        t += rng.randint(600, 20000)

    events.sort(key=lambda e: e[0])
    os.makedirs(out_dir, exist_ok=True)
    auth_path = os.path.join(out_dir, "auth.txt.gz")
    redteam_path = os.path.join(out_dir, "redteam.txt.gz")
    with gzip.open(auth_path, "wt") as f:
        for e in events:
            f.write(",".join(map(str, e)) + "\n")
    with gzip.open(redteam_path, "wt") as f:
        for e in redteam:
            f.write(",".join(map(str, e)) + "\n")
    return auth_path, redteam_path


if __name__ == "__main__":
    import sys

    print(generate(sys.argv[1]))
