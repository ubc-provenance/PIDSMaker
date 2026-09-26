# Provenance Data Collection Guide

## Overview

This guide covers the full pipeline for collecting benign system provenance data using Linux auditd, parsing it into a provenance graph (entities + edges), and extracting 1-hop neighborhoods for model training.

```
auditd (kernel) → audit.log → audit_to_provenance.py → entities.tsv + edges.tsv → build_neighborhoods.py → neighborhoods.txt
```

---

## 1. Install and Configure auditd

### Install
```bash
sudo apt-get install -y auditd audispd-plugins
```

### Deploy provenance rules
```bash
sudo cp provenance.rules /etc/audit/rules.d/provenance.rules
sudo augenrules --load
sudo systemctl restart auditd
```

### Verify rules loaded
```bash
sudo auditctl -l | grep prov | wc -l
# Expected: ~30+ rules
sudo auditctl -l | grep prov
```

### Increase log capacity (prevent rotation during collection)
```bash
sudo sed -i 's/^max_log_file .*/max_log_file = 500/' /etc/audit/auditd.conf
sudo sed -i 's/^num_logs .*/num_logs = 10/' /etc/audit/auditd.conf
sudo systemctl restart auditd
```

---

## 2. Auditd Rules Reference (provenance.rules)

The rules capture the following syscalls, each mapped to a provenance event type:

| Audit key | Syscall(s) | Provenance event | Entity types |
|-----------|-----------|------------------|-------------|
| `prov_exec` | execve | EVENT_EXECUTE | PROC → PROC, FILE → PROC |
| `prov_clone` | clone, clone3, fork, vfork | EVENT_CLONE | PROC → PROC |
| `prov_connect` | connect | EVENT_CONNECT | PROC → SOCK |
| `prov_accept` | accept (syscall 43) | EVENT_CONNECT | SOCK → PROC |
| `prov_sendto` | sendto | EVENT_SENDTO | PROC → SOCK |
| `prov_sendmsg` | sendmsg | EVENT_SENDMSG | PROC → SOCK |
| `prov_recvfrom` | recvfrom | EVENT_RECVFROM | SOCK → PROC |
| `prov_recvmsg` | recvmsg | EVENT_RECVMSG | SOCK → PROC |
| `prov_openat` | openat, open | EVENT_READ / EVENT_WRITE | FILE ↔ PROC |
| `prov_create` | mkdir, mkdirat | EVENT_WRITE | PROC → FILE |
| `prov_delete` | unlink, unlinkat | EVENT_WRITE | PROC → FILE |
| `prov_rename` | rename, renameat, renameat2 | EVENT_WRITE | PROC → FILE |
| `prov_chmod` | chmod, fchmod, fchmodat | EVENT_WRITE | PROC → FILE |
| `prov_chown` | chown, fchown, lchown, fchownat | EVENT_WRITE | PROC → FILE |
| `prov_ptrace` | ptrace | EVENT_OPEN | PROC → PROC |

### Key design decisions

- **No file watches (`-w`)**: They don't work inside Docker containers (overlay fs). We use `openat` syscall monitoring instead, which works across all filesystems.
- **recvfrom/recvmsg enabled**: Needed for full bidirectional network provenance.
- **`accept` uses syscall number 43**: The name `accept`/`accept4` is not recognized by all auditd versions.
- **pipe/dup excluded**: They add no semantic value for entity profiling (tracked internally but not captured).


---

## 3. Generate Workload Activity

### Inside a Docker container or on the host directly (dangerous)

```bash
# WARNING: scripts modify system state (create/delete users, install packages)
# Only run on a test machine or VM
cd scripts
sudo bash run_all.sh 2>&1 | tee workload.log
```

### Workload scripts summary (21 domains, ~2131 commands)

| Script | Commands | Content |
|--------|----------|---------|
| 00_install.sh | 27 | apt, pip, npm, gem, cargo |
| 01_core_utilities.sh | 485 | ls, cat, cp, mv, rm, chmod, find, grep, sed, awk, sort, tar, etc. |
| 02_process_mgmt.sh | 211 | ps, /proc reads, signals, strace, lsof |
| 03_user_perms.sh | 121 | useradd, passwd, groups, su, ACLs |
| 04_network.sh | 259 | dig, curl, wget, ssh tools, nmap, ss, netstat |
| 05_compilers.sh | 186 | gcc, clang, g++, Go, Rust, Java, Make |
| 06_python.sh | 93 | File I/O, network, subprocess, crypto scripts |
| 07_packages.sh | 59 | apt, dpkg, pip, npm, gem |
| 08_database.sh | 57 | SQLite, Redis, PostgreSQL/MySQL clients |
| 09_web_servers.sh | 44 | Python/Node/PHP/Ruby HTTP servers, nginx, Apache |
| 10_crypto.sh | 55 | OpenSSL keys, certs, encryption, GPG |
| 11_scripting.sh | 88 | Perl, Ruby, Lua, PHP, Node.js, awk, Bash |
| 12_git.sh | 53 | Full git workflow |
| 13_text.sh | 51 | jq, xmlstarlet, text processing |
| 14_media.sh | 32 | ImageMagick, FFmpeg |
| 15_monitoring.sh | 49 | vmstat, iostat, sysctl, hardware info |
| 16_disk.sh | 38 | df, mount, loopback, rsync |
| 17_cron.sh | 15 | crontab, at, systemd timers |
| 18_logging.sh | 31 | logger, journalctl, dmesg |
| 19_systemd.sh | 25 | systemctl, service creation |
| 20_attack.sh | 152 | Recon, privesc, reverse shells, LOLBins, exfil |
| 21_browser.sh | ~160 | Headless Chrome/Firefox, curl browsing, DNS |

---

## 4. Parse Audit Logs

### Clear previous data (optional)
```bash
rm -rf ~/provenance_data/
```

### Parse current log
```bash
sudo bash -c 'cat /var/log/audit/audit.log' | \
    python3 audit_to_provenance.py --input - --output ~/provenance_data/
```

### Parse all rotated logs together
```bash
sudo bash -c 'cat /var/log/audit/audit.log.* /var/log/audit/audit.log 2>/dev/null' | \
    python3 audit_to_provenance.py --input - --output ~/provenance_data/
```

### Expected output
```
Processed 3,473,122 audit lines

Exported 38683 entities, 243610 edges to provenance_data/
  Entities: {'PROC': 10185, 'FILE': 18935, 'SOCK': 9563}
  Events:   {'EVENT_READ': 153481, 'EVENT_WRITE': 51209, 'EVENT_CLONE': 16952,
             'EVENT_EXECUTE': 12005, 'EVENT_RECVMSG': 3672, 'EVENT_CONNECT': 2970,
             'EVENT_SENDMSG': 2467, 'EVENT_RECVFROM': 853, 'EVENT_SENDTO': 1}
```

### Output files
```
provenance_data/
├── entities.tsv    # entity_id \t entity_type \t entity_text
└── edges.tsv       # src_id \t event_type \t dst_id \t timestamp
```

---

## 5. Build Neighborhoods

```bash
python3 build_neighborhoods.py --input ~/provenance_data/ --output ~/neighborhoods.txt
```

### With walk triplets for training
```bash
python3 build_neighborhoods.py --input ~/provenance_data/ --output ~/neighborhoods.txt --walks ~/walks.txt
```

### Output format

**PROC entities are unique per instance** (each process invocation is separate):
```
--- [proc_42] [PROC] /usr/sbin/nginx -g daemon off ---
  OUTGOING (3):
    --EVENT_CONNECT--> [SOCK] 0.0.0.0 0 93.184.216.34 443
    --EVENT_WRITE--> [FILE] /var/log/nginx/access.log
    --EVENT_CLONE--> [PROC] nginx: worker process
  INCOMING (2):
    <--EVENT_EXECUTE-- [FILE] /usr/sbin/nginx
    <--EVENT_EXECUTE-- [PROC] /bin/bash
```

**FILE and SOCK entities are deduplicated by label** (same path = same entity):
```
--- [FILE] /etc/passwd ---
  INCOMING (15):
    <--EVENT_READ-- [PROC] cat /etc/passwd  x8
    <--EVENT_READ-- [PROC] grep root /etc/passwd  x4
```

---

## Load the data in a postgres database + generate a dump

Use this script: `create_database_provenance_benign.py`


## 6. Parser Noise Filtering (v3)

The v3 parser aggressively filters paths that appear in every process's neighborhood and add zero semantic value:

| Filtered category | Example paths | Why filtered |
|---|---|---|
| Shared libraries | `/usr/lib/x86_64-linux-gnu/libc.so` | Every process loads these identically |
| Locale/timezone | `/usr/lib/locale/*`, `/usr/share/zoneinfo/*` | Every process reads these |
| Linker cache | `/etc/ld.so.cache`, `/etc/ld.so.conf.d/*` | Every process reads at startup |
| SSL certificates | `/etc/ssl/certs/*`, `/usr/share/ca-certificates/*` | Every HTTPS client reads these |
| Python/Perl/Ruby internals | `/usr/lib/python3/*`, `/usr/share/perl/*` | Runtime internals, not application logic |
| NSS config | `/etc/nsswitch.conf`, `/etc/gai.conf` | Every process reads via libc |
| Common files | `/etc/passwd`, `/etc/group`, `/etc/hosts` | Every process reads via NSS |
| Kernel/virtual fs | `/proc/*`, `/sys/*`, `/dev/pts/*` | Not real files |
| Terminfo | `/usr/share/terminfo/*` | Every terminal program |
| Docs/man | `/usr/share/doc/*`, `/usr/share/man/*` | No runtime relevance |

### Additional v3 features

- **Process name fix**: When EXECVE args start with flags (e.g., `--install ...`), the exe basename is prepended
- **Socket deduplication**: Same `(remote_ip, remote_port)` → same SOCK entity
- **sendmsg/recvmsg fallback**: When no SOCKADDR record exists, uses the process's last known socket connection
- **Filtered path stats**: Reports what was filtered and counts, for tuning

---

## 7. Troubleshooting

### "you do not exist in the passwd database"
The workload scripts modified `/etc/passwd`. Restore from backup:
```bash
# If sudo doesn't work, use Docker's root privileges:
docker run --rm -v /etc:/host_etc ubuntu cp /host_etc/passwd- /host_etc/passwd
# Then verify:
sudo whoami
```

### Audit log rotation during collection
```bash
# Increase max log size before running workload
sudo sed -i 's/^max_log_file .*/max_log_file = 500/' /etc/audit/auditd.conf
sudo systemctl restart auditd

# Or parse all rotated logs together
sudo bash -c 'cat /var/log/audit/audit.log.* /var/log/audit/audit.log 2>/dev/null' | \
    python3 audit_to_provenance.py --input - --output ~/provenance_data/
```

### Permission denied on output directory
```bash
# Parser runs as your user, not root
sudo chown -R $USER:$USER ~/provenance_data/
# Or use your home directory
python3 audit_to_provenance.py --input - --output ~/provenance_data/
```

### Missing event types (only EVENT_EXECUTE and EVENT_CLONE)
Check that:
1. Rules are loaded: `sudo auditctl -l | grep prov_openat`
2. recvfrom/recvmsg are not commented out in the rules file
3. For Docker containers: use `openat` syscall rules, not file watches (`-w`)
4. Clear old log and re-run workload after updating rules

### Check what auditd is capturing
```bash
# Count events by key
sudo ausearch -k prov_exec --just-one 2>/dev/null && echo "exec: OK"
sudo ausearch -k prov_connect --just-one 2>/dev/null && echo "connect: OK"
sudo ausearch -k prov_openat --just-one 2>/dev/null && echo "openat: OK"

# Count events per key
sudo aureport -k | tail -20
```

---

## 8. File Inventory

| File | Purpose |
|------|---------|
| `provenance.rules` | Auditd rules for provenance capture |
| `audit_to_provenance.py` | Parse audit.log → entities.tsv + edges.tsv |
| `build_neighborhoods.py` | Build 1-hop neighborhoods from graph |
| `prov_scripts/prov_framework.sh` | Shared runner framework for workload scripts |
| `prov_scripts/run_all.sh` | Master orchestrator for all workload domains |
| `prov_scripts/00_install.sh` – `21_browser.sh` | Domain-specific workload generators |
