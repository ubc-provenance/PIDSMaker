# Datasets

PIDSMaker supports several public datasets commonly used in APT detection research. This page describes each dataset and its attack scenarios.

## Overview

| Dataset | OS | Attacks | Size (GB) |
|---------|------|---------|-----------|
| CADETS_E3 | FreeBSD | 3 | 10 |
| THEIA_E3 | Linux | 2 | 12 |
| CLEARSCOPE_E3 | Android | 1 | 4.8 |
| FIVEDIRECTIONS_E3 | Windows | 2 | 22 |
| TRACE_E3 | Linux | 3 | 100 |
| CADETS_E5 | FreeBSD | 2 | 276 |
| THEIA_E5 | Linux | 1 | 36 |
| CLEARSCOPE_E5 | Android | 2 | 49 |
| FIVEDIRECTIONS_E5 | Windows | 4 | 280 |
| TRACE_E5 | Linux | 1 | 710 |
| optc_h201 | Windows | 1 | 9 |
| optc_h501 | Windows | 1 | 6.7 |
| optc_h051 | Windows | 1 | 7.7 |
| ATLASV2_EDR | Windows | 10 | 1 |
| CARBANAKV2_EDR | Windows + Linux | 1 | 6.6 |


## DARPA TC

The DARPA Transparent Computing program produced benchmark datasets for evaluating provenance-based security systems.

### [Engagement 3 (E3) - April 2018](https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md)


#### CADETS_E3

FreeBSD host with Nginx server exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 49 min | Nginx exploited to deploy Drakon loader with root escalation. Netrecon executed after C2 connection, followed by failed `libdrakon` injection into `sshd`. Host crashed with kernel panic. |
| 1 | 40 min | Nginx re-exploited to deploy Drakon and MicroAPT implants under random names (`tmux`, `minions`, `sendmail`). Privilege escalation failed; MicroAPT ran unprivileged for port scanning. |
| 2 | 13 min | Nginx re-exploited to deploy new Drakon implant with root privileges. Multiple failed `sshd` injection attempts using renamed `libdrakon` copies. |

```shell
python pidsmaker/main.py SYSTEM CADETS_E3
```

#### THEIA_E3

Ubuntu host with Firefox exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 50 min | Malicious Firefox extension dropped Drakon implant. MicroAPT staged under `/var/log/mail`, connected to C2 for control and network scanning. |
| 1 | 30 min | Firefox exploited to drop Drakon implant as `/home/admin/clean` with root privileges, then copied as `profile`. Both connected to C2 server. |

```shell
python pidsmaker/main.py SYSTEM THEIA_E3
```

#### CLEARSCOPE_E3

Android device with Firefox exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 54 min | Firefox exploited via malicious website. Drakon implant installed and elevated, but module loading failed. Persistent C2 connection maintained. |

```shell
python pidsmaker/main.py SYSTEM CLEARSCOPE_E3
```

#### TRACE_E3

Ubuntu host with Firefox and Pine e-mail client exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 1h25 | Firefox exploited through a malicious ad on `www.allstate.com` after several crashes, loading Drakon in memory. A Drakon implant was written to `/home/admin/cache` and elevated to a root C2 shell; `libdrakon` was staged under `/var/log/xtmp` for later use. Connection lost when the operator console crashed. |
| 1 | 15 min | Phishing e-mail exploiting a backdoored Pine client that auto-runs attachments. A first `tcexec` executable failed on a missing library, so MicroAPT was re-sent as `tcexec` and ran when the e-mail was opened. It connected to C2 and port-scanned the network; attempts to spawn a shell failed. |
| 2 | 12 min | Drakon implant dropped to `/tmp/ztmp` via the malicious password-manager Firefox extension and run with user privileges (labelled *Pine Backdoor w/ Drakon Dropper* in the DARPA report). Injection into `sshd` and privilege escalation failed, so MicroAPT was executed from disk: it connected to C2, port-scanned the network (no open ports found) and ran netrecon. |

```shell
python pidsmaker/main.py SYSTEM TRACE_E3
```

#### FIVEDIRECTIONS_E3

Windows 10 host with Firefox and Excel macro exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 42 min | Firefox exploited by browsing to `www.cnpc.com.cn` after repeated crashes, loading Drakon in memory with a C2 connection. Netrecon module run and its output exfiltrated, then several documents (`.rtf`, `.docx`, `.xlsx`) read and exfiltrated from the user's documents directory. Access lost when the netrecon UDP exfiltration failed. |
| 1 | 37 min | Phishing e-mail with a `BoviaBenefitsOE.xlsm` attachment carrying an encoded PowerShell command. The macro did not fire automatically, so the user ran the command manually: it downloaded `update.ps1` from `208.75.117.5` and opened a powercat reverse shell, which the attacker used to read the hosts file and personal files. |

```shell
python pidsmaker/main.py SYSTEM FIVEDIRECTIONS_E3
```

### [Engagement 5 (E5) - May 2019](https://github.com/darpa-i2o/Transparent-Computing)

#### CADETS_E5

FreeBSD host with Nginx server exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 41 min | Nginx exploited with a malformed HTTP POST on the second attempt, loading Drakon APT in memory with a C2 connection to `4.21.51.250:80`. The attacker retrieved the hostname and username before the connection was unexpectedly lost. |
| 1 | 5h18 | Nginx re-exploited on both CADETS hosts. A first attempt called back to the wrong C2 listener, so C2 was moved to a pivot host (`128.55.12.233:80`). Drakon ran `hostname` and `whoami` and exfiltrated `/etc/passwd` from both hosts; the C2 connections were left open for the rest of the day. |

```shell
python pidsmaker/main.py SYSTEM CADETS_E5
```

#### THEIA_E5

Ubuntu host with Firefox exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 19 min | Firefox exploited via malicious website. Root gained with BinFmt-Elevate, Drakon shellcode injected into `sshd`, persistence file created, C2 access maintained. |

```shell
python pidsmaker/main.py SYSTEM THEIA_E5
```

#### CLEARSCOPE_E5

Android device with APK-based attacks.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 41 min | Malicious `appstarter` APK loaded MicroAPT. Elevate driver installed for privilege escalation. Sensitive databases exfiltrated (calllog, calendar, SMS) and screenshot captured. |
| 1 | 8 min | MicroAPT deployed directly via adb shell after APK dropper failed. Privilege escalation via BinFmt Elevate driver, then file exfiltration. |

```shell
python pidsmaker/main.py SYSTEM CLEARSCOPE_E5
```

#### TRACE_E5

Ubuntu host with Firefox exploitation. A second attempt on May 17 with the Azazel `LD_PRELOAD` rootkit never established C2 and is not labelled.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 1h28 | Firefox exploited via the hijacked `www.yale.edu` website, Drakon APT connected to C2. Root gained with the BinFmt-Elevate driver (installed beforehand as benign activity), then Drakon injected into `sshd` with the new ptrace-based `inject2` technique, opening further C2 connections used to read `/etc/passwd` and `/etc/shadow`. |

```shell
python pidsmaker/main.py SYSTEM TRACE_E5
```

#### FIVEDIRECTIONS_E5

Windows 10 hosts with Firefox exploitation.

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 21 min | Firefox backdoor used to download `ctfhost2.exe` through the Background Intelligent Transfer Service (BITS). MicroAPT executed and connected to C2 at `113.165.213.253:80`. Privilege escalation with several drivers failed because driver signature enforcement was re-enabled after a reboot. |
| 1 | 32 min | Firefox exploited via the hijacked `www.usdoj.gov` website, loading Drakon APT in memory. Privileges escalated to SYSTEM with the elevate driver, then the Copykatz module (Mimikatz) harvested cached credentials alongside `whoami`, `hostname` and sysinfo recon. The WMI step failed due to a target configuration issue. |
| 2 | 11 min | Firefox backdoor triggered over DNS to `xx--ls8h.com`; Drakon APT downloaded, executed and connected to C2. Privileges escalated to SYSTEM with the File System Filter elevate driver (triggered by opening a non-existent file), then host, network and `passwd` files exfiltrated. |
| 3 | 6 min | Follow-up to the Application Verifier persistence set up on May 16 (an unused date), which injects `drakon.dll` into every new Firefox process. A C2 connection from a fresh Firefox launch was used to escalate to SYSTEM with the File System Filter driver and exfiltrate the hosts file. |

```shell
python pidsmaker/main.py SYSTEM FIVEDIRECTIONS_E5
```

## [DARPA OpTC](https://github.com/FiveDirections/OpTC-data)

Windows enterprise environment with realistic APT scenarios.

### optc_h201

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 1h58 | PowerShell Empire stager executed with elevated access. Mimikatz used for credential theft, registry persistence set, recon performed, then pivoted to other hosts via WMI. |

```shell
python pidsmaker/main.py SYSTEM optc_h201
```

### optc_h501

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 5h01 | Phishing email launched PowerShell Empire stager. Escalated via DeathStar, WMI persistence established, RDP tunneling and file exfiltration performed, then pivoted to other hosts. |

```shell
python pidsmaker/main.py SYSTEM optc_h501
```

### optc_h051

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 3h56 | Malicious Notepad++ update installed Meterpreter. Escalated to SYSTEM, migrated into LSASS for Mimikatz credential theft, established persistence, timestomped files, added admin account for RDP. |

```shell
python pidsmaker/main.py SYSTEM optc_h051
```

## [ATLASv2](https://arxiv.org/abs/2401.01341)

Two Windows 7 hosts (h1, h2) attacked from Kali Linux. The dataset covers four benign days followed by one attack day on which all ten scenarios are executed. The provided data is Carbon Black Cloud EDR telemetry; the original dataset also includes Sysmon, Windows Security Auditing, Firefox, and DNS logs.

### ATLASV2_EDR

| Attack id | Duration | Description |
|---|----------|-------------|
| s1 | 40 min | CVE-2015-5122 Adobe Flash exploit via phishing email on h1. Meterpreter shell obtained, payload.exe dropped, PDFs exfiltrated over HTTPS. |
| s2 | 35 min | CVE-2015-3105 Adobe Flash exploit via phishing email on h1. Meterpreter shell, payload drop, and HTTPS PDF exfiltration. |
| s3 | 45 min | CVE-2017-11882 Microsoft Word memory corruption exploit via malicious attachment on h1. Meterpreter obtained, PDFs exfiltrated. |
| s4 | 44 min | CVE-2017-0199 Microsoft Word OLE2 link exploit via malicious document on h1. Meterpreter shell, payload drop, HTTPS exfiltration. |
| m1 | 1h50 | CVE-2015-5122 Adobe Flash exploit on h1 via phishing. h1's SimpleHTTP server poisoned to deliver payload to h2. PDFs exfiltrated from both hosts. |
| m2 | 30 min | CVE-2015-5119 Adobe Flash exploit on h1 with lateral movement to h2 via SimpleHTTP server poisoning. PDFs exfiltrated from both hosts. |
| m3 | 34 min | CVE-2015-3105 Adobe Flash exploit on h1 with lateral movement to h2 via poisoned SimpleHTTP server. PDFs exfiltrated. |
| m4 | 33 min | CVE-2018-8174 Microsoft Word VBScript engine exploit on h1 with lateral movement to h2 via HTTP server. PDFs exfiltrated from both hosts. |
| m5 | 30 min | CVE-2017-0199 Microsoft Word OLE2 link exploit on h1 with lateral movement to h2. PDFs exfiltrated from both hosts. |
| m6 | 33 min | CVE-2017-11882 Microsoft Word memory corruption exploit on h1 with lateral movement to h2 via poisoned HTTP server. PDFs exfiltrated. |

```shell
python pidsmaker/main.py SYSTEM ATLASV2_EDR
```

## [CARBANAKv2](https://www.ndss-symposium.org/wp-content/uploads/prism2026-12.pdf)

Multi-host testbed comprising four Windows 10 workstations (h1–h4), a CentOS 7 Linux fileserver (fs), and a Windows 10 Server AD domain controller (dc). Benign activity was generated by four graduate students using the machines as their primary workstations. The provided data is Carbon Black Cloud EDR telemetry.

### CARBANAKV2_EDR

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 10 days | MITRE Carbanak APT emulation. Attacker compromises h1 (Apr 30) via WScript-based Carbanak implant (`TransBaseOdbcDriver.js`). Pivots to Linux fileserver fs and then Windows AD domain controller dc via lateral movement (May 2). Subsequently spreads to h2, h3, and h4 (May 7–10), establishing persistence across all hosts. |

```shell
python pidsmaker/main.py SYSTEM CARBANAKV2_EDR
```

## PROVENANCE_BENIGN

A Linux audit corpus used only to [pretrain SPIDER](features/pretrained_encoders.md#pretraining). It has no ground truth, so it can't be used to evaluate detection.

It is built on a Linux machine: auditd records the system calls of 21 scripted workloads (from core utilities, compilers and package managers to web servers, databases, browsers and attack-like commands), the logs are parsed into entities and edges, and `create_database_provenance_benign.py` loads them into a `PROVENANCE_BENIGN` database with the DARPA TC schema.
The collection scripts are on the [`spider` branch](https://github.com/ubc-provenance/PIDSMaker/tree/spider/scripts/provenance_capture), and their [README](https://github.com/ubc-provenance/PIDSMaker/blob/spider/scripts/provenance_capture/README.md) describes each step.

The dates of `PROVENANCE_BENIGN` in `DATASET_DEFAULT_CONFIG` (`pidsmaker/config/config.py`) are those of the corpus used in the paper (2026-03-16, UTC). Set them to the days of your own capture.

## Data structure

### Graph partitioning

Each dataset is partitioned into daily graphs, split into:

- **Train graphs**: Normal activity for model training
- **Validation graphs**: Normal activity for threshold calibration
- **Test graphs**: Contains both normal activity and attacks

## Ground truth

Evaluation compares a system's predictions against a **ground truth**: the set of
node UUIDs considered malicious for each attack. PIDSMaker ships several ground-truth
sources, since the labeling of the DARPA TC attacks differs between prior works, and
the choice of source materially affects the reported metrics. The active source is
selected with `evaluation.ground_truth_version` (default `orthrus`):

```shell
# on the CLI
python pidsmaker/main.py SYSTEM CADETS_E3 --evaluation.ground_truth_version threatrace

# or in a YAML config
evaluation:
  ground_truth_version: threatrace
```

The ground-truth files live under `Ground_Truth/<version>/<dataset>/`, and the
selected version changes only which labels evaluation reads — it does not change
training or the graphs, so switching it re-evaluates an already-trained run.

| Version | Source | Format | Coverage |
|---|---|---|---|
| `orthrus` *(default)* | Labeling used by [Orthrus](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf) | One CSV per attack scenario, rows of `uuid, {labels}, count` | All datasets |
| `reapr` | Alternative labeling for the DARPA TC E3 datasets, in the same per-attack CSV format | Same as `orthrus` | E3: CADETS, THEIA, TRACE, FIVEDIRECTIONS |
| `threatrace` | Ground truth published by [ThreaTrace](https://arxiv.org/pdf/2111.04333) | One flat UUID-per-line `.txt` per dataset, collapsed to a single window over the test period | E3: CADETS, THEIA, TRACE, FIVEDIRECTIONS |

Selecting a source with no file for the chosen dataset fails fast with a clear
error. UUIDs listed in a ground truth but absent from the constructed graph are
skipped (and counted in the logs). Mimicry-attack ground-truth augmentation applies
only to the per-attack sources (`orthrus`/`reapr`), not to `threatrace`.

See [`Ground_Truth/threatrace/README.md`](https://github.com/ubc-provenance/PIDSMaker/blob/main/Ground_Truth/threatrace/README.md)
for the per-dataset ThreaTrace node counts and usage notes.

## Adding custom datasets

To add a new dataset, define its configuration in `pidsmaker/config/config.py`:

```python
DATASET_DEFAULT_CONFIG = {
    "MY_DATASET": {
        "raw_dir": "",
        "database": "my_database_name",
        "database_all_file": "my_database_name",
        "num_node_types": 3, # Number of node types in the dataset __format__ (i.e., in pidsmaker/utils/dataset_utils.py)
        "num_edge_types": 10, # Number of edge types in the dataset __format__ (i.e., in pidsmaker/utils/dataset_utils.py)
        "start_date": "2018-04-02", # Start date (Inclusive)
        "end_date": "2018-04-14", # End date (Exclusive)
        "train_dates": [
            # Dates/graphs used for training (i.e., benign activity)
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "val_dates": [
            # Dates/graphs used for validation/threshold calibration (i.e., benign activity)
            "2018-04-10"
        ],
        "test_dates": [
            # Dates/graphs used for testing (i.e., contains both benign and attack activity)
            "2018-04-06",
            "2018-04-11",
            "2018-04-12",
            "2018-04-13"
        ],
        "unused_dates": [
            # Any unused dates/graphs that should be ignored
             "2018-04-14"
        ],
        "ground_truth_relative_path": ["MY_DATASET/labels.csv"],
        "attack_to_time_window": [
            ["MY_DATASET/labels.csv", "2018-04-11 10:00:00", "2018-04-12 12:00:00"],
        ],
    },
}
```

Then follow the [database creation guide](create-db-from-scratch.md) to load your data.
