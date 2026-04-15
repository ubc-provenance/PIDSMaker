# Datasets

PIDSMaker supports several public datasets commonly used in APT detection research. This page describes each dataset and its attack scenarios.

## Overview

| Dataset | OS | Attacks | Size (GB) |
|---------|------|---------|-----------|
| CADETS_E3 | FreeBSD | 3 | 10 |
| THEIA_E3 | Linux | 2 | 12 |
| CLEARSCOPE_E3 | Android | 1 | 4.8 |
| FIVEDIRECTIONS_E3 | Linux | 2 | 22 |
| TRACE_E3 | Linux | 3 | 100 |
| CADETS_E5 | FreeBSD | 2 | 276 |
| THEIA_E5 | Linux | 1 | 36 |
| CLEARSCOPE_E5 | Android | 2 | 49 |
| FIVEDIRECTIONS_E5 | Linux | 4 | 280 |
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

### [Engagement 5 (E5) - May 2019](https://github.com/darpa-i2o/Transparent-Computing)

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

!!! note
    TODO: add descriptions for CADETS_E5, FIVED and TRACE datasets.

## [ATLASv2](https://arxiv.org/abs/2401.01341)

Two Windows 7 hosts (h1, h2) instrumented with Carbon Black Cloud EDR, Sysmon, and Windows Security Auditing, attacked from Kali Linux. The dataset covers four benign days followed by one attack day on which all ten scenarios are executed.

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

Multi-host testbed comprising four Windows 10 workstations (h1–h4), a CentOS 7 Linux fileserver (fs), and a Windows 10 Server AD domain controller (dc), all instrumented with Carbon Black Cloud XDR and Wireshark. Benign activity was generated by four graduate students using the machines as their primary workstations.

### CARBANAKV2_EDR

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 10 days | MITRE Carbanak APT emulation. Attacker compromises h1 (Apr 30) via WScript-based Carbanak implant (`TransBaseOdbcDriver.js`). Pivots to Linux fileserver fs and then Windows AD domain controller dc via lateral movement (May 2). Subsequently spreads to h2, h3, and h4 (May 7–10), establishing persistence across all hosts. |

```shell
python pidsmaker/main.py SYSTEM CARBANAKV2_EDR
```

## Data structure

### Graph partitioning

Each dataset is partitioned into daily graphs, split into:

- **Train graphs**: Normal activity for model training
- **Validation graphs**: Normal activity for threshold calibration
- **Test graphs**: Contains both normal activity and attacks

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
