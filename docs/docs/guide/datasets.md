---
title: Datasets
description: The provenance datasets supported by PIDSMaker (DARPA TC E3/E5, OpTC, ATLASv2, CARBANAKv2), their attack scenarios and ground-truth labels.
---

# Datasets

PIDSMaker supports the public provenance datasets most used in APT detection research, preprocessed into Postgres databases that you can [download and load in minutes](../getting-started/installation.md#1-download-the-datasets). This page describes each dataset, its attack scenarios and its ground truth.
{ .lead }

## Overview

Nodes and events are counted in the database, before graph construction. Size is the size of the database once loaded.

| Dataset | OS | Attacks | Nodes | Events | Size (GB) |
|---|---|--:|--:|--:|--:|
| CADETS_E3 | FreeBSD | 3 | 2.68M | 36.5M | 10 |
| THEIA_E3 | Linux | 2 | 1.26M | 44.4M | 12 |
| CLEARSCOPE_E3 | Android | 1 | 369K | 18.3M | 4.8 |
| FIVEDIRECTIONS_E3 | Windows | 2 | 1.38M | 78.9M | 22 |
| TRACE_E3 | Linux | 3 | 44.5M | 284M | 100 |
| CADETS_E5 | FreeBSD | 2 | 15.2M | 1.08B | 276 |
| THEIA_E5 | Linux | 1 | 2.29M | 141M | 36 |
| CLEARSCOPE_E5 | Android | 2 | 501K | 199M | 49 |
| FIVEDIRECTIONS_E5 | Windows | 4 | 18.8M | 1.09B | 280 |
| TRACE_E5 | Linux | 1 | 72.3M | 2.64B | 710 |
| optc_h201 | Windows | 1 | 5.27M | 38.2M | 9 |
| optc_h501 | Windows | 1 | 3.96M | 28.0M | 6.7 |
| optc_h051 | Windows | 1 | 4.62M | 31.6M | 7.7 |
| ATLASV2_EDR | Windows | 10 | 664K | 2.70M | 1 |
| CARBANAKV2_EDR | Windows + Linux | 1 | 4.15M | 17.1M | 6.6 |

??? info "Share of each node type"

    | Dataset | Processes | Files | Netflows |
    |---|--:|--:|--:|
    | CADETS_E3 | 8.4% | 85.9% | 5.8% |
    | THEIA_E3 | 22.1% | 63.1% | 14.8% |
    | CLEARSCOPE_E3 | 3.4% | 20.8% | 75.8% |
    | FIVEDIRECTIONS_E3 | 37.7% | 53.3% | 9.0% |
    | TRACE_E3 | 88.4% | 2.4% | 9.1% |
    | CADETS_E5 | 43.2% | 55.4% | 1.4% |
    | THEIA_E5 | 55.3% | 43.1% | 1.6% |
    | CLEARSCOPE_E5 | 14.6% | 64.7% | 20.7% |
    | FIVEDIRECTIONS_E5 | 88.5% | 5.3% | 6.2% |
    | TRACE_E5 | 74.5% | 10.1% | 15.4% |
    | optc_h201 | 1.9% | 9.0% | 89.1% |
    | optc_h501 | 1.8% | 8.7% | 89.5% |
    | optc_h051 | 1.1% | 8.3% | 90.7% |
    | ATLASV2_EDR | 10.1% | 85.8% | 4.1% |
    | CARBANAKV2_EDR | 3.2% | 94.5% | 2.3% |

## DARPA TC

The DARPA Transparent Computing program produced benchmark datasets for evaluating provenance-based security systems.

### Engagement 3 (E3), April 2018

[:lucide-external-link: Official E3 description](https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md)


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

### Engagement 5 (E5), May 2019

[:lucide-external-link: Official E5 description](https://github.com/darpa-i2o/Transparent-Computing)

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

## DARPA OpTC

Windows enterprise environment with realistic APT scenarios. [:lucide-external-link: Official repository](https://github.com/FiveDirections/OpTC-data)

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

## ATLASv2

[:lucide-file-text: Paper](https://arxiv.org/abs/2401.01341) · Two Windows 7 hosts (h1, h2) attacked from Kali Linux. The dataset covers four benign days followed by one attack day on which all ten scenarios are executed. The provided data is Carbon Black Cloud EDR telemetry; the original dataset also includes Sysmon, Windows Security Auditing, Firefox, and DNS logs.

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

## CARBANAKv2

[:lucide-file-text: Paper](https://www.ndss-symposium.org/wp-content/uploads/prism2026-12.pdf) · Multi-host testbed comprising four Windows 10 workstations (h1–h4), a CentOS 7 Linux fileserver (fs), and a Windows 10 Server AD domain controller (dc). Benign activity was generated by four graduate students using the machines as their primary workstations. The provided data is Carbon Black Cloud EDR telemetry.

### CARBANAKV2_EDR

| Attack id | Duration | Description |
|---|----------|-------------|
| 0 | 10 days | MITRE Carbanak APT emulation. Attacker compromises h1 (Apr 30) via WScript-based Carbanak implant (`TransBaseOdbcDriver.js`). Pivots to Linux fileserver fs and then Windows AD domain controller dc via lateral movement (May 2). Subsequently spreads to h2, h3, and h4 (May 7–10), establishing persistence across all hosts. |

```shell
python pidsmaker/main.py SYSTEM CARBANAKV2_EDR
```

## PROVENANCE_BENIGN

A Linux audit corpus used only to [pretrain SPIDER](../features/pretrained_encoders.md#pretraining). It has no ground truth, so it can't be used to evaluate detection.

It is built on a Linux machine: auditd records the system calls of 21 scripted workloads (from core utilities, compilers and package managers to web servers, databases, browsers and attack-like commands), the logs are parsed into entities and edges, and `create_database_provenance_benign.py` loads them into a `PROVENANCE_BENIGN` database with the DARPA TC schema.
The collection scripts are on the [`spider` branch](https://github.com/ubc-provenance/PIDSMaker/tree/spider/scripts/provenance_capture), and their [README](https://github.com/ubc-provenance/PIDSMaker/blob/spider/scripts/provenance_capture/README.md) describes each step.

The dates of `PROVENANCE_BENIGN` in `DATASET_DEFAULT_CONFIG` (`pidsmaker/config/config.py`) are those of the corpus used in the paper (2026-03-16, UTC). Set them to the days of your own capture.

## Train, validation and test splits

Each dataset is split by day, as defined by `train_dates`, `val_dates` and `test_dates` in `DATASET_DEFAULT_CONFIG` (`pidsmaker/config/config.py`):

- **Train days**: benign activity only, used to train the model.
- **Validation days**: benign activity only, used to pick the detection threshold.
- **Test days**: benign activity and attacks, used for evaluation.

The events of each day are then cut into time-window graphs during [construction](pipeline.md#the-stages).

## Ground truth

Each dataset comes with the list of the nodes involved in each attack, and PIDSMaker ships several labelings of the DARPA TC attacks (`orthrus`, `reapr`, `threatrace`). See [Ground truth](evaluation.md#ground-truth) for how they are selected and how labels are assigned to nodes, edges and time windows.

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

Then follow the [database creation guide](../getting-started/from-scratch.md) to load your data, and add its ground truth under `Ground_Truth/`.
