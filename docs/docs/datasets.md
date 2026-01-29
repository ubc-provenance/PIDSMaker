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
        "database": "my_database_name",
        "num_node_types": 3,
        "num_edge_types": 10,
        "train_files": ["graph_1", "graph_2", "graph_3"],
        "val_files": ["graph_4"],
        "test_files": ["graph_5", "graph_6"],
        "ground_truth_relative_path": ["MY_DATASET/labels.csv"],
        "attack_to_time_window": [
            ["MY_DATASET/labels.csv", "2024-01-05 10:00:00", "2024-01-05 12:00:00"],
        ],
    },
}
```

Then follow the [database creation guide](create-db-from-scratch.md) to load your data.
