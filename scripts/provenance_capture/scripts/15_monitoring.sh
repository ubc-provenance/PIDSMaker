#!/bin/bash
# ============================================================================
# DOMAIN 15: SYSTEM MONITORING
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "15 — MONITORING"

run "vmstat" vmstat 1 3
run "vmstat -s" vmstat -s
run "vmstat -w" vmstat -w 1 2
run "vmstat -d" vmstat -d 2>/dev/null
run "vmstat -a" vmstat -a 1 2
run "iostat" iostat 1 2 2>/dev/null
run "iostat -x" iostat -x 1 2 2>/dev/null
run "mpstat" mpstat 1 2 2>/dev/null
run "mpstat -P ALL" mpstat -P ALL 1 2 2>/dev/null
run "sar -u" sar -u 1 2 2>/dev/null
run "sar -r" sar -r 1 2 2>/dev/null
run "sar -b" sar -b 1 2 2>/dev/null
run "dstat" dstat -cdngy 1 3 2>/dev/null
run "dstat -a" dstat -a 1 3 2>/dev/null
run "free -h" free -h
run "free -m" free -m
run "uptime" uptime
run "uname -a" uname -a
run "uname -r" uname -r
run "uname -m" uname -m
run "uname -s" uname -s
run "uname -v" uname -v
run "hostnamectl" hostnamectl 2>/dev/null
run "timedatectl" timedatectl 2>/dev/null
run "lscpu" lscpu 2>/dev/null
run "lsmem" lsmem 2>/dev/null
run "lsblk" lsblk 2>/dev/null
run "lsblk -f" lsblk -f 2>/dev/null
run "lsblk -o" lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null
run "lsusb" lsusb 2>/dev/null
run "lspci" lspci 2>/dev/null
run "lshw -short" lshw -short 2>/dev/null | head -20
run "dmesg" dmesg 2>/dev/null | tail -20
run "dmesg -T" dmesg -T 2>/dev/null | tail -10
run "dmesg --level=err" dmesg --level=err 2>/dev/null | tail -5
run "sysctl -a" sysctl -a 2>/dev/null | head -30
run "sysctl kernel" sysctl kernel.hostname kernel.osrelease 2>/dev/null
run "env" env | head -20
run "printenv" printenv | head -20
run "locale" locale
run "locale -a" locale -a | head -10
run "date" date
run "date -R" date -R
run "date +%s" date +%s
run "date -u" date -u
run "cal" cal
run "nproc" nproc
run "getconf" getconf -a 2>/dev/null | head -15
run "arch" arch

domain_end
