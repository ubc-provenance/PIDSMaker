#!/bin/bash
# ============================================================================
# DOMAIN 02: PROCESS MANAGEMENT & SIGNALS
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "02 — PROCESS MANAGEMENT"

S=$(generate_sample_files)

# ============================================================================
section "ps — process listing"
# ============================================================================
run "ps" ps
run "ps aux" ps aux
run "ps auxf" ps auxf
run "ps -ef" ps -ef
run "ps -eF" ps -eF
run "ps -ely" ps -ely
run "ps -eo pid,ppid,uid,cmd" ps -eo pid,ppid,uid,cmd --sort=-pid | head -20
run "ps -eo pid,ppid,%mem,%cpu" ps -eo pid,ppid,%mem,%cpu,cmd --sort=-%mem | head -10
run "ps -eo pid,stat" ps -eo pid,stat,cmd | head -20
run "ps -eo user,pid,vsz,rss" ps -eo user,pid,vsz,rss,cmd --sort=-rss | head -10
run "ps -p 1" ps -p 1 -o pid,ppid,cmd
run "ps -p 1 -f" ps -p 1 -f
run "ps -C bash" ps -C bash -o pid,ppid,cmd 2>/dev/null || true
run "ps -u root" ps -u root -o pid,cmd | head -10
run "ps -U root" ps -U root | head -10
run "ps --forest" ps --forest -eo pid,ppid,cmd | head -20
run "ps -t" ps -t pts/0 2>/dev/null || true
run "ps -L threads" ps -eLf | head -10
run "ps -o lstart" ps -eo pid,lstart,cmd | head -5
run "ps -o etime" ps -eo pid,etime,cmd | head -5
run "ps -o nice" ps -eo pid,ni,cmd | head -10
run "ps -o wchan" ps -eo pid,wchan,cmd | head -10
run "ps ww" ps auxww | head -5

# ============================================================================
section "pgrep / pkill / pidof"
# ============================================================================
run "pgrep bash" pgrep -la bash || true
run "pgrep -f" pgrep -fa bash || true
run "pgrep -u root" pgrep -u root | head -10
run "pgrep -c" pgrep -c bash || true
run "pgrep -l" pgrep -l bash || true
run "pgrep -n newest" pgrep -n bash || true
run "pgrep -o oldest" pgrep -o bash || true
run "pgrep -P ppid" pgrep -P 1 | head -10
run "pgrep -x exact" pgrep -x bash || true
run "pidof bash" pidof bash || true
run "pidof init" pidof init 2>/dev/null || pidof systemd 2>/dev/null || true

# ============================================================================
section "top / htop / uptime / free / vmstat"
# ============================================================================
run_t 3 "top -bn1" top -bn1 | head -20
run_t 3 "top -bn1 -o %MEM" top -bn1 -o %MEM | head -15
run_t 3 "top -bn1 -o %CPU" top -bn1 -o %CPU | head -15
run_t 3 "top -bn1 -p 1" top -bn1 -p 1
run "uptime" uptime
run "uptime -s" uptime -s 2>/dev/null || true
run "uptime -p" uptime -p 2>/dev/null || true
run "free" free
run "free -h" free -h
run "free -m" free -m
run "free -g" free -g
run "free -b" free -b
run "free -t" free -t
run "free -s 1 -c 2" free -s 1 -c 2
run "free --si" free --si 2>/dev/null || true
run "vmstat 1 3" vmstat 1 3
run "vmstat -s" vmstat -s
run "vmstat -d" vmstat -d 2>/dev/null || true
run "vmstat -w" vmstat -w 1 2
run "vmstat -a" vmstat -a 1 2

# ============================================================================
section "/proc filesystem reads"
# ============================================================================
run "cat /proc/1/status" cat /proc/1/status 2>/dev/null
run "cat /proc/1/stat" cat /proc/1/stat 2>/dev/null
run "cat /proc/1/statm" cat /proc/1/statm 2>/dev/null
run "cat /proc/1/cmdline" cat /proc/1/cmdline 2>/dev/null | tr '\0' ' '; echo
run "cat /proc/1/comm" cat /proc/1/comm 2>/dev/null
run "cat /proc/1/environ" cat /proc/1/environ 2>/dev/null | tr '\0' '\n' | head -10
run "cat /proc/1/io" cat /proc/1/io 2>/dev/null
run "cat /proc/1/limits" cat /proc/1/limits 2>/dev/null
run "cat /proc/1/maps" cat /proc/1/maps 2>/dev/null | head -10
run "cat /proc/1/smaps" cat /proc/1/smaps 2>/dev/null | head -20
run "cat /proc/1/fd list" ls -la /proc/1/fd/ 2>/dev/null | head -10
run "cat /proc/1/cgroup" cat /proc/1/cgroup 2>/dev/null
run "cat /proc/1/oom_score" cat /proc/1/oom_score 2>/dev/null
run "cat /proc/1/oom_adj" cat /proc/1/oom_score_adj 2>/dev/null
run "cat /proc/1/loginuid" cat /proc/1/loginuid 2>/dev/null
run "cat /proc/1/sessionid" cat /proc/1/sessionid 2>/dev/null
run "readlink /proc/1/exe" readlink /proc/1/exe 2>/dev/null
run "readlink /proc/1/cwd" readlink /proc/1/cwd 2>/dev/null
run "readlink /proc/1/root" readlink /proc/1/root 2>/dev/null

run "cat /proc/self/status" cat /proc/self/status
run "cat /proc/self/maps" cat /proc/self/maps | head -15
run "cat /proc/self/cmdline" cat /proc/self/cmdline | tr '\0' ' '; echo
run "ls /proc/self/fd" ls -la /proc/self/fd | head -10
run "cat /proc/self/stack" cat /proc/self/stack 2>/dev/null || true
run "cat /proc/self/syscall" cat /proc/self/syscall 2>/dev/null || true
run "cat /proc/self/mountinfo" cat /proc/self/mountinfo | head -10

run "cat /proc/cpuinfo" cat /proc/cpuinfo | head -30
run "cat /proc/meminfo" cat /proc/meminfo
run "cat /proc/version" cat /proc/version
run "cat /proc/loadavg" cat /proc/loadavg
run "cat /proc/uptime" cat /proc/uptime
run "cat /proc/stat" cat /proc/stat | head -10
run "cat /proc/mounts" cat /proc/mounts | head -10
run "cat /proc/filesystems" cat /proc/filesystems
run "cat /proc/partitions" cat /proc/partitions 2>/dev/null
run "cat /proc/diskstats" cat /proc/diskstats 2>/dev/null | head -5
run "cat /proc/swaps" cat /proc/swaps 2>/dev/null
run "cat /proc/modules" cat /proc/modules 2>/dev/null | head -10
run "cat /proc/interrupts" cat /proc/interrupts 2>/dev/null | head -10
run "cat /proc/softirqs" cat /proc/softirqs 2>/dev/null | head -10
run "cat /proc/vmstat" cat /proc/vmstat | head -20
run "cat /proc/zoneinfo" cat /proc/zoneinfo 2>/dev/null | head -20
run "cat /proc/buddyinfo" cat /proc/buddyinfo 2>/dev/null
run "cat /proc/cgroups" cat /proc/cgroups 2>/dev/null
run "cat /proc/cmdline" cat /proc/cmdline 2>/dev/null
run "cat /proc/crypto" cat /proc/crypto 2>/dev/null | head -20
run "cat /proc/devices" cat /proc/devices
run "cat /proc/dma" cat /proc/dma 2>/dev/null
run "cat /proc/iomem" cat /proc/iomem 2>/dev/null | head -10
run "cat /proc/ioports" cat /proc/ioports 2>/dev/null | head -10
run "cat /proc/kallsyms" cat /proc/kallsyms 2>/dev/null | head -5
run "cat /proc/keys" cat /proc/keys 2>/dev/null | head -5
run "cat /proc/locks" cat /proc/locks 2>/dev/null | head -5
run "cat /proc/misc" cat /proc/misc 2>/dev/null
run "cat /proc/net/tcp" cat /proc/net/tcp | head -10
run "cat /proc/net/udp" cat /proc/net/udp | head -10
run "cat /proc/net/tcp6" cat /proc/net/tcp6 | head -10
run "cat /proc/net/udp6" cat /proc/net/udp6 | head -10
run "cat /proc/net/unix" cat /proc/net/unix | head -10
run "cat /proc/net/arp" cat /proc/net/arp
run "cat /proc/net/route" cat /proc/net/route
run "cat /proc/net/dev" cat /proc/net/dev
run "cat /proc/net/sockstat" cat /proc/net/sockstat
run "cat /proc/net/netstat" cat /proc/net/netstat | head -5
run "cat /proc/net/snmp" cat /proc/net/snmp
run "cat /proc/net/protocols" cat /proc/net/protocols | head -10

run "ls /proc" ls /proc | head -20
run "ls /proc -d [0-9]*" ls -d /proc/[0-9]* 2>/dev/null | head -20

# ============================================================================
section "Background processes, job control, signals"
# ============================================================================
run "sleep background" sleep 0.5 &
run "wait" wait

run "bash -c subshell" bash -c 'echo subshell PID=$$'
run "sh -c subshell" sh -c 'echo sh PID=$$'
run "nested subshell" bash -c 'bash -c "bash -c \"echo depth=3 PID=\$\$\""'
run "subshell group" bash -c '(echo group1) && (echo group2)'
run "pipe chain 2" echo hello | tr a-z A-Z
run "pipe chain 3" cat /etc/passwd | grep root | wc -l
run "pipe chain 4" cat /etc/passwd | cut -d: -f1 | sort | head -5
run "pipe chain 5" find /usr/bin -type f | head -10 | xargs ls -la | sort -k5 -rn | head -3

# Process substitution
run "process substitution" diff <(echo hello) <(echo world) || true
run "process substitution 2" cat <(echo "from proc subst")
run "process substitution 3" wc -l <(cat /etc/passwd) <(cat /etc/group)

# Signals
sleep 300 & P1=$!
run "kill -0 check" kill -0 $P1
run "kill SIGTERM" kill -TERM $P1 2>/dev/null; wait $P1 2>/dev/null

sleep 300 & P2=$!
run "kill SIGKILL" kill -9 $P2 2>/dev/null; wait $P2 2>/dev/null

sleep 300 & P3=$!
run "kill SIGHUP" kill -HUP $P3 2>/dev/null; wait $P3 2>/dev/null

sleep 300 & P4=$!
run "kill SIGINT" kill -INT $P4 2>/dev/null; wait $P4 2>/dev/null

sleep 300 & P5=$!
run "kill SIGUSR1" kill -USR1 $P5 2>/dev/null; wait $P5 2>/dev/null

sleep 300 & P6=$!
run "kill SIGSTOP" kill -STOP $P6 2>/dev/null
run "kill SIGCONT" kill -CONT $P6 2>/dev/null
run "kill final" kill -TERM $P6 2>/dev/null; wait $P6 2>/dev/null

run "kill -l" kill -l

# nice / renice
run "nice" nice echo "nice default"
run "nice -n 10" nice -n 10 echo "nice 10"
run "nice -n 19" nice -n 19 echo "nice 19"
run "nice -n -5" nice -n -5 echo "nice -5" 2>/dev/null || true

# nohup / timeout / time
run "nohup" nohup echo "nohup test" 2>/dev/null; rm -f nohup.out
run "timeout 1" timeout 1 sleep 0.5
run "timeout fail" timeout 1 sleep 5 || true
run "time" bash -c 'time echo hello' 2>/dev/null

# Named pipes
run "mkfifo" mkfifo "$S/testpipe" 2>/dev/null
run_t 5 "named pipe rw" bash -c "echo 'pipe data' > '$S/testpipe' & cat '$S/testpipe'"
rm -f "$S/testpipe"

# ============================================================================
section "env / printenv / export"
# ============================================================================
run "env" env | head -20
run "env -i" env -i PATH=/usr/bin:/bin HOME=/tmp echo "clean env"
run "env VAR=test" env PROV_TEST=hello bash -c 'echo $PROV_TEST'
run "printenv" printenv | head -20
run "printenv PATH" printenv PATH
run "printenv HOME" printenv HOME
run "printenv USER" printenv USER 2>/dev/null || true
run "printenv SHELL" printenv SHELL 2>/dev/null || true

# ============================================================================
section "lsof — list open files"
# ============================================================================
run "lsof" lsof 2>/dev/null | head -20
run "lsof -p self" lsof -p $$ 2>/dev/null | head -15
run "lsof -p 1" lsof -p 1 2>/dev/null | head -15
run "lsof -i" lsof -i 2>/dev/null | head -15
run "lsof -i tcp" lsof -i tcp 2>/dev/null | head -10
run "lsof -i udp" lsof -i udp 2>/dev/null | head -10
run "lsof -i :22" lsof -i :22 2>/dev/null | head -5
run "lsof -u root" lsof -u root 2>/dev/null | head -15
run "lsof -c bash" lsof -c bash 2>/dev/null | head -15
run "lsof +D /tmp" lsof +D /tmp 2>/dev/null | head -10
run "lsof -t" lsof -t -i :22 2>/dev/null || true
run "lsof -P" lsof -P -i 2>/dev/null | head -10
run "lsof -n" lsof -n -i 2>/dev/null | head -10
run "lsof -nP" lsof -nP -i 2>/dev/null | head -10

# ============================================================================
section "strace / ltrace"
# ============================================================================
run "strace ls" strace -e trace=open,openat,read,write ls /tmp 2>/dev/null
run "strace -c ls" strace -c ls /tmp 2>/dev/null
run "strace -f" strace -f -e trace=clone,execve bash -c 'echo traced' 2>/dev/null
run "strace -e network" strace -e trace=network curl -s https://example.com -o /dev/null 2>/dev/null
run "strace -e file" strace -e trace=file cat /etc/hostname 2>/dev/null
run "strace -e process" strace -e trace=process bash -c 'echo test' 2>/dev/null
run "strace -e memory" strace -e trace=memory ls 2>/dev/null
run "strace -e signal" strace -e trace=signal bash -c 'kill -0 $$' 2>/dev/null
run "strace -t" strace -t -e trace=write echo hello 2>/dev/null
run "strace -T" strace -T -e trace=write echo hello 2>/dev/null
run "strace -o file" strace -o "$S/strace.log" -e trace=open ls /tmp 2>/dev/null; head -5 "$S/strace.log"
run "strace -p timeout" timeout 2 strace -p 1 -e trace=write 2>/dev/null || true
run "ltrace ls" ltrace -e printf ls /tmp 2>/dev/null || true

# ============================================================================
section "ulimit / resource limits"
# ============================================================================
run "ulimit -a" ulimit -a
run "ulimit -n" ulimit -n
run "ulimit -u" ulimit -u
run "ulimit -s" ulimit -s
run "ulimit -v" ulimit -v
run "ulimit -m" ulimit -m
run "ulimit -l" ulimit -l
run "ulimit -f" ulimit -f
run "ulimit -c" ulimit -c
run "ulimit -t" ulimit -t
run "ulimit -Sn" ulimit -Sn
run "ulimit -Hn" ulimit -Hn

# ============================================================================
section "fuser / pstree"
# ============================================================================
run "fuser /" fuser / 2>/dev/null || true
run "fuser /tmp" fuser /tmp 2>/dev/null || true
run "fuser -v" fuser -v / 2>/dev/null || true
run "pstree" pstree 2>/dev/null || true
run "pstree -p" pstree -p 2>/dev/null || true
run "pstree -a" pstree -a 2>/dev/null || true
run "pstree -u" pstree -u 2>/dev/null || true
run "pstree -l" pstree -l 2>/dev/null || true
run "pstree 1" pstree 1 2>/dev/null || true

rm -f "$S/strace.log"

domain_end
