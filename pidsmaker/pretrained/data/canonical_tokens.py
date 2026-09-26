"""
Canonical neighbor target tokens for provenance graph foundation model.

These tokens abstract dataset-specific details (paths, IPs, usernames) into
OS-agnostic functional categories. When canonicalize_neighbors is enabled,
these tokens are:
  - Appended to encoder input (full detail + category signal)
  - Used as the ONLY content for decoder targets (existing [] special tokens
    + these category tokens, stripping all dataset-specific BPE words)

Design principle: same function = same token, regardless of OS.
  - Linux /var/log/nginx/ = Windows IIS LogFiles/ = [FCAT_LOG_WEB]
  - Linux /etc/shadow = Windows SAM hive = [FCAT_CREDENTIAL]
  - Linux cron = Windows Task Scheduler = [CAT_CRON]
"""

from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS CATEGORIES (PROC_CATEGORIES)
# ═══════════════════════════════════════════════════════════════════════════
#
# Maps binary names (extracted from command lines, stripped of path and .exe)
# to functional category tokens. OS-agnostic: nginx on Linux and Windows
# both map to [CAT_WEBSERVER].
#
# Lookup procedure:
#   1. Extract binary name: /usr/sbin/nginx → nginx
#                           C:/Windows/system32/svchost.exe → svchost
#                           com.android.chrome → com.android.chrome
#   2. Look up in PROC_CATEGORIES
#   3. If not found, fall back to tokenizing the binary name directly
# ═══════════════════════════════════════════════════════════════════════════

PROC_CATEGORIES = {
    # ── Web servers ──
    "nginx":            "[CAT_WEBSERVER]",
    "apache2":          "[CAT_WEBSERVER]",
    "httpd":            "[CAT_WEBSERVER]",
    "lighttpd":         "[CAT_WEBSERVER]",
    "caddy":            "[CAT_WEBSERVER]",
    "traefik":          "[CAT_WEBSERVER]",
    "w3wp":             "[CAT_WEBSERVER]",
    "iisexpress":       "[CAT_WEBSERVER]",

    # ── Application servers ──
    "gunicorn":         "[CAT_WEBSERVER]",
    "uwsgi":            "[CAT_WEBSERVER]",
    "php-fpm":          "[CAT_WEBSERVER]",
    "php-fpm7.4":       "[CAT_WEBSERVER]",
    "php-fpm8.0":       "[CAT_WEBSERVER]",
    "php-fpm8.1":       "[CAT_WEBSERVER]",
    "php-fpm8.2":       "[CAT_WEBSERVER]",
    "php-fpm8.3":       "[CAT_WEBSERVER]",
    "php-cgi":          "[CAT_WEBSERVER]",
    "puma":             "[CAT_WEBSERVER]",
    "unicorn":          "[CAT_WEBSERVER]",
    "daphne":           "[CAT_WEBSERVER]",
    "uvicorn":          "[CAT_WEBSERVER]",
    "passenger":        "[CAT_WEBSERVER]",
    "thin":             "[CAT_WEBSERVER]",
    "waitress-serve":   "[CAT_WEBSERVER]",

    # ── Reverse proxy / load balancer ──
    "haproxy":          "[CAT_PROXY]",
    "envoy":            "[CAT_PROXY]",
    "squid":            "[CAT_PROXY]",
    "varnishd":         "[CAT_PROXY]",
    "pound":            "[CAT_PROXY]",

    # ── Databases ──
    "mysqld":           "[CAT_DATABASE]",
    "mariadbd":         "[CAT_DATABASE]",
    "postgres":         "[CAT_DATABASE]",
    "mongod":           "[CAT_DATABASE]",
    "mongos":           "[CAT_DATABASE]",
    "redis-server":     "[CAT_DATABASE]",
    "sqlservr":         "[CAT_DATABASE]",
    "oracle":           "[CAT_DATABASE]",
    "clickhouse-server":"[CAT_DATABASE]",
    "influxd":          "[CAT_DATABASE]",
    "cassandra":        "[CAT_DATABASE]",
    "couchdb":          "[CAT_DATABASE]",
    "etcd":             "[CAT_DATABASE]",

    # ── Database clients ──
    "mysql":            "[CAT_DB_CLIENT]",
    "psql":             "[CAT_DB_CLIENT]",
    "mongo":            "[CAT_DB_CLIENT]",
    "redis-cli":        "[CAT_DB_CLIENT]",
    "sqlcmd":           "[CAT_DB_CLIENT]",
    "sqlite3":          "[CAT_DB_CLIENT]",
    "pg_dump":          "[CAT_DB_CLIENT]",
    "pg_restore":       "[CAT_DB_CLIENT]",
    "pg_basebackup":    "[CAT_DB_CLIENT]",
    "mysqldump":        "[CAT_DB_CLIENT]",
    "mongodump":        "[CAT_DB_CLIENT]",
    "mongorestore":     "[CAT_DB_CLIENT]",

    # ── Message queues / brokers ──
    "beam.smp":         "[CAT_MSGQUEUE]",
    "rabbitmq-server":  "[CAT_MSGQUEUE]",
    "kafka":            "[CAT_MSGQUEUE]",
    "mosquitto":        "[CAT_MSGQUEUE]",
    "nats-server":      "[CAT_MSGQUEUE]",
    "activemq":         "[CAT_MSGQUEUE]",

    # ── Cache ──
    "memcached":        "[CAT_CACHE]",

    # ── Shells ──
    "bash":             "[CAT_SHELL]",
    "sh":               "[CAT_SHELL]",
    "zsh":              "[CAT_SHELL]",
    "dash":             "[CAT_SHELL]",
    "fish":             "[CAT_SHELL]",
    "csh":              "[CAT_SHELL]",
    "tcsh":             "[CAT_SHELL]",
    "ksh":              "[CAT_SHELL]",
    "ash":              "[CAT_SHELL]",
    "cmd":              "[CAT_SHELL]",
    "powershell":       "[CAT_SHELL]",
    "pwsh":             "[CAT_SHELL]",
    "explorer":         "[CAT_SHELL]",
    "WindowsTerminal":  "[CAT_SHELL]",

    # ── SSH ──
    "sshd":             "[CAT_SSH]",
    "ssh":              "[CAT_SSH]",
    "scp":              "[CAT_SSH]",
    "sftp":             "[CAT_SSH]",
    "sftp-server":      "[CAT_SSH]",
    "ssh-agent":        "[CAT_SSH]",
    "ssh-keygen":       "[CAT_SSH]",
    "ssh-add":          "[CAT_SSH]",
    "ssh-keyscan":      "[CAT_SSH]",
    "dropbear":         "[CAT_SSH]",
    "dbclient":         "[CAT_SSH]",
    "rsync":            "[CAT_SSH]",
    "putty":            "[CAT_SSH]",
    "plink":            "[CAT_SSH]",

    # ── Mail ──
    "smtpd":            "[CAT_MAIL]",
    "smtp":             "[CAT_MAIL]",
    "sendmail":         "[CAT_MAIL]",
    "exim":             "[CAT_MAIL]",
    "exim4":            "[CAT_MAIL]",
    "postfix":          "[CAT_MAIL]",
    "dovecot":          "[CAT_MAIL]",
    "imap":             "[CAT_MAIL]",
    "imap-login":       "[CAT_MAIL]",
    "pop3":             "[CAT_MAIL]",
    "pop3-login":       "[CAT_MAIL]",
    "lmtp":             "[CAT_MAIL]",
    "cleanup":          "[CAT_MAIL]",
    "qmgr":             "[CAT_MAIL]",
    "master":           "[CAT_MAIL]",
    "pickup":           "[CAT_MAIL]",
    "bounce":           "[CAT_MAIL]",
    "local":            "[CAT_MAIL]",
    "amavisd-new":      "[CAT_MAIL]",
    "spamd":            "[CAT_MAIL]",
    "opendkim":         "[CAT_MAIL]",
    "OUTLOOK":          "[CAT_MAIL]",
    "thunderbird":      "[CAT_MAIL]",

    # ── DNS ──
    "named":            "[CAT_INIT]",
    "unbound":          "[CAT_INIT]",
    "dnsmasq":          "[CAT_INIT]",
    "systemd-resolved": "[CAT_INIT]",
    "coredns":          "[CAT_INIT]",
    "dig":              "[CAT_INIT]",
    "nslookup":         "[CAT_INIT]",
    "host":             "[CAT_INIT]",

    # ── Init / process supervisors ──
    "systemd":          "[CAT_INIT]",
    "init":             "[CAT_INIT]",
    "busybox":          "[CAT_INIT]",
    "openrc":           "[CAT_INIT]",
    "upstart":          "[CAT_INIT]",
    "services":         "[CAT_INIT]",
    "wininit":          "[CAT_INIT]",
    "smss":             "[CAT_INIT]",
    "launchd":          "[CAT_INIT]",

    # ── Service hosting ──
    "svchost":          "[CAT_SVCHOST]",

    # ── Cron / scheduled tasks ──
    "cron":             "[CAT_CRON]",
    "crond":            "[CAT_CRON]",
    "CRON":             "[CAT_CRON]",
    "anacron":          "[CAT_CRON]",
    "atd":              "[CAT_CRON]",
    "at":               "[CAT_CRON]",
    "schtasks":         "[CAT_CRON]",
    "taskeng":          "[CAT_CRON]",
    "taskhostw":        "[CAT_CRON]",

    # ── Container / orchestration ──
    "dockerd":              "[CAT_EXECUTABLE]",
    "containerd":           "[CAT_EXECUTABLE]",
    "containerd-shim":      "[CAT_EXECUTABLE]",
    "containerd-shim-runc-v2": "[CAT_EXECUTABLE]",
    "runc":                 "[CAT_EXECUTABLE]",
    "crun":                 "[CAT_EXECUTABLE]",
    "conmon":               "[CAT_EXECUTABLE]",
    "cri-o":                "[CAT_EXECUTABLE]",
    "podman":               "[CAT_EXECUTABLE]",
    "buildah":              "[CAT_EXECUTABLE]",
    "skopeo":               "[CAT_EXECUTABLE]",
    "docker":               "[CAT_EXECUTABLE]",
    "docker-compose":       "[CAT_EXECUTABLE]",
    "kubelet":              "[CAT_EXECUTABLE]",
    "kube-apiserver":       "[CAT_EXECUTABLE]",
    "kube-controller-manager": "[CAT_EXECUTABLE]",
    "kube-scheduler":       "[CAT_EXECUTABLE]",
    "kube-proxy":           "[CAT_EXECUTABLE]",
    "kubectl":              "[CAT_EXECUTABLE]",
    "helm":                 "[CAT_EXECUTABLE]",

    # ── Logging ──
    "rsyslogd":             "[CAT_SYSADMIN]",
    "syslog-ng":            "[CAT_SYSADMIN]",
    "syslogd":              "[CAT_SYSADMIN]",
    "systemd-journald":     "[CAT_SYSADMIN]",
    "filebeat":             "[CAT_SYSADMIN]",
    "fluentd":              "[CAT_SYSADMIN]",
    "fluent-bit":           "[CAT_SYSADMIN]",
    "vector":               "[CAT_SYSADMIN]",
    "logstash":             "[CAT_SYSADMIN]",
    "logd":                 "[CAT_SYSADMIN]",

    # ── Monitoring / observability ──
    "prometheus":           "[CAT_MONITORING]",
    "grafana-server":       "[CAT_MONITORING]",
    "grafana":              "[CAT_MONITORING]",
    "node_exporter":        "[CAT_MONITORING]",
    "blackbox_exporter":    "[CAT_MONITORING]",
    "alertmanager":         "[CAT_MONITORING]",
    "telegraf":             "[CAT_MONITORING]",
    "collectd":             "[CAT_MONITORING]",
    "zabbix_agentd":        "[CAT_MONITORING]",
    "zabbix_server":        "[CAT_MONITORING]",
    "nagios":               "[CAT_MONITORING]",
    "icinga2":              "[CAT_MONITORING]",

    # ── Security / AV / audit ──
    "clamd":                "[CAT_SYSADMIN]",
    "clamscan":             "[CAT_SYSADMIN]",
    "freshclam":            "[CAT_SYSADMIN]",
    "falco":                "[CAT_SYSADMIN]",
    "auditd":               "[CAT_SYSADMIN]",
    "osqueryd":             "[CAT_SYSADMIN]",
    "ossec":                "[CAT_SYSADMIN]",
    "fail2ban-server":      "[CAT_SYSADMIN]",
    "MsMpEng":              "[CAT_SYSADMIN]",
    "MpCmdRun":             "[CAT_SYSADMIN]",
    "NisSrv":               "[CAT_SYSADMIN]",
    "Sysmon":               "[CAT_SYSADMIN]",
    "Sysmon64":             "[CAT_SYSADMIN]",

    # ── Authentication ──
    "login":                "[CAT_SYSADMIN]",
    "su":                   "[CAT_SYSADMIN]",
    "sudo":                 "[CAT_SYSADMIN]",
    "passwd":               "[CAT_SYSADMIN]",
    "chpasswd":             "[CAT_SYSADMIN]",
    "useradd":              "[CAT_SYSADMIN]",
    "usermod":              "[CAT_SYSADMIN]",
    "userdel":              "[CAT_SYSADMIN]",
    "groupadd":             "[CAT_SYSADMIN]",
    "chage":                "[CAT_SYSADMIN]",
    "kinit":                "[CAT_SYSADMIN]",
    "klist":                "[CAT_SYSADMIN]",
    "kdestroy":             "[CAT_SYSADMIN]",
    "lsass":                "[CAT_SYSADMIN]",
    "winlogon":             "[CAT_SYSADMIN]",
    "LogonUI":              "[CAT_SYSADMIN]",
    "consent":              "[CAT_SYSADMIN]",

    # ── Package managers ──
    "apt":                  "[CAT_PKGMGR]",
    "apt-get":              "[CAT_PKGMGR]",
    "apt-cache":            "[CAT_PKGMGR]",
    "dpkg":                 "[CAT_PKGMGR]",
    "yum":                  "[CAT_PKGMGR]",
    "dnf":                  "[CAT_PKGMGR]",
    "rpm":                  "[CAT_PKGMGR]",
    "pacman":               "[CAT_PKGMGR]",
    "zypper":               "[CAT_PKGMGR]",
    "apk":                  "[CAT_PKGMGR]",
    "snap":                 "[CAT_PKGMGR]",
    "flatpak":              "[CAT_PKGMGR]",
    "pip":                  "[CAT_PKGMGR]",
    "pip3":                 "[CAT_PKGMGR]",
    "npm":                  "[CAT_PKGMGR]",
    "yarn":                 "[CAT_PKGMGR]",
    "cargo":                "[CAT_PKGMGR]",
    "gem":                  "[CAT_PKGMGR]",
    "composer":             "[CAT_PKGMGR]",
    "go":                   "[CAT_PKGMGR]",
    "msiexec":              "[CAT_PKGMGR]",
    "choco":                "[CAT_PKGMGR]",
    "winget":               "[CAT_PKGMGR]",
    "wusa":                 "[CAT_PKGMGR]",
    "TrustedInstaller":     "[CAT_PKGMGR]",
    "UsoClient":            "[CAT_PKGMGR]",
    "installd":             "[CAT_PKGMGR]",

    # ── Compilers / build tools ──
    "gcc":                  "[CAT_COMPILER]",
    "g++":                  "[CAT_COMPILER]",
    "cc1":                  "[CAT_COMPILER]",
    "cc1plus":              "[CAT_COMPILER]",
    "as":                   "[CAT_COMPILER]",
    "ld":                   "[CAT_COMPILER]",
    "collect2":             "[CAT_COMPILER]",
    "clang":                "[CAT_COMPILER]",
    "clang++":              "[CAT_COMPILER]",
    "rustc":                "[CAT_COMPILER]",
    "javac":                "[CAT_COMPILER]",
    "cl":                   "[CAT_COMPILER]",
    "link":                 "[CAT_COMPILER]",
    "MSBuild":              "[CAT_COMPILER]",
    "tsc":                  "[CAT_COMPILER]",
    "swc":                  "[CAT_COMPILER]",
    "esbuild":              "[CAT_COMPILER]",

    # ── Build systems ──
    "make":                 "[CAT_BUILD]",
    "gmake":                "[CAT_BUILD]",
    "cmake":                "[CAT_BUILD]",
    "ninja":                "[CAT_BUILD]",
    "bazel":                "[CAT_BUILD]",
    "gradle":               "[CAT_BUILD]",
    "gradlew":              "[CAT_BUILD]",
    "mvn":                  "[CAT_BUILD]",
    "ant":                  "[CAT_BUILD]",
    "meson":                "[CAT_BUILD]",

    # ── CI/CD runners ──
    "gitlab-runner":        "[CAT_CIRUNNER]",
    "jenkins-agent":        "[CAT_CIRUNNER]",
    "github-actions-runner":"[CAT_CIRUNNER]",

    # ── Version control ──
    "git":                  "[CAT_VCS]",
    "git-receive-pack":     "[CAT_VCS]",
    "git-upload-pack":      "[CAT_VCS]",
    "git-remote-https":     "[CAT_VCS]",
    "svn":                  "[CAT_VCS]",
    "hg":                   "[CAT_VCS]",

    # ── Browsers ──
    "firefox":              "[CAT_BROWSER]",
    "chrome":               "[CAT_BROWSER]",
    "chromium":             "[CAT_BROWSER]",
    "chromium-browser":     "[CAT_BROWSER]",
    "msedge":               "[CAT_BROWSER]",
    "iexplore":             "[CAT_BROWSER]",
    "opera":                "[CAT_BROWSER]",
    "brave":                "[CAT_BROWSER]",
    "vivaldi":              "[CAT_BROWSER]",
    "epiphany":             "[CAT_BROWSER]",

    # ── Office / productivity ──
    "soffice.bin":          "[CAT_OFFICE]",
    "soffice":              "[CAT_OFFICE]",
    "WINWORD":              "[CAT_OFFICE]",
    "EXCEL":                "[CAT_OFFICE]",
    "POWERPNT":             "[CAT_OFFICE]",
    "MSACCESS":             "[CAT_OFFICE]",
    "ONENOTE":              "[CAT_OFFICE]",

    # ── Runtimes / interpreters ──
    "python":               "[CAT_RUNTIME]",
    "python3":              "[CAT_RUNTIME]",
    "python3.10":           "[CAT_RUNTIME]",
    "python3.11":           "[CAT_RUNTIME]",
    "python3.12":           "[CAT_RUNTIME]",
    "node":                 "[CAT_RUNTIME]",
    "ruby":                 "[CAT_RUNTIME]",
    "perl":                 "[CAT_RUNTIME]",
    "php":                  "[CAT_RUNTIME]",
    "java":                 "[CAT_RUNTIME]",
    "dotnet":               "[CAT_RUNTIME]",
    "mono":                 "[CAT_RUNTIME]",
    "lua":                  "[CAT_RUNTIME]",
    "Rscript":              "[CAT_RUNTIME]",
    "app_process64":        "[CAT_RUNTIME]",
    "app_process32":        "[CAT_RUNTIME]",

    # ── Windows system processes ──
    "csrss":                "[CAT_WIN_SYSTEM]",
    "dwm":                  "[CAT_WIN_SYSTEM]",
    "RuntimeBroker":        "[CAT_WIN_SYSTEM]",
    "backgroundTaskHost":   "[CAT_WIN_SYSTEM]",
    "SearchUI":             "[CAT_WIN_SYSTEM]",
    "SearchApp":            "[CAT_WIN_SYSTEM]",
    "ShellExperienceHost":  "[CAT_WIN_SYSTEM]",
    "ApplicationFrameHost": "[CAT_WIN_SYSTEM]",
    "SystemSettings":       "[CAT_WIN_SYSTEM]",
    "wmiprvse":             "[CAT_WIN_SYSTEM]",
    "WmiApSrv":             "[CAT_WIN_SYSTEM]",
    "WerFault":             "[CAT_WIN_SYSTEM]",
    "dllhost":              "[CAT_WIN_SYSTEM]",
    "conhost":              "[CAT_WIN_SYSTEM]",
    "System":               "[CAT_WIN_SYSTEM]",
    "spoolsv":              "[CAT_WIN_SYSTEM]",

    # ── Android system processes ──
    "zygote":               "[CAT_ANDROID_SYS]",
    "zygote64":             "[CAT_ANDROID_SYS]",
    "system_server":        "[CAT_ANDROID_SYS]",
    "surfaceflinger":       "[CAT_ANDROID_SYS]",
    "servicemanager":       "[CAT_ANDROID_SYS]",
    "hwservicemanager":     "[CAT_ANDROID_SYS]",
    "vold":                 "[CAT_ANDROID_SYS]",
    "netd":                 "[CAT_ANDROID_SYS]",
    "logd":                 "[CAT_ANDROID_SYS]",
    "audioserver":          "[CAT_ANDROID_SYS]",
    "mediaserver":          "[CAT_ANDROID_SYS]",
    "cameraserver":         "[CAT_ANDROID_SYS]",
    "installd":             "[CAT_ANDROID_SYS]",
    "adbd":                 "[CAT_ANDROID_SYS]",
    "healthd":              "[CAT_ANDROID_SYS]",
    "gatekeeperd":          "[CAT_ANDROID_SYS]",
    "tombstoned":           "[CAT_ANDROID_SYS]",
    "ueventd":              "[CAT_ANDROID_SYS]",
    "storaged":             "[CAT_ANDROID_SYS]",
    "lmkd":                 "[CAT_ANDROID_SYS]",

    # ── VPN / tunneling ──
    "openvpn":              "[CAT_VPN]",
    "wg":                   "[CAT_VPN]",
    "wg-quick":             "[CAT_VPN]",
    "wireguard-go":         "[CAT_VPN]",
    "strongswan":           "[CAT_VPN]",
    "charon":               "[CAT_VPN]",
    "tailscaled":           "[CAT_VPN]",
    "tailscale":            "[CAT_VPN]",

    # ── Firewall / network tools ──
    "iptables":             "[CAT_SYSADMIN]",
    "ip6tables":            "[CAT_SYSADMIN]",
    "nft":                  "[CAT_SYSADMIN]",
    "ufw":                  "[CAT_SYSADMIN]",
    "firewalld":            "[CAT_SYSADMIN]",
    "netsh":                "[CAT_SYSADMIN]",
    "pf":                   "[CAT_SYSADMIN]",

    # ── Infra-as-code / config management ──
    "ansible":              "[CAT_EXECUTABLE]",
    "ansible-playbook":     "[CAT_EXECUTABLE]",
    "terraform":            "[CAT_EXECUTABLE]",
    "puppet":               "[CAT_EXECUTABLE]",
    "chef-client":          "[CAT_EXECUTABLE]",
    "salt-minion":          "[CAT_EXECUTABLE]",
    "salt-master":          "[CAT_EXECUTABLE]",
    "packer":               "[CAT_EXECUTABLE]",

    # ── Backup tools ──
    "tar":                  "[CAT_ARCHIVE]",
    "gzip":                 "[CAT_ARCHIVE]",
    "bzip2":                "[CAT_ARCHIVE]",
    "xz":                   "[CAT_ARCHIVE]",
    "zip":                  "[CAT_ARCHIVE]",
    "unzip":                "[CAT_ARCHIVE]",
    "7z":                   "[CAT_ARCHIVE]",
    "borg":                 "[CAT_ARCHIVE]",
    "restic":               "[CAT_ARCHIVE]",
    "duplicity":            "[CAT_ARCHIVE]",

    # ── Text editors / IDEs ──
    "vim":                  "[CAT_EDITOR]",
    "nvim":                 "[CAT_EDITOR]",
    "nano":                 "[CAT_EDITOR]",
    "emacs":                "[CAT_EDITOR]",
    "code":                 "[CAT_EDITOR]",
    "Code":                 "[CAT_EDITOR]",
    "devenv":               "[CAT_EDITOR]",
    "gedit":                "[CAT_EDITOR]",
    "kate":                 "[CAT_EDITOR]",
    "mousepad":             "[CAT_EDITOR]",
    "notepad":              "[CAT_EDITOR]",
    "notepad++":            "[CAT_EDITOR]",
    "sublime_text":         "[CAT_EDITOR]",

    # ── File transfer ──
    "curl":                 "[CAT_TRANSFER]",
    "wget":                 "[CAT_TRANSFER]",
    "aria2c":               "[CAT_TRANSFER]",
    "ftp":                  "[CAT_TRANSFER]",
    "lftp":                 "[CAT_TRANSFER]",

    # ── Network scanning / diagnostic ──
    "nmap":                 "[CAT_NETSCAN]",
    "masscan":              "[CAT_NETSCAN]",
    "ping":                 "[CAT_NETSCAN]",
    "traceroute":           "[CAT_NETSCAN]",
    "tracert":              "[CAT_NETSCAN]",
    "mtr":                  "[CAT_NETSCAN]",
    "ss":                   "[CAT_NETSCAN]",
    "netstat":              "[CAT_NETSCAN]",
    "tcpdump":              "[CAT_NETSCAN]",
    "tshark":               "[CAT_NETSCAN]",
    "ncat":                 "[CAT_NETSCAN]",
    "nc":                   "[CAT_NETSCAN]",
    "socat":                "[CAT_NETSCAN]",

    # ── Crypto / certificate tools ──
    "openssl":              "[CAT_CRYPTO]",
    "certbot":              "[CAT_CRYPTO]",
    "gpg":                  "[CAT_CRYPTO]",
    "gpg-agent":            "[CAT_CRYPTO]",
    "certutil":             "[CAT_CRYPTO]",

    # ── System info / admin ──
    "systemctl":            "[CAT_SYSADMIN]",
    "journalctl":           "[CAT_SYSADMIN]",
    "service":              "[CAT_SYSADMIN]",
    "sc":                   "[CAT_SYSADMIN]",
    "tasklist":             "[CAT_SYSADMIN]",
    "taskkill":             "[CAT_SYSADMIN]",
    "kill":                 "[CAT_SYSADMIN]",
    "pkill":                "[CAT_SYSADMIN]",
    "ps":                   "[CAT_SYSADMIN]",
    "top":                  "[CAT_SYSADMIN]",
    "htop":                 "[CAT_SYSADMIN]",
    "lsof":                 "[CAT_SYSADMIN]",
    "strace":               "[CAT_SYSADMIN]",
    "ltrace":               "[CAT_SYSADMIN]",
    "dmesg":                "[CAT_SYSADMIN]",
    "mount":                "[CAT_SYSADMIN]",
    "umount":               "[CAT_SYSADMIN]",
    "fdisk":                "[CAT_SYSADMIN]",
    "lsblk":                "[CAT_SYSADMIN]",
    "df":                   "[CAT_SYSADMIN]",
    "du":                   "[CAT_SYSADMIN]",
    "free":                 "[CAT_SYSADMIN]",
    "uptime":               "[CAT_SYSADMIN]",
    "reboot":               "[CAT_SYSADMIN]",
    "shutdown":             "[CAT_SYSADMIN]",
    "hostname":             "[CAT_SYSADMIN]",
    "uname":                "[CAT_SYSADMIN]",
    "id":                   "[CAT_SYSADMIN]",
    "whoami":               "[CAT_SYSADMIN]",
    "w":                    "[CAT_SYSADMIN]",
    "who":                  "[CAT_SYSADMIN]",
    "last":                 "[CAT_SYSADMIN]",
    "regedit":              "[CAT_SYSADMIN]",
    "reg":                  "[CAT_SYSADMIN]",

    # ── Logrotate / maintenance ──
    "logrotate":            "[CAT_SYSADMIN]",
    "fstrim":               "[CAT_SYSADMIN]",
    "updatedb":             "[CAT_SYSADMIN]",
    "mandb":                "[CAT_SYSADMIN]",
    "ldconfig":             "[CAT_SYSADMIN]",

    # ── NFS / file sharing ──
    "nfsd":                 "[CAT_FILESHARE]",
    "mount.nfs":            "[CAT_FILESHARE]",
    "rpc.mountd":           "[CAT_FILESHARE]",
    "rpc.statd":            "[CAT_FILESHARE]",
    "smbd":                 "[CAT_FILESHARE]",
    "nmbd":                 "[CAT_FILESHARE]",
    "winbindd":             "[CAT_FILESHARE]",

    # ── NTP / time ──
    "ntpd":                 "[CAT_INIT]",
    "chronyd":              "[CAT_INIT]",
    "systemd-timesyncd":    "[CAT_INIT]",
    "w32tm":                "[CAT_INIT]",

    # ── DHCP ──
    "dhclient":             "[CAT_INIT]",
    "dhcpd":                "[CAT_INIT]",
    "NetworkManager":       "[CAT_INIT]",
    "systemd-networkd":     "[CAT_INIT]",

    # ── Scripting engines (often LOLBins in attack context) ──
    "wscript":              "[CAT_SCRIPTING]",
    "cscript":              "[CAT_SCRIPTING]",
    "mshta":                "[CAT_SCRIPTING]",

    # ── Windows LOLBins (legitimate tools abused for attacks) ──
    "bitsadmin":            "[CAT_EXECUTABLE]",
    "regsvr32":             "[CAT_EXECUTABLE]",
    "rundll32":             "[CAT_EXECUTABLE]",
    "msiexec":              "[CAT_EXECUTABLE]",
    "forfiles":             "[CAT_EXECUTABLE]",
    "pcalua":               "[CAT_EXECUTABLE]",
    "procdump":             "[CAT_EXECUTABLE]",
}


# ═══════════════════════════════════════════════════════════════════════════
# FILE PATH CATEGORIES (FILE_DIR_CATEGORIES)
# ═══════════════════════════════════════════════════════════════════════════
#
# Maps directory path substrings to functional categories.
# Matched most-specific-first (sort by length descending before matching).
#
# OS-AGNOSTIC: same function = same token regardless of OS.
# ═══════════════════════════════════════════════════════════════════════════

FILE_DIR_CATEGORIES = {
    # ── Config: service-specific ──
    "/etc/nginx/":                          "[FCAT_CONFIG_WEB]",
    "/etc/apache2/":                        "[FCAT_CONFIG_WEB]",
    "/etc/httpd/":                          "[FCAT_CONFIG_WEB]",
    "/etc/caddy/":                          "[FCAT_CONFIG_WEB]",
    "/etc/lighttpd/":                       "[FCAT_CONFIG_WEB]",
    "/etc/traefik/":                        "[FCAT_CONFIG_WEB]",
    "/inetsrv/":                            "[FCAT_CONFIG_WEB]",
    "/etc/haproxy/":                        "[FCAT_CONFIG_WEB]",
    "/etc/squid/":                          "[FCAT_CONFIG_WEB]",

    "/etc/ssh/":                            "[FCAT_CONFIG]",
    "/ProgramData/ssh/":                    "[FCAT_CONFIG]",
    "/.ssh/":                               "[FCAT_CONFIG]",

    "/etc/mysql/":                          "[FCAT_CONFIG]",
    "/etc/postgresql/":                     "[FCAT_CONFIG]",
    "/etc/redis/":                          "[FCAT_CONFIG]",
    "/etc/mongod":                          "[FCAT_CONFIG]",
    "/MSSQL/":                              "[FCAT_CONFIG]",

    "/etc/postfix/":                        "[FCAT_CONFIG_MAIL]",
    "/etc/dovecot/":                        "[FCAT_CONFIG_MAIL]",
    "/etc/exim":                            "[FCAT_CONFIG_MAIL]",
    "/etc/amavis/":                         "[FCAT_CONFIG_MAIL]",

    "/etc/pam.d/":                          "[CAT_SYSADMIN]",
    "/etc/sudoers":                         "[CAT_SYSADMIN]",
    "/etc/security/":                       "[CAT_SYSADMIN]",
    "/etc/krb5":                            "[CAT_SYSADMIN]",
    "/etc/ldap/":                           "[CAT_SYSADMIN]",
    "/etc/sssd/":                           "[CAT_SYSADMIN]",
    "/etc/nsswitch":                        "[CAT_SYSADMIN]",

    "/etc/systemd/":                        "[FCAT_CONFIG]",
    "/lib/systemd/":                        "[FCAT_CONFIG]",
    "/etc/init.d/":                         "[FCAT_CONFIG]",
    "/etc/rc":                              "[FCAT_CONFIG]",

    "/etc/docker/":                         "[FCAT_CONFIG_CONTAINER]",
    "/etc/containerd/":                     "[FCAT_CONFIG_CONTAINER]",
    "/etc/kubernetes/":                     "[FCAT_CONFIG_CONTAINER]",
    "/.kube/":                              "[FCAT_CONFIG_CONTAINER]",

    "/etc/prometheus/":                     "[FCAT_CONFIG_MONITORING]",
    "/etc/grafana/":                        "[FCAT_CONFIG_MONITORING]",
    "/etc/zabbix/":                         "[FCAT_CONFIG_MONITORING]",
    "/etc/nagios/":                         "[FCAT_CONFIG_MONITORING]",
    "/etc/telegraf/":                       "[FCAT_CONFIG_MONITORING]",

    "/etc/openvpn/":                        "[FCAT_CONFIG_VPN]",
    "/etc/wireguard/":                      "[FCAT_CONFIG_VPN]",

    "/etc/fail2ban/":                       "[FCAT_CONFIG_SECURITY]",
    "/etc/audit/":                          "[FCAT_CONFIG_SECURITY]",
    "/etc/clamav/":                         "[FCAT_CONFIG_SECURITY]",
    "/Windows Defender/":                   "[FCAT_CONFIG_SECURITY]",

    "/etc/logrotate":                       "[FCAT_CONFIG]",
    "/etc/rsyslog":                         "[FCAT_CONFIG]",
    "/etc/syslog":                          "[FCAT_CONFIG]",
    "/etc/filebeat/":                       "[FCAT_CONFIG]",
    "/etc/fluentd/":                        "[FCAT_CONFIG]",
    "/etc/fluent-bit/":                     "[FCAT_CONFIG]",

    "/etc/cron":                            "[FCAT_CONFIG]",
    "/system32/Tasks/":                     "[FCAT_CONFIG]",

    "/etc/fstab":                           "[FCAT_CONFIG_FS]",
    "/etc/exports":                         "[FCAT_CONFIG_FS]",
    "/etc/samba/":                          "[FCAT_CONFIG_FS]",
    "/etc/nfs":                             "[FCAT_CONFIG_FS]",

    "/etc/network/":                        "[FCAT_CONFIG]",
    "/etc/netplan/":                        "[FCAT_CONFIG]",
    "/etc/NetworkManager/":                 "[FCAT_CONFIG]",
    "/etc/resolv.conf":                     "[FCAT_CONFIG]",
    "/etc/hosts":                           "[FCAT_CONFIG]",
    "/etc/hostname":                        "[FCAT_CONFIG]",
    "/etc/iptables/":                       "[FCAT_CONFIG]",
    "/etc/nftables":                        "[FCAT_CONFIG]",

    "/etc/default/":                        "[FCAT_CONFIG]",
    "/etc/sysctl":                          "[FCAT_CONFIG]",
    "/etc/environment":                     "[FCAT_CONFIG]",
    "/etc/profile":                         "[FCAT_CONFIG]",
    "/etc/login.defs":                      "[FCAT_CONFIG]",
    "/etc/locale":                          "[FCAT_CONFIG]",
    "/etc/timezone":                        "[FCAT_CONFIG]",
    "/etc/os-release":                      "[FCAT_CONFIG]",
    "/etc/lsb-release":                     "[FCAT_CONFIG]",
    "/etc/mime.types":                      "[FCAT_CONFIG]",
    "/etc/":                                "[FCAT_CONFIG]",

    # ── Logs ──
    "/var/log/nginx/":                      "[FCAT_LOG_WEB]",
    "/var/log/apache2/":                    "[FCAT_LOG_WEB]",
    "/var/log/httpd/":                      "[FCAT_LOG_WEB]",
    "/LogFiles/W3SVC":                      "[FCAT_LOG_WEB]",

    "/var/log/mysql/":                      "[FCAT_LOG_DB]",
    "/var/log/postgresql/":                 "[FCAT_LOG_DB]",
    "/var/log/redis/":                      "[FCAT_LOG_DB]",
    "/var/log/mongodb/":                    "[FCAT_LOG_DB]",

    "/var/log/auth":                        "[FCAT_LOG]",
    "/var/log/secure":                      "[FCAT_LOG]",
    "/var/log/btmp":                        "[FCAT_LOG]",
    "/var/log/wtmp":                        "[FCAT_LOG]",
    "/var/log/lastlog":                     "[FCAT_LOG]",
    "/var/log/faillog":                     "[FCAT_LOG]",
    "/var/log/tallylog":                    "[FCAT_LOG]",
    "/var/run/utmp":                        "[FCAT_LOG]",
    "/var/log/fail2ban":                    "[FCAT_LOG]",

    "/var/log/mail":                        "[FCAT_LOG]",

    "/var/log/journal/":                    "[FCAT_LOG]",
    "/winevt/Logs/":                        "[FCAT_LOG]",

    "/var/log/audit/":                      "[FCAT_LOG_AUDIT]",
    "/var/log/clamav/":                     "[FCAT_LOG_AUDIT]",
    "Defender/Scans/":                      "[FCAT_LOG_AUDIT]",

    "/var/log/syslog":                      "[FCAT_LOG]",
    "/var/log/messages":                    "[FCAT_LOG]",
    "/var/log/daemon":                      "[FCAT_LOG]",
    "/var/log/kern":                        "[FCAT_LOG]",
    "/var/log/cron":                        "[FCAT_LOG]",
    "/var/log/dmesg":                       "[FCAT_LOG]",
    "/var/log/boot":                        "[FCAT_LOG]",
    "/var/log/dpkg":                        "[FCAT_LOG]",
    "/var/log/apt/":                        "[FCAT_LOG]",
    "/var/log/yum":                         "[FCAT_LOG]",
    "/var/log/dnf":                         "[FCAT_LOG]",
    "/Logs/CBS/":                           "[FCAT_LOG]",
    "/var/log/":                            "[FCAT_LOG]",

    # ── Data / state ──
    "/var/lib/mysql/":                      "[FCAT_DATA]",
    "/var/lib/postgresql/":                 "[FCAT_DATA]",
    "/var/lib/redis/":                      "[FCAT_DATA]",
    "/var/lib/mongodb/":                    "[FCAT_DATA]",
    "/var/lib/cassandra/":                  "[FCAT_DATA]",

    "/var/lib/docker/":                     "[FCAT_DATA_CONTAINER]",
    "/var/lib/containerd/":                 "[FCAT_DATA_CONTAINER]",
    "/var/lib/kubelet/":                    "[FCAT_DATA_CONTAINER]",

    "/var/lib/dpkg/":                       "[FCAT_DATA_PKG]",
    "/var/lib/apt/":                        "[FCAT_DATA_PKG]",
    "/var/lib/rpm/":                        "[FCAT_DATA_PKG]",
    "/var/cache/apt/":                      "[FCAT_DATA_PKG]",
    "/var/cache/yum/":                      "[FCAT_DATA_PKG]",
    "/var/cache/dnf/":                      "[FCAT_DATA_PKG]",
    "/var/cache/pacman/":                   "[FCAT_DATA_PKG]",
    "/SoftwareDistribution/":               "[FCAT_DATA_PKG]",
    "/chocolatey/":                         "[FCAT_DATA_PKG]",
    "/node_modules/":                       "[FCAT_DATA_PKG]",
    "/site-packages/":                      "[FCAT_DATA_PKG]",
    "/dist-packages/":                      "[FCAT_DATA_PKG]",

    "/var/spool/postfix/":                  "[FCAT_DATA_MAIL]",
    "/var/spool/mail/":                     "[FCAT_DATA_MAIL]",
    "/var/mail/":                           "[FCAT_DATA_MAIL]",
    "/ImapMail/":                           "[FCAT_DATA_MAIL]",

    "/var/spool/cron/":                     "[FCAT_DATA]",

    "/var/lib/prometheus/":                 "[FCAT_DATA_MONITORING]",
    "/var/lib/grafana/":                    "[FCAT_DATA_MONITORING]",

    "/var/lib/ntp/":                        "[FCAT_DATA_NTP]",
    "/var/lib/chrony/":                     "[FCAT_DATA_NTP]",
    "/var/lib/dhcp/":                       "[FCAT_DATA]",

    "/var/lib/sss/":                        "[CAT_SYSADMIN]",
    "/var/run/faillock/":                   "[CAT_SYSADMIN]",
    "/var/db/nscd/":                        "[CAT_SYSADMIN]",

    "/var/lib/letsencrypt/":                "[FCAT_DATA]",
    "/etc/letsencrypt/":                    "[FCAT_DATA]",
    "/etc/ssl/":                            "[FCAT_DATA]",
    "/etc/pki/":                            "[FCAT_DATA]",
    "/Crypto/":                             "[FCAT_DATA]",
    "/SystemCertificates/":                 "[FCAT_DATA]",

    # ── Credentials ──
    "/etc/shadow":                          "[CAT_SYSADMIN]",
    "/etc/passwd":                          "[CAT_SYSADMIN]",
    "/etc/gshadow":                         "[CAT_SYSADMIN]",
    "/etc/group":                           "[CAT_SYSADMIN]",
    "/system32/config/SAM":                 "[CAT_SYSADMIN]",
    "/system32/config/SECURITY":            "[CAT_SYSADMIN]",
    "/system32/config/SYSTEM":              "[CAT_SYSADMIN]",
    "/NTDS/ntds.dit":                       "[CAT_SYSADMIN]",
    "/krb5cc_":                             "[CAT_SYSADMIN]",
    "/keytab":                              "[CAT_SYSADMIN]",
    "/.aws/credentials":                    "[CAT_SYSADMIN]",
    "/.git-credentials":                    "[CAT_SYSADMIN]",

    # ── Binaries ──
    "/usr/bin/":                            "[CAT_EXECUTABLE]",
    "/usr/sbin/":                           "[CAT_EXECUTABLE]",
    "/usr/local/bin/":                      "[CAT_EXECUTABLE]",
    "/usr/local/sbin/":                     "[CAT_EXECUTABLE]",
    "/bin/":                                "[CAT_EXECUTABLE]",
    "/sbin/":                               "[CAT_EXECUTABLE]",
    # ── Snap packages ──
    # /snap/<app>/<rev>/ is the read-only mount of bundled package contents.
    # Sub-paths mirror a full filesystem, so classify by internal structure.
    "/snap/firefox/":                       "[FCAT_BROWSER_DATA]",
    "/snap/chromium/":                      "[FCAT_BROWSER_DATA]",
    "/snap/":                               "[CAT_EXECUTABLE]",
    # /var/snap/<app>/common/ is mutable persistent app data (like /var/lib/).
    "/var/snap/firefox/":                   "[FCAT_BROWSER_DATA]",
    "/var/snap/chromium/":                  "[FCAT_BROWSER_DATA]",
    "/var/snap/cups/":                      "[FCAT_SPOOL]",
    "/var/snap/":                           "[FCAT_APP_DATA]",
    "/system/bin/":                         "[CAT_EXECUTABLE]",
    "/system32/":                           "[CAT_EXECUTABLE]",
    "/SysWOW64/":                           "[CAT_EXECUTABLE]",

    # ── Libraries ──
    "/usr/lib/":                            "[FCAT_LIBRARY]",
    "/lib/":                                "[FCAT_LIBRARY]",
    "/lib64/":                              "[FCAT_LIBRARY]",
    "/lib/x86_64-linux-gnu/":              "[FCAT_LIBRARY]",
    "/system/lib/":                         "[FCAT_LIBRARY]",
    "/system/lib64/":                       "[FCAT_LIBRARY]",
    "/system/framework/":                   "[FCAT_LIBRARY]",
    "/System32/drivers/":                   "[FCAT_LIBRARY]",

    # ── User directories ──
    "/.bashrc":                             "[FCAT_SHELLRC]",
    "/.bash_profile":                       "[FCAT_SHELLRC]",
    "/.profile":                            "[FCAT_SHELLRC]",
    "/.zshrc":                              "[FCAT_SHELLRC]",
    "/.bash_history":                       "[FCAT_SHELLRC]",
    "/.zsh_history":                        "[FCAT_SHELLRC]",

    "/.config/":                            "[FCAT_CONFIG]",
    "/AppData/Roaming/":                    "[FCAT_CONFIG]",
    "/.local/share/":                       "[FCAT_CONFIG]",

    "/.cache/":                             "[FCAT_USER_CACHE]",
    "/AppData/Local/":                      "[FCAT_USER_CACHE]",

    "/Documents/":                          "[FCAT_DOCUMENT]",
    "/Downloads/":                          "[FCAT_USER_DOWNLOADS]",
    "/Desktop/":                            "[FCAT_USER_DESKTOP]",
    "/Pictures/":                           "[FCAT_USER_MEDIA]",
    "/Music/":                              "[FCAT_USER_MEDIA]",
    "/Videos/":                             "[FCAT_USER_MEDIA]",

    "/home/":                               "[FCAT_HOMEDIR]",
    "/Users/":                              "[FCAT_HOMEDIR]",
    "/root/":                               "[FCAT_HOMEDIR]",

    # ── Temp ──
    "/tmp/":                                "[FCAT_TMP]",
    "/var/tmp/":                            "[FCAT_TMP]",
    "/dev/shm/":                            "[FCAT_TMP]",
    "/AppData/Local/Temp/":                 "[FCAT_TMP]",
    "/Windows/Temp/":                       "[FCAT_TMP]",

    # ── Devices ──
    "/dev/pts/":                            "[FCAT_SYSFS]",
    "/dev/tty":                             "[FCAT_SYSFS]",
    "/dev/console":                         "[FCAT_SYSFS]",
    "/dev/stdin":                           "[FCAT_PIPE]",
    "/dev/stdout":                          "[FCAT_PIPE]",
    "/dev/stderr":                          "[FCAT_PIPE]",
    "/dev/null":                            "[FCAT_SYSFS]",
    "/dev/zero":                            "[FCAT_SYSFS]",
    "/dev/urandom":                         "[FCAT_SYSFS]",
    "/dev/random":                          "[FCAT_SYSFS]",
    "/dev/log":                             "[FCAT_TMP]",
    "/dev/binder":                          "[FCAT_TMP]",
    "/dev/hwbinder":                        "[FCAT_TMP]",
    "/dev/vndbinder":                       "[FCAT_TMP]",
    "/run/dbus/":                           "[FCAT_TMP]",
    "/var/run/dbus/":                       "[FCAT_TMP]",

    # ── Virtual filesystems ──
    "/proc/":                               "[FCAT_PROCFS]",
    "/sys/":                                "[FCAT_SYSFS]",

    # ── Runtime / PID files ──
    "/var/run/":                            "[FCAT_RUNTIME]",
    "/run/":                                "[FCAT_RUNTIME]",

    # ── App data (Android) ──
    "/data/data/":                          "[FCAT_APP_DATA]",
    "/data/app/":                           "[FCAT_APP_DATA]",
    "/sdcard/":                             "[FCAT_APP_DATA]",
    "/dalvik-cache/":                       "[FCAT_APP_DATA]",

    # ── Windows program directories ──
    "/ProgramData/":                        "[FCAT_PROGDATA]",
    "/Program Files/":                      "[FCAT_PROGDIR]",
    "/Program Files (x86)/":                "[FCAT_PROGDIR]",
    "/WindowsApps/":                        "[FCAT_PROGDIR]",

    # ── Windows specific ──
    "/Windows/Prefetch/":                   "[FCAT_PREFETCH]",
    "/Windows/SoftwareDistribution/":       "[FCAT_DATA_PKG]",
    "/$Recycle.Bin/":                       "[FCAT_RECYCLE]",
    "/Windows/Minidump/":                   "[FCAT_CRASHDUMP]",
    "/Windows/wdi/":                        "[CAT_SYSADMIN]",
    "/SRU/":                                "[CAT_SYSADMIN]",

    # ── Git ──
    "/.git/":                               "[FCAT_VCS]",
    "/.gitconfig":                          "[FCAT_VCS]",

    # ── Browser data ──
    "/.mozilla/":                           "[FCAT_BROWSER_DATA]",
    "/Google/Chrome/":                      "[FCAT_BROWSER_DATA]",
    "/chromium/":                           "[FCAT_BROWSER_DATA]",
    "/Microsoft/Edge/":                     "[FCAT_BROWSER_DATA]",
    "/Internet Explorer/":                  "[FCAT_BROWSER_DATA]",

    # ── Spool (print, etc.) ──
    "/var/spool/cups/":                     "[FCAT_SPOOL]",
    "/spool/PRINTERS/":                     "[FCAT_SPOOL]",
    "/var/spool/":                          "[FCAT_SPOOL]",

    # ── Catch-all for known structures ──
    "/opt/":                                "[FCAT_HOMEDIR]",
    "/srv/":                                "[FCAT_HOMEDIR]",
    "/mnt/":                                "[FCAT_MOUNT]",
    "/media/":                              "[FCAT_MOUNT]",
    "/boot/":                               "[FCAT_SYSFS]",
}


# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED LOOKUP STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

# FILE_DIR_CATEGORIES sorted by key length descending for most-specific-first matching
_FILE_DIR_SORTED = sorted(FILE_DIR_CATEGORIES.items(), key=lambda x: len(x[0]), reverse=True)

ALL_PROC_CATEGORY_TOKENS = sorted(set(PROC_CATEGORIES.values()))
ALL_FILE_CATEGORY_TOKENS = sorted(set(FILE_DIR_CATEGORIES.values()))
ALL_CATEGORY_TOKENS = sorted(set(ALL_PROC_CATEGORY_TOKENS + ALL_FILE_CATEGORY_TOKENS))


def match_proc_category(label_str: str) -> "Optional[str]":
    """Extract binary name from a process command line and look up its category.

    Extraction procedure:
      1. Take the first whitespace-delimited token (the binary/command)
      2. Strip any path prefix: /usr/sbin/nginx → nginx
      3. Strip .exe suffix: svchost.exe → svchost
      4. Look up in PROC_CATEGORIES

    Returns the category token string or None if not found.
    """
    parts = label_str.split()
    # Strip leading "subject" type prefix if present
    if parts and parts[0] == "subject":
        parts = parts[1:]
    if not parts:
        return None

    binary = parts[0]
    # Strip path: /usr/sbin/nginx → nginx, C:\Windows\system32\svchost.exe → svchost.exe
    binary = binary.replace("\\", "/")
    if "/" in binary:
        binary = binary.rsplit("/", 1)[-1]
    # Strip .exe suffix
    if binary.lower().endswith(".exe"):
        binary = binary[:-4]

    # Direct lookup (case-sensitive first, then case-insensitive for Windows)
    if binary in PROC_CATEGORIES:
        return PROC_CATEGORIES[binary]
    binary_lower = binary.lower()
    for name, cat in PROC_CATEGORIES.items():
        if name.lower() == binary_lower:
            return cat
    return None


def match_file_category(label_str: str) -> "Optional[str]":
    """Match a file path against FILE_DIR_CATEGORIES (most-specific-first).

    The path is normalized (backslashes → forward slashes) before matching.
    Returns the category token string or None if not found.
    """
    parts = label_str.split()
    # Strip leading "file" type prefix if present
    if parts and parts[0] == "file":
        parts = parts[1:]
    if not parts:
        return None

    # Use the first part as the path (file labels are typically a single path)
    path = parts[0].replace("\\", "/")

    for pattern, category in _FILE_DIR_SORTED:
        if pattern in path:
            return category
    return None


if __name__ == "__main__":
    print(f"Process categories: {len(PROC_CATEGORIES)} binary names → {len(ALL_PROC_CATEGORY_TOKENS)} tokens")
    print(f"File categories:    {len(FILE_DIR_CATEGORIES)} path patterns → {len(ALL_FILE_CATEGORY_TOKENS)} tokens")
    print(f"Total new tokens:   {len(ALL_CATEGORY_TOKENS)}")
    print()
    print("── PROC category tokens ──")
    for t in ALL_PROC_CATEGORY_TOKENS:
        count = sum(1 for v in PROC_CATEGORIES.values() if v == t)
        print(f"  {t:30s}  ({count} binaries)")
    print()
    print("── FILE category tokens ──")
    for t in ALL_FILE_CATEGORY_TOKENS:
        count = sum(1 for v in FILE_DIR_CATEGORIES.values() if v == t)
        print(f"  {t:30s}  ({count} patterns)")
