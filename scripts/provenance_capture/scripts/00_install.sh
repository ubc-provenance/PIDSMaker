#!/bin/bash
# ============================================================================
# DOMAIN 00: Install all tools needed by subsequent domains
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "00 — INSTALL TOOLS"

export DEBIAN_FRONTEND=noninteractive

section "System packages"
run "apt update" apt-get update -qq

run "install coreutils" apt-get install -y -qq \
    coreutils findutils grep sed gawk mawk procps psmisc \
    util-linux bsdutils file pv tree rename dos2unix bc dc \
    lsof strace ltrace time moreutils parallel xxd bsdmainutils

run "install network tools" apt-get install -y -qq \
    curl wget netcat-openbsd socat nmap tcpdump tshark \
    dnsutils net-tools iproute2 iputils-ping iputils-tracepath traceroute mtr-tiny \
    openssh-client openssh-server iperf3 whois telnet ftp lftp rsync \
    httpie aria2 hping3 2>/dev/null

run "install security tools" apt-get install -y -qq \
    openssl gnupg2 ca-certificates pass \
    nikto hydra john \
    binutils elfutils checksec 2>/dev/null

run "install compilers" apt-get install -y -qq \
    build-essential gcc g++ clang nasm yasm \
    make cmake ninja-build autoconf automake libtool pkg-config \
    gdb valgrind binutils-dev flex bison

run "install java" apt-get install -y -qq default-jdk default-jre maven ant 2>/dev/null

run "install golang" apt-get install -y -qq golang-go 2>/dev/null

run "install nodejs" apt-get install -y -qq nodejs npm 2>/dev/null

run "install ruby" apt-get install -y -qq ruby ruby-dev 2>/dev/null

run "install perl" apt-get install -y -qq perl libio-socket-ssl-perl libnet-ssleay-perl 2>/dev/null

run "install php" apt-get install -y -qq php-cli php-curl php-mbstring php-xml php-json 2>/dev/null

run "install lua" apt-get install -y -qq lua5.4 2>/dev/null

run "install databases" apt-get install -y -qq \
    sqlite3 libsqlite3-dev \
    postgresql-client mysql-client redis-tools 2>/dev/null

run "install web servers" apt-get install -y -qq nginx apache2 2>/dev/null

run "install text tools" apt-get install -y -qq \
    jq xmlstarlet pandoc groff ghostscript \
    csvtool colordiff wdiff 2>/dev/null

run "install media" apt-get install -y -qq imagemagick ffmpeg exiftool 2>/dev/null

run "install vcs" apt-get install -y -qq git subversion 2>/dev/null

run "install archive" apt-get install -y -qq \
    p7zip-full xz-utils lz4 zstd brotli 2>/dev/null

run "install monitoring" apt-get install -y -qq sysstat dstat atop 2>/dev/null

run "install misc" apt-get install -y -qq \
    expect screen tmux cron at acl attr \
    fakeroot faketime inotify-tools 2>/dev/null

section "Rust toolchain"
if ! command -v rustc &>/dev/null; then
    run "install rustup" bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null'
fi
export PATH="$HOME/.cargo/bin:$PATH"

section "Python packages"
run "pip core" pip install -q numpy pandas scipy scikit-learn matplotlib 2>/dev/null
run "pip web" pip install -q requests flask django fastapi uvicorn httpx aiohttp beautifulsoup4 2>/dev/null
run "pip data" pip install -q sqlalchemy psycopg2-binary pymysql redis pymongo 2>/dev/null
run "pip crypto" pip install -q cryptography paramiko pycryptodome 2>/dev/null
run "pip system" pip install -q psutil watchdog fabric invoke 2>/dev/null
run "pip misc" pip install -q pyyaml toml lxml pillow pexpect 2>/dev/null

section "NPM packages"
run "npm global" npm install -g express http-server typescript nodemon 2>/dev/null

domain_end
