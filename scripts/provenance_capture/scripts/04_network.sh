#!/bin/bash
# ============================================================================
# DOMAIN 04: NETWORK ACTIVITY — Exhaustive coverage
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "04 — NETWORK"

S=$(generate_sample_files)

# ============================================================================
section "dig — DNS queries (every record type, every flag)"
# ============================================================================
for target in google.com github.com example.com cloudflare.com amazon.com; do
    run "dig A $target" dig $target A +short 2>/dev/null
    run "dig AAAA $target" dig $target AAAA +short 2>/dev/null
done
run "dig MX gmail.com" dig gmail.com MX +short 2>/dev/null
run "dig MX yahoo.com" dig yahoo.com MX +short 2>/dev/null
run "dig MX microsoft.com" dig microsoft.com MX +short 2>/dev/null
run "dig TXT google.com" dig google.com TXT +short 2>/dev/null
run "dig TXT _dmarc.google.com" dig _dmarc.google.com TXT +short 2>/dev/null
run "dig NS google.com" dig google.com NS +short 2>/dev/null
run "dig NS com." dig com. NS +short 2>/dev/null
run "dig SOA google.com" dig google.com SOA +short 2>/dev/null
run "dig CNAME www.google.com" dig www.google.com CNAME +short 2>/dev/null
run "dig PTR 8.8.8.8" dig -x 8.8.8.8 +short 2>/dev/null
run "dig PTR 1.1.1.1" dig -x 1.1.1.1 +short 2>/dev/null
run "dig ANY google.com" dig google.com ANY +short 2>/dev/null
run "dig SRV" dig _http._tcp.google.com SRV +short 2>/dev/null
run "dig CAA" dig google.com CAA +short 2>/dev/null
run "dig @8.8.8.8" dig @8.8.8.8 example.com A +short 2>/dev/null
run "dig @1.1.1.1" dig @1.1.1.1 example.com A +short 2>/dev/null
run "dig @9.9.9.9" dig @9.9.9.9 example.com A +short 2>/dev/null
run "dig +tcp" dig +tcp example.com 2>/dev/null
run "dig +trace" dig +trace example.com 2>/dev/null | tail -10
run "dig +dnssec" dig +dnssec cloudflare.com 2>/dev/null | head -20
run "dig +norecurse" dig +norecurse example.com @8.8.8.8 2>/dev/null
run "dig +noadditional" dig +noadditional google.com 2>/dev/null | head -10
run "dig +multiline" dig +multiline google.com SOA 2>/dev/null
run "dig +stats" dig +stats example.com 2>/dev/null | tail -5
run "dig +short" dig +short example.com 2>/dev/null
run "dig -t AXFR" dig -t AXFR example.com @8.8.8.8 2>/dev/null || true
run "dig -b source" dig -b 0.0.0.0 example.com 2>/dev/null

run "nslookup" nslookup example.com 2>/dev/null
run "nslookup 8.8.8.8" nslookup example.com 8.8.8.8 2>/dev/null
run "nslookup -type=mx" nslookup -type=mx gmail.com 2>/dev/null
run "nslookup -type=ns" nslookup -type=ns google.com 2>/dev/null
run "nslookup -type=soa" nslookup -type=soa google.com 2>/dev/null
run "nslookup -type=txt" nslookup -type=txt google.com 2>/dev/null
run "host" host example.com 2>/dev/null
run "host -t MX" host -t MX gmail.com 2>/dev/null
run "host -t NS" host -t NS google.com 2>/dev/null
run "host -t AAAA" host -t AAAA google.com 2>/dev/null
run "host -a" host -a example.com 2>/dev/null
run "host -v" host -v example.com 2>/dev/null
run "host reverse" host 8.8.8.8 2>/dev/null

# ============================================================================
section "curl — HTTP client (exhaustive flags)"
# ============================================================================
TARGETS=(
    "https://httpbin.org/get"
    "https://httpbin.org/post"
    "https://example.com"
    "https://httpbin.org/ip"
    "https://httpbin.org/headers"
    "https://httpbin.org/user-agent"
    "http://httpbin.org/get"
)

# Methods
run "curl GET" curl -s -o /dev/null -w "%{http_code}" https://httpbin.org/get
run "curl HEAD" curl -s -I https://example.com
run "curl POST form" curl -s -X POST -d "key=value&other=data" https://httpbin.org/post -o /dev/null
run "curl POST json" curl -s -X POST -H "Content-Type: application/json" -d '{"key":"value","nested":{"a":1}}' https://httpbin.org/post -o /dev/null
run "curl PUT" curl -s -X PUT -d "update data" https://httpbin.org/put -o /dev/null
run "curl DELETE" curl -s -X DELETE https://httpbin.org/delete -o /dev/null
run "curl PATCH" curl -s -X PATCH -d "patch data" https://httpbin.org/patch -o /dev/null
run "curl OPTIONS" curl -s -X OPTIONS -I https://httpbin.org/get 2>/dev/null

# Headers
run "curl -H custom" curl -s -H "X-Custom-Header: provenance-test" https://httpbin.org/headers -o /dev/null
run "curl -H accept" curl -s -H "Accept: application/json" https://httpbin.org/get -o /dev/null
run "curl -H auth" curl -s -H "Authorization: Bearer fake_token_123" https://httpbin.org/headers -o /dev/null
run "curl -H multi" curl -s -H "X-A: 1" -H "X-B: 2" -H "X-C: 3" https://httpbin.org/headers -o /dev/null

# Auth
run "curl basic auth" curl -s -u user:pass https://httpbin.org/basic-auth/user/pass -o /dev/null
run "curl basic auth fail" curl -s -u wrong:wrong https://httpbin.org/basic-auth/user/pass -o /dev/null || true
run "curl bearer" curl -s -H "Authorization: Bearer test123" https://httpbin.org/bearer -o /dev/null 2>/dev/null

# Cookies
run "curl send cookie" curl -s -b "session=abc123; user=test" https://httpbin.org/cookies -o /dev/null
run "curl cookie jar" curl -s -c "$S/cookies.txt" https://httpbin.org/cookies/set?prov=test -o /dev/null -L
run "curl use cookie jar" curl -s -b "$S/cookies.txt" https://httpbin.org/cookies -o /dev/null

# Redirect
run "curl -L follow" curl -s -L -o /dev/null https://httpbin.org/redirect/3
run "curl -L max" curl -s -L --max-redirs 2 https://httpbin.org/redirect/5 -o /dev/null 2>/dev/null || true
run "curl no redirect" curl -s -o /dev/null -w "%{http_code}" https://httpbin.org/redirect/1

# Timeouts
run "curl --connect-timeout" curl -s --connect-timeout 5 https://example.com -o /dev/null
run "curl --max-time" curl -s --max-time 10 https://example.com -o /dev/null
run "curl timeout fail" curl -s --connect-timeout 1 --max-time 2 https://10.255.255.1/ 2>/dev/null || true

# Output
run "curl -o file" curl -s -o "$S/curl_out.html" https://example.com
run "curl -O" cd "$S" && curl -s -O https://example.com/index.html 2>/dev/null; cd /
run "curl -w format" curl -s -o /dev/null -w "code=%{http_code} size=%{size_download} time=%{time_total}s ip=%{remote_ip}\n" https://example.com
run "curl -v verbose" curl -v -s -o /dev/null https://example.com 2>/dev/null
run "curl -D headers" curl -s -D "$S/headers.txt" -o /dev/null https://example.com
run "curl --trace" curl -s --trace "$S/trace.txt" -o /dev/null https://example.com 2>/dev/null
run "curl --trace-ascii" curl -s --trace-ascii "$S/trace_ascii.txt" -o /dev/null https://example.com

# User agent
run "curl -A chrome" curl -s -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0" https://httpbin.org/user-agent -o /dev/null
run "curl -A firefox" curl -s -A "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0" https://httpbin.org/user-agent -o /dev/null
run "curl -A curl" curl -s -A "curl/8.0" https://httpbin.org/user-agent -o /dev/null
run "curl -A bot" curl -s -A "ProvenanceBot/1.0" https://httpbin.org/user-agent -o /dev/null

# TLS/SSL
run "curl --tls-max 1.2" curl -s --tls-max 1.2 -o /dev/null https://example.com 2>/dev/null
run "curl --tlsv1.3" curl -s --tlsv1.3 -o /dev/null https://example.com 2>/dev/null
run "curl -k insecure" curl -s -k https://self-signed.badssl.com/ -o /dev/null 2>/dev/null
run "curl --cert-status" curl -s --cert-status https://example.com -o /dev/null 2>/dev/null || true
run "curl --ciphers" curl -s --ciphers 'ECDHE-RSA-AES256-GCM-SHA384' https://example.com -o /dev/null 2>/dev/null || true

# Protocol
run "curl --http1.1" curl -s --http1.1 -o /dev/null https://example.com
run "curl --http2" curl -s --http2 -o /dev/null https://example.com 2>/dev/null
run "curl -4 ipv4" curl -s -4 -o /dev/null https://example.com
run "curl -6 ipv6" curl -s -6 -o /dev/null https://example.com 2>/dev/null || true
run "curl --compressed" curl -s --compressed -o /dev/null https://example.com

# Upload
echo "upload content" > "$S/upload.txt"
run "curl -F file upload" curl -s -F "file=@$S/upload.txt" https://httpbin.org/post -o /dev/null
run "curl -T PUT upload" curl -s -T "$S/upload.txt" https://httpbin.org/put -o /dev/null
run "curl --data-binary" curl -s --data-binary @"$S/upload.txt" https://httpbin.org/post -o /dev/null
run "curl --data-urlencode" curl -s --data-urlencode "content@$S/upload.txt" https://httpbin.org/post -o /dev/null

# Range
run "curl --range" curl -s --range 0-99 -o /dev/null https://example.com
run "curl -C resume" curl -s -C - -o /dev/null https://example.com 2>/dev/null || true

# Multiple URLs
run "curl multi" curl -s -o /dev/null https://example.com -o /dev/null https://httpbin.org/get

# ============================================================================
section "wget — file downloader"
# ============================================================================
run "wget basic" wget -q -O /dev/null https://example.com
run "wget -O file" wget -q -O "$S/wget_out.html" https://example.com
run "wget -P dir" wget -q -P "$S/" https://example.com/index.html 2>/dev/null || true
run "wget --spider" wget -q --spider https://example.com
run "wget -N timestamp" wget -q -N -P "$S/" https://example.com 2>/dev/null || true
run "wget -c continue" wget -q -c -O "$S/wget_cont.html" https://example.com 2>/dev/null || true
run "wget -U agent" wget -q -U "ProvenanceBot/1.0" -O /dev/null https://example.com
run "wget --header" wget -q --header="X-Test: prov" -O /dev/null https://example.com
run "wget --no-check-certificate" wget -q --no-check-certificate -O /dev/null https://self-signed.badssl.com/ 2>/dev/null || true
run "wget --timeout" wget -q --timeout=5 -O /dev/null https://example.com
run "wget --tries" wget -q --tries=2 -O /dev/null https://example.com
run "wget --limit-rate" wget -q --limit-rate=100k -O /dev/null https://example.com
run "wget -r recursive" wget -q -r -l 1 -P "$S/wget_r/" --no-parent --reject="*.css,*.js,*.png,*.jpg" https://example.com 2>/dev/null || true
run "wget --mirror" wget -q --mirror -l 1 -P "$S/wget_m/" https://example.com 2>/dev/null || true
run "wget --convert-links" wget -q -p --convert-links -P "$S/wget_c/" https://example.com 2>/dev/null || true
run "wget --post-data" wget -q --post-data="key=value" -O /dev/null https://httpbin.org/post 2>/dev/null || true
run "wget --referer" wget -q --referer="https://google.com" -O /dev/null https://example.com
run "wget --max-redirect" wget -q --max-redirect=3 -O /dev/null https://httpbin.org/redirect/5 2>/dev/null || true
run "wget -q quiet" wget -q -O /dev/null https://example.com
run "wget -v verbose" wget -v -O /dev/null https://example.com 2>/dev/null
rm -rf "$S/wget_r" "$S/wget_m" "$S/wget_c"

# ============================================================================
section "ping / traceroute / mtr"
# ============================================================================
run "ping -c 1" ping -c 1 -W 2 8.8.8.8
run "ping -c 3" ping -c 3 -W 2 8.8.8.8
run "ping -c 1 -s 64" ping -c 1 -W 2 -s 64 8.8.8.8
run "ping -c 1 -s 1400" ping -c 1 -W 2 -s 1400 8.8.8.8
run "ping -c 1 -t 10" ping -c 1 -W 2 -t 10 8.8.8.8
run "ping -c 1 -q" ping -c 1 -W 2 -q 8.8.8.8
run "ping -c 1 -n" ping -c 1 -W 2 -n 8.8.8.8
run "ping -c 1 localhost" ping -c 1 -W 2 127.0.0.1
run "ping -c 1 1.1.1.1" ping -c 1 -W 2 1.1.1.1
run "ping -c 1 9.9.9.9" ping -c 1 -W 2 9.9.9.9
run "ping -I" ping -c 1 -W 2 -I lo 127.0.0.1 2>/dev/null || true
run "ping6 localhost" ping -c 1 -W 2 ::1 2>/dev/null || true

run_t 15 "traceroute" traceroute -m 5 8.8.8.8 2>/dev/null
run_t 15 "traceroute -n" traceroute -n -m 5 8.8.8.8 2>/dev/null
run_t 15 "traceroute -I" traceroute -I -m 3 8.8.8.8 2>/dev/null
run_t 15 "traceroute -T" traceroute -T -m 3 8.8.8.8 2>/dev/null
run_t 10 "mtr report" mtr -c 3 -r 8.8.8.8 2>/dev/null
run_t 10 "mtr -n" mtr -c 3 -r -n 8.8.8.8 2>/dev/null

# ============================================================================
section "SSH tools"
# ============================================================================
run "ssh-keygen rsa 2048" ssh-keygen -t rsa -b 2048 -f "$S/k_rsa2048" -N "" -q
run "ssh-keygen rsa 4096" ssh-keygen -t rsa -b 4096 -f "$S/k_rsa4096" -N "" -q
run "ssh-keygen ecdsa 256" ssh-keygen -t ecdsa -b 256 -f "$S/k_ecdsa256" -N "" -q
run "ssh-keygen ecdsa 384" ssh-keygen -t ecdsa -b 384 -f "$S/k_ecdsa384" -N "" -q
run "ssh-keygen ecdsa 521" ssh-keygen -t ecdsa -b 521 -f "$S/k_ecdsa521" -N "" -q
run "ssh-keygen ed25519" ssh-keygen -t ed25519 -f "$S/k_ed25519" -N "" -q
run "ssh-keygen dsa" ssh-keygen -t dsa -f "$S/k_dsa" -N "" -q 2>/dev/null || true
run "ssh-keygen ed25519 comment" ssh-keygen -t ed25519 -f "$S/k_ed_c" -N "" -C "provenance@test" -q
run "ssh-keygen passphrase" ssh-keygen -t ed25519 -f "$S/k_ed_pass" -N "testpass123" -q
run "ssh-keygen fingerprint" ssh-keygen -l -f "$S/k_rsa2048.pub"
run "ssh-keygen fingerprint sha256" ssh-keygen -l -E sha256 -f "$S/k_rsa2048.pub"
run "ssh-keygen fingerprint md5" ssh-keygen -l -E md5 -f "$S/k_rsa2048.pub"
run "ssh-keygen visual" ssh-keygen -lv -f "$S/k_ed25519.pub"
run "ssh-keygen change comment" ssh-keygen -c -C "new_comment" -f "$S/k_ed25519" -N "" 2>/dev/null || true
run "ssh-keygen pubkey extract" ssh-keygen -y -f "$S/k_rsa2048" 2>/dev/null
run "ssh-keygen convert PEM" ssh-keygen -e -m PEM -f "$S/k_rsa2048.pub" 2>/dev/null
run "ssh-keygen convert PKCS8" ssh-keygen -e -m PKCS8 -f "$S/k_rsa2048.pub" 2>/dev/null
run "ssh-keygen convert RFC4716" ssh-keygen -e -m RFC4716 -f "$S/k_rsa2048.pub" 2>/dev/null

run "ssh-keyscan localhost" ssh-keyscan -T 3 localhost 2>/dev/null
run "ssh-keyscan localhost -t rsa" ssh-keyscan -T 3 -t rsa localhost 2>/dev/null
run "ssh-keyscan localhost -t ed25519" ssh-keyscan -T 3 -t ed25519 localhost 2>/dev/null
run "ssh-keyscan github" ssh-keyscan -T 5 github.com 2>/dev/null
run "ssh-keyscan github -t rsa" ssh-keyscan -T 5 -t rsa github.com 2>/dev/null
run "ssh-keyscan gitlab" ssh-keyscan -T 5 gitlab.com 2>/dev/null
run "ssh-keyscan -H hashed" ssh-keyscan -T 3 -H localhost 2>/dev/null

run "ssh-agent start" eval $(ssh-agent -s) 2>/dev/null
run "ssh-add" ssh-add "$S/k_ed25519" 2>/dev/null || true
run "ssh-add -l list" ssh-add -l 2>/dev/null || true
run "ssh-add -L public" ssh-add -L 2>/dev/null || true
run "ssh-add -d delete" ssh-add -d "$S/k_ed25519" 2>/dev/null || true
run "ssh-add -D delete all" ssh-add -D 2>/dev/null || true
run "ssh-agent kill" ssh-agent -k 2>/dev/null || true

rm -f "$S"/k_*

# ============================================================================
section "netcat / socat"
# ============================================================================
# TCP
run_t 5 "nc tcp send" bash -c 'nc -l -p 19001 &>/dev/null & sleep 0.3; echo "hello" | nc -w 1 127.0.0.1 19001' 2>/dev/null
run_t 5 "nc tcp file transfer" bash -c 'nc -l -p 19002 > /tmp/nc_recv.txt &
    sleep 0.3; echo "transferred data" | nc -w 1 127.0.0.1 19002; cat /tmp/nc_recv.txt' 2>/dev/null
run "nc port check 22" nc -z -w 1 127.0.0.1 22 2>/dev/null || true
run "nc port check 80" nc -z -w 1 127.0.0.1 80 2>/dev/null || true
run "nc port check 443" nc -z -w 1 127.0.0.1 443 2>/dev/null || true
run "nc port range" nc -z -w 1 127.0.0.1 20-25 2>/dev/null || true
run "nc -v port" nc -v -z -w 1 127.0.0.1 22 2>/dev/null || true

# socat
run_t 5 "socat TCP echo" bash -c 'socat TCP-LISTEN:19003,reuseaddr,fork EXEC:cat &
    sleep 0.5; echo "socat tcp test" | socat - TCP:127.0.0.1:19003' 2>/dev/null
run_t 5 "socat UNIX" bash -c 'socat UNIX-LISTEN:/tmp/prov_socat.sock,fork EXEC:cat &
    sleep 0.5; echo "socat unix test" | socat - UNIX-CONNECT:/tmp/prov_socat.sock; rm -f /tmp/prov_socat.sock' 2>/dev/null
run_t 5 "socat UDP" bash -c 'socat UDP-LISTEN:19004,fork EXEC:cat &
    sleep 0.5; echo "socat udp test" | socat - UDP:127.0.0.1:19004' 2>/dev/null
run_t 5 "socat exec" echo "hello" | socat - EXEC:cat 2>/dev/null
run_t 5 "socat file" socat -u OPEN:"$S/tiny.txt",rdonly CREATE:/tmp/socat_copy.txt 2>/dev/null
rm -f /tmp/socat_copy.txt /tmp/nc_recv.txt

# ============================================================================
section "nmap — port scanner"
# ============================================================================
run_t 30 "nmap -sT TCP" nmap -sT -T4 -p 22,80,443,8080 127.0.0.1 2>/dev/null
run_t 30 "nmap -sS SYN" nmap -sS -T4 -p 22,80,443 127.0.0.1 2>/dev/null
run_t 30 "nmap -sU UDP" nmap -sU -T4 --top-ports 10 127.0.0.1 2>/dev/null
run_t 30 "nmap -sV version" nmap -sV -T4 -p 22 127.0.0.1 2>/dev/null
run_t 30 "nmap -O OS" nmap -O -T4 127.0.0.1 2>/dev/null
run_t 30 "nmap -A aggressive" nmap -A -T4 -p 22 127.0.0.1 2>/dev/null
run_t 30 "nmap -sC scripts" nmap -sC -T4 -p 22 127.0.0.1 2>/dev/null
run_t 30 "nmap --script vuln" nmap --script=vuln -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -sn ping" nmap -sn 127.0.0.1/24 2>/dev/null
run_t 60 "nmap full port" nmap -sT -T4 -p- 127.0.0.1 2>/dev/null
run_t 15 "nmap -sW window" nmap -sW -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -sF FIN" nmap -sF -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -sX Xmas" nmap -sX -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -sN NULL" nmap -sN -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -sA ACK" nmap -sA -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -f fragment" nmap -f -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap --data-length" nmap --data-length 50 -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -D decoy" nmap -D RND:5 -T4 -p 22 127.0.0.1 2>/dev/null
run_t 15 "nmap -oN output" nmap -sT -T4 -p 22 127.0.0.1 -oN "$S/nmap_normal.txt" 2>/dev/null
run_t 15 "nmap -oX xml" nmap -sT -T4 -p 22 127.0.0.1 -oX "$S/nmap.xml" 2>/dev/null
run_t 15 "nmap -oG grep" nmap -sT -T4 -p 22 127.0.0.1 -oG "$S/nmap_grep.txt" 2>/dev/null
run_t 15 "nmap --top-ports" nmap -sT -T4 --top-ports 100 127.0.0.1 2>/dev/null

# ============================================================================
section "ss / netstat / ip / lsof"
# ============================================================================
run "ss -t" ss -t
run "ss -u" ss -u
run "ss -l" ss -l
run "ss -a" ss -a | head -20
run "ss -tlnp" ss -tlnp
run "ss -ulnp" ss -ulnp
run "ss -anp" ss -anp | head -20
run "ss -s" ss -s
run "ss -4" ss -4 | head -10
run "ss -6" ss -6 | head -10
run "ss -e" ss -e | head -10
run "ss -i" ss -i | head -10
run "ss -o" ss -o | head -10
run "ss state established" ss state established | head -10
run "ss state listen" ss state listening
run "ss sport" ss -tlnp sport = :22
run "ss dport" ss -tnp dport = :443 | head -5

run "netstat -tlnp" netstat -tlnp 2>/dev/null
run "netstat -ulnp" netstat -ulnp 2>/dev/null
run "netstat -anp" netstat -anp 2>/dev/null | head -20
run "netstat -s" netstat -s 2>/dev/null | head -20
run "netstat -r" netstat -r 2>/dev/null
run "netstat -i" netstat -i 2>/dev/null

run "ip addr" ip addr show
run "ip -4 addr" ip -4 addr show
run "ip -6 addr" ip -6 addr show
run "ip route" ip route show
run "ip route default" ip route get 8.8.8.8
run "ip neigh" ip neigh show
run "ip link" ip link show
run "ip -s link" ip -s link show
run "ip -s -s link" ip -s -s link show
run "ip -j addr" ip -j addr show 2>/dev/null
run "ip rule" ip rule show
run "ip tunnel" ip tunnel show 2>/dev/null || true

run "lsof -i" lsof -i 2>/dev/null | head -15
run "lsof -i tcp" lsof -i tcp 2>/dev/null | head -10
run "lsof -i udp" lsof -i udp 2>/dev/null | head -10
run "lsof -i :22" lsof -i :22 2>/dev/null | head -5
run "lsof -i :80" lsof -i :80 2>/dev/null | head -5
run "lsof -P" lsof -P -i 2>/dev/null | head -10
run "lsof -n" lsof -n -i 2>/dev/null | head -10

# ============================================================================
section "tcpdump / tshark"
# ============================================================================
run_t 5 "tcpdump lo" tcpdump -i lo -c 5 -w "$S/lo.pcap" 2>/dev/null
run "tcpdump read" tcpdump -r "$S/lo.pcap" 2>/dev/null | head -5
run_t 5 "tcpdump -n" tcpdump -i lo -n -c 5 2>/dev/null
run_t 5 "tcpdump port 22" tcpdump -i lo -c 3 port 22 2>/dev/null || true
run_t 5 "tcpdump host" tcpdump -i lo -c 3 host 127.0.0.1 2>/dev/null
run_t 5 "tcpdump -A" tcpdump -i lo -A -c 3 2>/dev/null
run_t 5 "tcpdump -X" tcpdump -i lo -X -c 3 2>/dev/null
run_t 5 "tshark" tshark -i lo -c 5 2>/dev/null
run_t 5 "tshark fields" tshark -i lo -c 5 -T fields -e ip.src -e ip.dst -e tcp.port 2>/dev/null
rm -f "$S/lo.pcap"

# ============================================================================
section "rsync"
# ============================================================================
mkdir -p "$S/rsync_src" "$S/rsync_dst"
cp "$S/tiny.txt" "$S/small.txt" "$S/multiline.txt" "$S/rsync_src/"
run "rsync basic" rsync "$S/rsync_src/" "$S/rsync_dst/"
run "rsync -a" rsync -a "$S/rsync_src/" "$S/rsync_dst/"
run "rsync -av" rsync -av "$S/rsync_src/" "$S/rsync_dst/"
run "rsync -avz" rsync -avz "$S/rsync_src/" "$S/rsync_dst/"
run "rsync --delete" rsync -av --delete "$S/rsync_src/" "$S/rsync_dst/"
run "rsync -n dry-run" rsync -avn "$S/rsync_src/" "$S/rsync_dst/"
run "rsync --exclude" rsync -av --exclude="*.bin" "$S/rsync_src/" "$S/rsync_dst/"
run "rsync --include" rsync -av --include="*.txt" --exclude="*" "$S/rsync_src/" "$S/rsync_dst/"
run "rsync --progress" rsync -av --progress "$S/rsync_src/" "$S/rsync_dst/"
run "rsync -u update" rsync -avu "$S/rsync_src/" "$S/rsync_dst/"
run "rsync --checksum" rsync -av --checksum "$S/rsync_src/" "$S/rsync_dst/"
run "rsync --backup" rsync -av --backup --suffix=.bak "$S/rsync_src/" "$S/rsync_dst/"
rm -rf "$S/rsync_src" "$S/rsync_dst"

# Cleanup
rm -f "$S/curl_out.html" "$S/wget_out.html" "$S/cookies.txt" "$S/headers.txt"
rm -f "$S/trace.txt" "$S/trace_ascii.txt" "$S/upload.txt" "$S/nmap"*

domain_end
