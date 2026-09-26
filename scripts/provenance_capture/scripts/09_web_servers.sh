#!/bin/bash
# ============================================================================
# DOMAIN 09: WEB SERVERS & SERVICES
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "09 — WEB SERVERS"
WEB="$PROV_WORKDIR/web"; mkdir -p "$WEB"

section "Python HTTP servers"
run_t 10 "python http.server" bash -c 'python3 -m http.server 18770 --directory /tmp &>/dev/null & P=$!; sleep 1; curl -s http://localhost:18770/ -o /dev/null; curl -s http://localhost:18770/etc/ -o /dev/null 2>/dev/null; kill $P 2>/dev/null'
run_t 10 "python https server" bash -c 'python3 -c "
import http.server,ssl,threading
h=http.server.HTTPServer((\"127.0.0.1\",18771),http.server.SimpleHTTPRequestHandler)
ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
try:
    ctx.load_cert_chain(\"/tmp/prov_workdir/keys/selfsign_cert.pem\",\"/tmp/prov_workdir/keys/selfsign_key.pem\")
    h.socket=ctx.wrap_socket(h.socket)
    t=threading.Thread(target=h.serve_forever);t.daemon=True;t.start()
    import time;time.sleep(1)
    import urllib.request;urllib.request.urlopen(\"https://127.0.0.1:18771/\",context=ssl._create_unverified_context())
except: pass
finally: h.shutdown()
" 2>/dev/null'

section "Node.js HTTP"
cat > "$WEB/node_srv.js" << 'EOF'
const http=require('http'),fs=require('fs');
const s=http.createServer((q,r)=>{r.writeHead(200);r.end('Node OK\n')});
s.listen(18772,'127.0.0.1',()=>{setTimeout(()=>{s.close();process.exit(0)},3000)});
EOF
run_t 10 "node http" bash -c 'node "$WEB/node_srv.js" & sleep 1; curl -s http://localhost:18772/; curl -s -X POST -d test http://localhost:18772/; wait 2>/dev/null'

section "PHP built-in server"
echo '<?php echo "PHP ".phpversion()."\n"; echo json_encode($_SERVER)."\n"; ?>' > "$WEB/index.php"
run_t 10 "php server" bash -c 'php -S 127.0.0.1:18773 -t "$WEB/" &>/dev/null & P=$!; sleep 1; curl -s http://localhost:18773/; curl -s http://localhost:18773/index.php; kill $P 2>/dev/null'

section "Ruby WEBrick"
run_t 10 "ruby webrick" bash -c 'ruby -e "require\"webrick\";s=WEBrick::HTTPServer.new(Port:18774,DocumentRoot:\"/tmp\",Logger:WEBrick::Log.new(\"/dev/null\"),AccessLog:[]);Thread.new{sleep 3;s.shutdown};s.start" &>/dev/null & P=$!; sleep 1; curl -s http://localhost:18774/ -o /dev/null; wait $P 2>/dev/null'

section "Perl HTTP::Tiny server"
run_t 10 "perl http" bash -c 'perl -e "use IO::Socket::INET;my \$s=IO::Socket::INET->new(LocalPort=>18775,Listen=>5,Reuse=>1);my \$c=\$s->accept();print \$c \"HTTP/1.0 200 OK\r\n\r\nPerl OK\n\";\$c->close();\$s->close()" & sleep 0.5; curl -s http://localhost:18775/'

section "Nginx"
run "nginx -t" nginx -t 2>/dev/null
run "nginx -T" nginx -T 2>/dev/null | head -30
run "nginx -v" nginx -v 2>/dev/null
run "nginx -V" nginx -V 2>/dev/null
run "cat nginx.conf" cat /etc/nginx/nginx.conf 2>/dev/null
run "ls nginx sites" ls /etc/nginx/sites-enabled/ 2>/dev/null
run "ls nginx conf.d" ls /etc/nginx/conf.d/ 2>/dev/null
run "nginx start" service nginx start 2>/dev/null || systemctl start nginx 2>/dev/null || true
run_t 5 "curl nginx" curl -s http://localhost/ -o /dev/null -w "%{http_code}"
run_t 5 "curl nginx HEAD" curl -s -I http://localhost/
run_t 5 "curl nginx POST" curl -s -X POST -d "test" http://localhost/ -o /dev/null
run "nginx stop" service nginx stop 2>/dev/null || systemctl stop nginx 2>/dev/null || true

section "Apache"
run "apache2 -t" apache2 -t 2>/dev/null
run "apache2 -v" apache2 -v 2>/dev/null
run "apache2 -V" apache2 -V 2>/dev/null | head -10
run "apache2 -S" apache2 -S 2>/dev/null
run "apache2 -M" apache2 -M 2>/dev/null | head -10
run "cat apache2.conf" cat /etc/apache2/apache2.conf 2>/dev/null | head -20
run "ls apache mods" ls /etc/apache2/mods-enabled/ 2>/dev/null

section "Multiple HTTP request patterns"
# Start a background server
python3 -m http.server 18780 --directory /tmp &>/dev/null &
HTTP_PID=$!
sleep 1

# Various HTTP methods and patterns
run_t 5 "GET /" curl -s http://localhost:18780/ -o /dev/null -w "%{http_code}"
run_t 5 "GET /etc/" curl -s http://localhost:18780/etc/ -o /dev/null 2>/dev/null || true
run_t 5 "HEAD /" curl -s -I http://localhost:18780/
run_t 5 "POST /" curl -s -X POST -d "key=value" http://localhost:18780/ -o /dev/null || true
run_t 5 "PUT /" curl -s -X PUT -d "data" http://localhost:18780/ -o /dev/null || true
run_t 5 "DELETE /" curl -s -X DELETE http://localhost:18780/ -o /dev/null || true
run_t 5 "OPTIONS /" curl -s -X OPTIONS http://localhost:18780/ -o /dev/null || true
run_t 5 "wget server" wget -q -O /dev/null http://localhost:18780/
run_t 5 "python urllib" python3 -c "import urllib.request; r=urllib.request.urlopen('http://localhost:18780/'); print(r.status)"
run_t 5 "node http.get" node -e "require('http').get('http://localhost:18780/',(r)=>{console.log(r.statusCode)})" 2>/dev/null
run_t 5 "ruby net/http" ruby -e 'require "net/http"; puts Net::HTTP.get_response(URI("http://localhost:18780/")).code' 2>/dev/null
run_t 5 "php file_get" php -r 'echo strlen(@file_get_contents("http://localhost:18780/"))." bytes\n";' 2>/dev/null
run_t 5 "perl LWP" perl -e 'use IO::Socket::INET;$s=IO::Socket::INET->new(PeerAddr=>"127.0.0.1",PeerPort=>18780,Proto=>"tcp");print $s "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n";print <$s>;close($s)' 2>/dev/null
run_t 5 "curl user-agent bot" curl -s -A "Googlebot/2.1" http://localhost:18780/ -o /dev/null
run_t 5 "curl user-agent chrome" curl -s -A "Mozilla/5.0 Chrome/120.0" http://localhost:18780/ -o /dev/null
run_t 5 "curl with headers" curl -s -H "Accept: text/html" -H "Accept-Language: en" http://localhost:18780/ -o /dev/null
run_t 5 "curl cookie" curl -s -b "session=test123" http://localhost:18780/ -o /dev/null
run_t 5 "curl range" curl -s --range 0-50 http://localhost:18780/ -o /dev/null
run_t 5 "concurrent requests" bash -c 'for i in $(seq 1 10); do curl -s http://localhost:18780/ -o /dev/null & done; wait'

kill $HTTP_PID 2>/dev/null; wait $HTTP_PID 2>/dev/null

domain_end
