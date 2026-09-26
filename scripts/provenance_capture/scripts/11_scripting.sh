#!/bin/bash
# ============================================================================
# DOMAIN 11: SCRIPTING ENGINES
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "11 — SCRIPTING ENGINES"

section "Perl"
run "perl -e print" perl -e 'print "Hello Perl\n"'
run "perl -e file read" perl -e 'open(F,"/etc/passwd");@l=<F>;close(F);print scalar @l," lines\n"'
run "perl -e file write" perl -e 'open(F,">","/tmp/perl_out.txt");print F "perl output\n";close(F)'
run "perl -ne" perl -ne 'print if /root/' /etc/passwd
run "perl -pe substitute" echo "hello world" | perl -pe 's/world/perl/'
run "perl -an fields" perl -ane 'print "$F[0]\n"' /etc/passwd | head -3
run "perl -i edit" cp /etc/hosts /tmp/perl_hosts && perl -i -pe 's/localhost/LOCALHOST/' /tmp/perl_hosts; rm /tmp/perl_hosts
run "perl regex" echo "test 123 abc 456" | perl -pe 's/\d+/NUM/g'
run "perl hash" perl -e 'use Digest::MD5 qw(md5_hex);print md5_hex("test"),"\n"'
run "perl sha" perl -e 'use Digest::SHA qw(sha256_hex);print sha256_hex("test"),"\n"'
run "perl base64" perl -e 'use MIME::Base64;print encode_base64("hello perl")'
run "perl json" perl -e 'use JSON::PP;print encode_json({key=>"value"}),"\n"' 2>/dev/null
run "perl network" perl -e 'use IO::Socket::INET;$s=IO::Socket::INET->new(PeerAddr=>"example.com",PeerPort=>80,Proto=>"tcp");print $s "GET / HTTP/1.0\r\nHost: example.com\r\n\r\n";$l=<$s>;print $l;close($s)' 2>/dev/null
run "perl dns" perl -e 'use Socket;my @r=gethostbyname("google.com");print "resolved\n" if @r'
run "perl system" perl -e 'system("echo perl_subprocess");$o=`hostname`;chomp $o;print "host=$o\n"'
run "perl fork" perl -e 'if(fork()==0){print "child $$\n";exit}wait;print "parent $$\n"'
run "perl env" perl -e 'foreach(sort keys %ENV){print "$_=$ENV{$_}\n" if /^(HOME|PATH|USER)/}'
run "perl file test" perl -e 'for("/etc/passwd","/etc/shadow","/tmp"){print "$_: ",(-e $_?"exists":"missing"),(-r $_?" readable":""),"\n"}'
run "perl glob" perl -e 'print "$_\n" for glob("/etc/*.conf")' | head -5
run "perl -w" perl -w -e 'print "warnings on\n"'
run "perl -T taint" perl -T -e 'print "taint mode\n"' 2>/dev/null || true
rm -f /tmp/perl_out.txt

section "Ruby"
run "ruby -e print" ruby -e 'puts "Hello Ruby"'
run "ruby file read" ruby -e 'puts File.readlines("/etc/passwd").length.to_s + " lines"'
run "ruby file write" ruby -e 'File.write("/tmp/ruby_out.txt","ruby output\n")'
run "ruby json" ruby -e 'require "json"; puts JSON.generate({"key"=>"value","arr"=>[1,2,3]})'
run "ruby yaml" ruby -e 'require "yaml"; puts YAML.dump({"key"=>"value"})' 2>/dev/null
run "ruby net http" ruby -e 'require "net/http"; r=Net::HTTP.get_response(URI("https://example.com")); puts "HTTP #{r.code}"' 2>/dev/null
run "ruby digest" ruby -e 'require "digest"; puts Digest::SHA256.hexdigest("test")'
run "ruby base64" ruby -e 'require "base64"; puts Base64.encode64("hello ruby")'
run "ruby system" ruby -e 'puts `hostname`.strip'
run "ruby backtick" ruby -e 'puts `ls /tmp | head -5`'
run "ruby fork" ruby -e 'pid=fork{puts "child #{$$}"}; Process.wait(pid); puts "parent #{$$}"'
run "ruby glob" ruby -e 'Dir.glob("/etc/*.conf").each{|f| puts f}' | head -5
run "ruby env" ruby -e 'ENV.each{|k,v| puts "#{k}=#{v}" if k=~/^(HOME|PATH|USER)/}'
run "ruby socket" ruby -e 'require "socket"; s=TCPSocket.new("example.com",80); s.puts "GET / HTTP/1.0\r\nHost: example.com\r\n\r\n"; puts s.gets; s.close' 2>/dev/null
rm -f /tmp/ruby_out.txt

section "Lua"
run "lua print" lua5.4 -e 'print("Hello Lua")' 2>/dev/null
run "lua file" lua5.4 -e 'f=io.open("/etc/hostname","r");print(f:read("*a"));f:close()' 2>/dev/null
run "lua math" lua5.4 -e 'for i=1,10 do print(i,math.sqrt(i),math.sin(i)) end' 2>/dev/null
run "lua table" lua5.4 -e 't={1,2,3,"hello"}; for i,v in ipairs(t) do print(i,v) end' 2>/dev/null
run "lua string" lua5.4 -e 'print(string.format("pi=%.5f",math.pi))' 2>/dev/null
run "lua os" lua5.4 -e 'os.execute("echo lua_subprocess")' 2>/dev/null
run "lua os.clock" lua5.4 -e 'print("clock="..os.clock())' 2>/dev/null
run "lua os.date" lua5.4 -e 'print("date="..os.date())' 2>/dev/null

section "PHP"
run "php -r print" php -r 'echo "Hello PHP ".phpversion()."\n";'
run "php file" php -r 'echo count(file("/etc/passwd"))." lines\n";'
run "php file_get" php -r '$r=@file_get_contents("https://example.com");echo strlen($r)." bytes\n";' 2>/dev/null
run "php json" php -r 'echo json_encode(["k"=>"v","a"=>[1,2,3]])."\n";'
run "php hash md5" php -r 'echo md5("test")."\n";'
run "php hash sha256" php -r 'echo hash("sha256","test")."\n";'
run "php base64" php -r 'echo base64_encode("hello php")."\n";'
run "php system" php -r 'echo shell_exec("hostname");'
run "php exec" php -r 'exec("id",$o);echo implode("\n",$o)."\n";'
run "php glob" php -r 'print_r(array_slice(glob("/etc/*.conf"),0,5));'
run "php date" php -r 'echo date("Y-m-d H:i:s")."\n";'
run "php getenv" php -r 'echo getenv("HOME")."\n";'
run "php phpinfo" php -r 'phpinfo(INFO_GENERAL);' 2>/dev/null | head -10

section "Node.js"
run "node -e print" node -e 'console.log("Hello Node")'
run "node fs read" node -e 'const fs=require("fs");console.log(fs.readFileSync("/etc/passwd","utf8").split("\n").length+" lines")'
run "node fs write" node -e 'require("fs").writeFileSync("/tmp/node_out.txt","node output\n")'
run "node crypto" node -e 'const c=require("crypto");console.log(c.createHash("sha256").update("test").digest("hex"))'
run "node crypto random" node -e 'console.log(require("crypto").randomBytes(16).toString("hex"))'
run "node https" node -e 'const https=require("https");https.get("https://example.com",(r)=>{let d="";r.on("data",c=>d+=c);r.on("end",()=>console.log(d.length+" bytes"))})' 2>/dev/null
run "node dns" node -e 'require("dns").resolve4("google.com",(e,a)=>{console.log(a)})' 2>/dev/null
run "node child_process" node -e 'console.log(require("child_process").execSync("hostname").toString().trim())'
run "node os" node -e 'const os=require("os");console.log(os.hostname(),os.platform(),os.arch(),os.cpus().length+"cpus")'
run "node path" node -e 'const p=require("path");console.log(p.resolve("."),p.join("/usr","bin","node"))'
run "node json" node -e 'console.log(JSON.stringify({k:"v",a:[1,2,3]},null,2))'
run "node buffer" node -e 'console.log(Buffer.from("hello node").toString("base64"))'
run "node url" node -e 'const u=new URL("https://example.com:443/path?q=1");console.log(u.hostname,u.port,u.pathname)'
rm -f /tmp/node_out.txt

section "awk as scripting"
run "awk script" awk 'BEGIN{for(i=1;i<=10;i++)printf "%d^2=%d\n",i,i*i}'
run "awk getline" echo test | awk '{cmd="hostname";cmd|getline h;close(cmd);print $0,h}'
run "awk system" awk 'BEGIN{system("echo awk_subprocess")}'
run "awk math" awk 'BEGIN{print sin(1),cos(1),exp(1),log(10),sqrt(2)}'
run "awk rand" awk 'BEGIN{srand();for(i=0;i<5;i++)printf "%.3f ",rand();print""}'
run "awk time" awk 'BEGIN{print systime(),strftime("%Y-%m-%d",systime())}'

section "Bash scripting"
run "bash arithmetic" bash -c 'echo $((2**16)) $((RANDOM%100)) $((1+2*3))'
run "bash arrays" bash -c 'a=(one two three four five);echo "${#a[@]} elements: ${a[*]}"'
run "bash assoc array" bash -c 'declare -A m; m[key]=val; m[other]=123; for k in "${!m[@]}"; do echo "$k=${m[$k]}"; done'
run "bash heredoc" bash -c 'cat <<END
line 1
line 2
line 3
END'
run "bash for loop" bash -c 'for i in {1..10}; do echo -n "$i "; done; echo'
run "bash while" bash -c 'i=0; while [ $i -lt 5 ]; do echo -n "$i "; i=$((i+1)); done; echo'
run "bash case" bash -c 'x=hello; case $x in he*) echo "starts with he";; *) echo "other";; esac'
run "bash function" bash -c 'greet(){ echo "Hello $1"; }; greet World; greet Provenance'
run "bash trap" bash -c 'trap "echo trapped" EXIT; echo "before exit"'
run "bash subshell" bash -c '(cd /tmp && pwd); pwd'
run "bash coprocess" bash -c 'coproc CAT { cat; }; echo hello >&${CAT[1]}; read line <&${CAT[0]}; echo $line; exec {CAT[1]}>&-' 2>/dev/null || true
run "bash mapfile" bash -c 'mapfile -t lines < /etc/passwd; echo "${#lines[@]} lines"'
run "bash select" bash -c 'echo 1 | { select opt in one two three; do echo $opt; break; done; }' 2>/dev/null || true

domain_end
