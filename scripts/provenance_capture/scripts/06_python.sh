#!/bin/bash
# ============================================================================
# DOMAIN 06: PYTHON ECOSYSTEM
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "06 — PYTHON ECOSYSTEM"

S=$(generate_sample_files)
SCRIPTS="$PROV_WORKDIR/pyscripts"
mkdir -p "$SCRIPTS"

section "Python basics and imports"
run "python3 --version" python3 --version
run "python3 -V" python3 -V
run "python3 -c hello" python3 -c "print('Hello Python')"
run "python3 sysinfo" python3 -c "import sys,os,platform; print(sys.version); print(platform.uname())"
run "python3 all imports" python3 -c "
import os, sys, json, re, hashlib, base64, urllib.request, socket, struct
import ctypes, signal, threading, multiprocessing, subprocess, shutil
import tempfile, glob, fnmatch, pathlib, io, csv, configparser
import sqlite3, http.client, http.server, email, smtplib
import xml.etree.ElementTree, html.parser, logging, unittest
import collections, itertools, functools, operator, copy, math, random
import time, datetime, calendar, locale, codecs, unicodedata
import gzip, bz2, zipfile, tarfile, lzma
import pdb, traceback, inspect, dis, ast, tokenize, token
print('All stdlib imports OK')
"

section "File I/O operations"
cat > "$SCRIPTS/fileio.py" << 'PYEOF'
import os, tempfile, shutil, json, csv, configparser, pathlib, gzip, bz2, zipfile, tarfile
# Basic read/write
with open("/tmp/py_rw.txt","w") as f: f.write("test\n"*100)
with open("/tmp/py_rw.txt") as f: lines = f.readlines()
print(f"Read {len(lines)} lines")
# Binary
with open("/tmp/py_bin.dat","wb") as f: f.write(os.urandom(1024))
with open("/tmp/py_bin.dat","rb") as f: data = f.read()
# JSON
json.dump({"k":"v","l":[1,2,3],"n":{"a":"b"}}, open("/tmp/py.json","w"))
json.load(open("/tmp/py.json"))
# CSV
with open("/tmp/py.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["a","b"]); w.writerows([[1,2],[3,4]])
with open("/tmp/py.csv") as f: list(csv.reader(f))
# Config
c=configparser.ConfigParser(); c["S"]={"k":"v"}; c.write(open("/tmp/py.ini","w"))
c.read("/tmp/py.ini")
# Pathlib
p=pathlib.Path("/tmp/py_pathlib"); p.mkdir(exist_ok=True)
(p/"test.txt").write_text("pathlib test")
(p/"test.txt").read_text()
# Temp files
fd,path=tempfile.mkstemp(); os.write(fd,b"temp"); os.close(fd); os.unlink(path)
with tempfile.NamedTemporaryFile(delete=True) as f: f.write(b"named temp")
d=tempfile.mkdtemp(); shutil.rmtree(d)
# Compression
with gzip.open("/tmp/py.gz","wt") as f: f.write("gzip test\n"*10)
with gzip.open("/tmp/py.gz","rt") as f: f.read()
with bz2.open("/tmp/py.bz2","wt") as f: f.write("bz2 test\n"*10)
with bz2.open("/tmp/py.bz2","rt") as f: f.read()
with zipfile.ZipFile("/tmp/py.zip","w") as z: z.writestr("inner.txt","zip content")
with zipfile.ZipFile("/tmp/py.zip") as z: z.namelist(); z.read("inner.txt")
with tarfile.open("/tmp/py.tar.gz","w:gz") as t: t.add("/etc/hostname","hostname")
with tarfile.open("/tmp/py.tar.gz") as t: t.getnames()
# Directory operations
os.makedirs("/tmp/py_dirs/a/b/c",exist_ok=True)
for f in os.listdir("/etc")[:10]: print(f)
shutil.rmtree("/tmp/py_dirs")
# Cleanup
for f in ["/tmp/py_rw.txt","/tmp/py_bin.dat","/tmp/py.json","/tmp/py.csv","/tmp/py.ini",
          "/tmp/py.gz","/tmp/py.bz2","/tmp/py.zip","/tmp/py.tar.gz"]:
    try: os.unlink(f)
    except: pass
shutil.rmtree("/tmp/py_pathlib",ignore_errors=True)
print("File I/O complete")
PYEOF
run "python fileio" python3 "$SCRIPTS/fileio.py"

section "Networking"
cat > "$SCRIPTS/network.py" << 'PYEOF'
import socket, urllib.request, http.client, json, ssl
# DNS
for host in ["google.com","github.com","example.com"]:
    ips=socket.getaddrinfo(host,80,socket.AF_INET)
    print(f"{host}: {len(ips)} results")
# HTTP GET
for url in ["https://example.com","https://httpbin.org/get","https://httpbin.org/ip"]:
    try:
        r=urllib.request.urlopen(url,timeout=10)
        print(f"GET {url}: {r.status} {len(r.read())}b")
    except: pass
# HTTP POST
data=json.dumps({"key":"value"}).encode()
req=urllib.request.Request("https://httpbin.org/post",data=data,headers={"Content-Type":"application/json"})
try:
    r=urllib.request.urlopen(req,timeout=10)
    print(f"POST: {r.status}")
except: pass
# Raw socket
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(5)
try:
    s.connect(("example.com",80))
    s.send(b"GET / HTTP/1.0\r\nHost: example.com\r\n\r\n")
    print(f"Raw: {len(s.recv(2048))}b")
except: pass
finally: s.close()
# UDP
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(2)
try: s.sendto(b"\x00"*12,(("8.8.8.8",53))); print(f"UDP: {len(s.recv(512))}b")
except: pass
finally: s.close()
# HTTPSConnection
for host in ["httpbin.org","example.com"]:
    try:
        c=http.client.HTTPSConnection(host,timeout=10)
        c.request("GET","/"); r=c.getresponse(); print(f"HTTPS {host}: {r.status}")
        c.close()
    except: pass
# Multiple ports
for port in [80,443,22,8080]:
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(1)
    try:
        result=s.connect_ex(("127.0.0.1",port))
        print(f"Port {port}: {'open' if result==0 else 'closed'}")
    except: pass
    finally: s.close()
print("Network complete")
PYEOF
run "python network" python3 "$SCRIPTS/network.py"

section "Subprocess and multiprocessing"
cat > "$SCRIPTS/subprocess_mp.py" << 'PYEOF'
import subprocess, os, multiprocessing, threading, signal, time
# subprocess.run
for cmd in [["ls","-la","/tmp"],["cat","/etc/hostname"],["id"],["uname","-a"],["ps","aux"]]:
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
    print(f"{cmd[0]}: {len(r.stdout)} chars")
# Shell commands
for sh in ["echo hello | tr a-z A-Z","cat /etc/passwd | wc -l","find /usr/bin -maxdepth 1 | head -5"]:
    r=subprocess.run(sh,shell=True,capture_output=True,text=True,timeout=5)
    print(f"shell: {r.stdout.strip()[:40]}")
# Popen pipe chain
p1=subprocess.Popen(["cat","/etc/passwd"],stdout=subprocess.PIPE)
p2=subprocess.Popen(["grep","root"],stdin=p1.stdout,stdout=subprocess.PIPE)
p3=subprocess.Popen(["wc","-l"],stdin=p2.stdout,stdout=subprocess.PIPE)
p1.stdout.close(); p2.stdout.close()
print(f"Pipe chain: {p3.communicate()[0].strip()}")
# Multiprocessing
def worker(n): time.sleep(0.05); return n*n
with multiprocessing.Pool(4) as pool:
    print(f"Pool: {pool.map(worker,range(10))}")
# Threading
results=[]
def tworker(n): results.append(n*2)
ts=[threading.Thread(target=tworker,args=(i,)) for i in range(8)]
for t in ts: t.start()
for t in ts: t.join()
print(f"Threads: {sorted(results)}")
# Fork
pid=os.fork()
if pid==0: os._exit(0)
os.waitpid(pid,0)
print("Subprocess/MP complete")
PYEOF
run "python subprocess" python3 "$SCRIPTS/subprocess_mp.py"

section "HTTP servers"
cat > "$SCRIPTS/servers.py" << 'PYEOF'
import http.server, threading, urllib.request, time, json
# SimpleHTTPServer
h=http.server.SimpleHTTPRequestHandler
srv=http.server.HTTPServer(("127.0.0.1",18765),h)
t=threading.Thread(target=srv.serve_forever); t.daemon=True; t.start()
time.sleep(0.5)
r=urllib.request.urlopen("http://127.0.0.1:18765/"); print(f"HTTP: {r.status}")
srv.shutdown()
# Flask
try:
    from flask import Flask, jsonify
    app=Flask(__name__)
    @app.route("/") 
    def idx(): return "Flask OK"
    @app.route("/api")
    def api(): return jsonify({"status":"ok"})
    t=threading.Thread(target=lambda:app.run(host="127.0.0.1",port=18766,debug=False))
    t.daemon=True; t.start(); time.sleep(1)
    for p in ["/","/api"]:
        r=urllib.request.urlopen(f"http://127.0.0.1:18766{p}")
        print(f"Flask {p}: {r.status}")
except ImportError: print("Flask not installed")
except: pass
print("Servers complete")
PYEOF
run "python servers" python3 "$SCRIPTS/servers.py"

section "Crypto"
cat > "$SCRIPTS/crypto.py" << 'PYEOF'
import hashlib, hmac, base64, os, secrets
d=b"provenance crypto test data"
for alg in ["md5","sha1","sha224","sha256","sha384","sha512","sha3_256","sha3_512","blake2b","blake2s"]:
    try: print(f"{alg}: {hashlib.new(alg,d).hexdigest()[:32]}...")
    except: pass
# HMAC
k=os.urandom(32); print(f"HMAC: {hmac.new(k,d,hashlib.sha256).hexdigest()}")
# Base64
print(f"B64: {base64.b64encode(d).decode()}")
print(f"B32: {base64.b32encode(d).decode()}")
print(f"B16: {base64.b16encode(d).decode()[:32]}...")
print(f"URLsafe: {base64.urlsafe_b64encode(d).decode()}")
# Secrets
print(f"Token: {secrets.token_hex(16)}")
print(f"URL token: {secrets.token_urlsafe(16)}")
# Cryptography lib
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    # Fernet
    key=Fernet.generate_key(); f=Fernet(key)
    token=f.encrypt(d); print(f"Fernet enc: {len(token)}b, dec: {f.decrypt(token)[:20]}")
    # RSA
    pk=rsa.generate_private_key(65537,2048)
    ct=pk.public_key().encrypt(d,padding.OAEP(padding.MGF1(hashes.SHA256()),hashes.SHA256(),None))
    pt=pk.decrypt(ct,padding.OAEP(padding.MGF1(hashes.SHA256()),hashes.SHA256(),None))
    print(f"RSA: enc={len(ct)}b dec_ok={pt==d}")
    # EC
    eck=ec.generate_private_key(ec.SECP256R1())
    print(f"EC key: {eck.curve.name}")
    # Ed25519
    edk=ed25519.Ed25519PrivateKey.generate()
    sig=edk.sign(d); edk.public_key().verify(sig,d); print("Ed25519 sign+verify OK")
except ImportError: print("cryptography not installed")
print("Crypto complete")
PYEOF
run "python crypto" python3 "$SCRIPTS/crypto.py"

section "Database"
cat > "$SCRIPTS/database.py" << 'PYEOF'
import sqlite3, os
db="/tmp/prov_eval.db"
conn=sqlite3.connect(db); c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,type TEXT,target TEXT,ts REAL)")
c.execute("CREATE TABLE IF NOT EXISTS entities(id INTEGER PRIMARY KEY,type TEXT,name TEXT)")
for i in range(1000):
    c.execute("INSERT INTO events VALUES(?,?,?,?)",(i,f"EVT_{i%10}",f"/path/{i%50}",i*0.1))
for i in range(200):
    c.execute("INSERT INTO entities VALUES(?,?,?)",(i,["PROC","FILE","SOCK"][i%3],f"entity_{i}"))
conn.commit()
print(f"Events: {c.execute('SELECT COUNT(*) FROM events').fetchone()[0]}")
print(f"Entities: {c.execute('SELECT COUNT(*) FROM entities').fetchone()[0]}")
c.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(type)")
c.execute("CREATE INDEX IF NOT EXISTS idx_target ON events(target)")
c.execute("SELECT type,COUNT(*) FROM events GROUP BY type")
for row in c.fetchall(): print(f"  {row[0]}: {row[1]}")
c.execute("SELECT * FROM events WHERE type='EVT_0' LIMIT 3")
for row in c.fetchall(): print(f"  {row}")
c.execute("SELECT e.type,COUNT(*) FROM events e JOIN entities n ON e.id=n.id GROUP BY e.type LIMIT 5")
c.execute("VACUUM")
conn.close(); os.unlink(db)
print("Database complete")
PYEOF
run "python database" python3 "$SCRIPTS/database.py"

section "System info gathering"
cat > "$SCRIPTS/sysinfo.py" << 'PYEOF'
import os, sys, platform, socket, glob, pathlib, resource
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"Node: {platform.node()}")
print(f"Host: {socket.gethostname()}")
print(f"PID:{os.getpid()} PPID:{os.getppid()} UID:{os.getuid()} GID:{os.getgid()}")
print(f"CWD: {os.getcwd()}")
print(f"Uname: {os.uname()}")
print(f"CPU count: {os.cpu_count()}")
print(f"Load: {os.getloadavg()}")
try: print(f"Login: {os.getlogin()}")
except: pass
for f in ["/proc/cpuinfo","/proc/meminfo","/proc/version","/proc/loadavg","/proc/uptime"]:
    try: print(f"{f}: {open(f).readline().strip()}")
    except: pass
for f in glob.glob("/etc/*release"):
    try: print(f"{f}: {open(f).readline().strip()}")
    except: pass
for d in ["/usr/bin","/usr/sbin","/usr/lib","/etc","/var/log"]:
    try: print(f"{d}: {len(os.listdir(d))} entries")
    except: pass
r=resource.getrusage(resource.RUSAGE_SELF)
print(f"RSS: {r.ru_maxrss}KB, User: {r.ru_utime:.3f}s, Sys: {r.ru_stime:.3f}s")
print("Sysinfo complete")
PYEOF
run "python sysinfo" python3 "$SCRIPTS/sysinfo.py"

section "Web scraping"
cat > "$SCRIPTS/scrape.py" << 'PYEOF'
try:
    from bs4 import BeautifulSoup
    import urllib.request
    for url in ["https://example.com","https://httpbin.org/html"]:
        try:
            r=urllib.request.urlopen(url,timeout=10)
            soup=BeautifulSoup(r.read(),"html.parser")
            print(f"{url}: title={soup.title.string if soup.title else 'N/A'} links={len(soup.find_all('a'))} p={len(soup.find_all('p'))}")
        except Exception as e: print(f"{url}: {e}")
except ImportError: print("bs4 not installed")
PYEOF
run "python scrape" python3 "$SCRIPTS/scrape.py"

section "Pip operations"
run "pip list" pip list 2>/dev/null | head -20
run "pip freeze" pip freeze 2>/dev/null | head -20
run "pip show requests" pip show requests 2>/dev/null
run "pip show numpy" pip show numpy 2>/dev/null
run "pip show flask" pip show flask 2>/dev/null
run "pip check" pip check 2>/dev/null
run "pip config list" pip config list 2>/dev/null
run "pip cache list" pip cache list 2>/dev/null | head -5

section "Python one-liners — diverse syscall patterns"
# File operations
run "py read /etc/passwd" python3 -c "print(len(open('/etc/passwd').readlines()),'lines')"
run "py read /etc/hosts" python3 -c "print(open('/etc/hosts').read())"
run "py read /proc/cpuinfo" python3 -c "print(open('/proc/cpuinfo').readline().strip())"
run "py read /proc/meminfo" python3 -c "print(open('/proc/meminfo').readline().strip())"
run "py read binary" python3 -c "print(len(open('/usr/bin/ls','rb').read()),'bytes')"
run "py write" python3 -c "open('/tmp/py_one.txt','w').write('one-liner\n')"
run "py append" python3 -c "open('/tmp/py_one.txt','a').write('appended\n')"
run "py readlines" python3 -c "print(open('/tmp/py_one.txt').readlines())"
run "py pathlib read" python3 -c "from pathlib import Path; print(Path('/etc/hostname').read_text().strip())"
run "py pathlib write" python3 -c "from pathlib import Path; Path('/tmp/py_pathlib.txt').write_text('pathlib one-liner\n')"
run "py pathlib glob" python3 -c "from pathlib import Path; print(list(Path('/etc').glob('*.conf'))[:5])"
run "py os.listdir" python3 -c "import os; print(os.listdir('/etc')[:10])"
run "py os.walk" python3 -c "import os; [print(d,len(fs)) for d,_,fs in list(os.walk('/usr/bin'))[:3]]"
run "py os.stat" python3 -c "import os; s=os.stat('/usr/bin/ls'); print(f'size={s.st_size} mode={oct(s.st_mode)}')"
run "py shutil.which" python3 -c "import shutil; print(shutil.which('python3'), shutil.which('gcc'), shutil.which('curl'))"
run "py shutil.disk_usage" python3 -c "import shutil; u=shutil.disk_usage('/'); print(f'total={u.total//1e9:.0f}G used={u.used//1e9:.0f}G free={u.free//1e9:.0f}G')"
run "py tempfile" python3 -c "import tempfile,os; f=tempfile.NamedTemporaryFile(delete=False); f.write(b'temp'); f.close(); print(f.name); os.unlink(f.name)"
run "py mmap" python3 -c "import mmap,os; f=open('/etc/passwd','rb'); m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); print(m[:50]); m.close(); f.close()"

# Process operations
run "py os.getpid" python3 -c "import os; print(f'PID={os.getpid()} PPID={os.getppid()} UID={os.getuid()} GID={os.getgid()}')"
run "py os.uname" python3 -c "import os; print(os.uname())"
run "py os.cpu_count" python3 -c "import os; print(f'CPUs={os.cpu_count()}')"
run "py os.getloadavg" python3 -c "import os; print(f'Load={os.getloadavg()}')"
run "py os.environ" python3 -c "import os; print({k:v for k,v in list(os.environ.items())[:5]})"
run "py os.system" python3 -c "import os; os.system('echo py_os_system')"
run "py os.popen" python3 -c "import os; print(os.popen('hostname').read().strip())"
run "py os.fork" python3 -c "import os; pid=os.fork(); (os._exit(0) if pid==0 else os.waitpid(pid,0)); print('fork done')"
run "py subprocess.run ls" python3 -c "import subprocess; r=subprocess.run(['ls','/tmp'],capture_output=True,text=True); print(r.stdout[:100])"
run "py subprocess.run id" python3 -c "import subprocess; r=subprocess.run(['id'],capture_output=True,text=True); print(r.stdout.strip())"
run "py subprocess.check_output" python3 -c "import subprocess; print(subprocess.check_output(['uname','-a']).decode().strip())"
run "py subprocess shell" python3 -c "import subprocess; subprocess.run('echo hello | tr a-z A-Z',shell=True)"
run "py subprocess.Popen" python3 -c "import subprocess; p=subprocess.Popen(['cat','/etc/hostname'],stdout=subprocess.PIPE); print(p.communicate()[0].decode().strip())"
run "py multiprocessing.Process" python3 -c "import multiprocessing; p=multiprocessing.Process(target=print,args=('child',)); p.start(); p.join()"
run "py threading.Thread" python3 -c "import threading; t=threading.Thread(target=print,args=('thread',)); t.start(); t.join()"

# Network operations
run "py socket dns" python3 -c "import socket; print(socket.getaddrinfo('google.com',80)[:2])"
run "py socket gethostname" python3 -c "import socket; print(socket.gethostname(),socket.getfqdn())"
run "py socket gethostbyname" python3 -c "import socket; print(socket.gethostbyname('example.com'))"
run "py urllib GET" python3 -c "import urllib.request; r=urllib.request.urlopen('https://example.com',timeout=10); print(f'{r.status} {len(r.read())}b')"
run "py urllib POST" python3 -c "import urllib.request,json; d=json.dumps({'k':'v'}).encode(); req=urllib.request.Request('https://httpbin.org/post',data=d,headers={'Content-Type':'application/json'}); r=urllib.request.urlopen(req,timeout=10); print(r.status)"
run "py urllib headers" python3 -c "import urllib.request; r=urllib.request.urlopen('https://example.com',timeout=10); print(dict(r.headers))"
run "py http.client GET" python3 -c "import http.client; c=http.client.HTTPSConnection('example.com',timeout=10); c.request('GET','/'); print(c.getresponse().status); c.close()"
run "py raw socket" python3 -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('example.com',80)); s.send(b'GET / HTTP/1.0\r\nHost: example.com\r\n\r\n'); print(len(s.recv(1024)),'bytes'); s.close()"
run "py socket server+client" python3 -c "
import socket,threading
def srv():
    s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',19876)); s.listen(1)
    c,_=s.accept(); c.send(b'hello'); c.close(); s.close()
t=threading.Thread(target=srv); t.start()
import time; time.sleep(0.3)
c=socket.socket(); c.connect(('127.0.0.1',19876)); print(c.recv(10)); c.close(); t.join()
"

# Hashing one-liners
run "py md5" python3 -c "import hashlib; print(hashlib.md5(b'test').hexdigest())"
run "py sha256" python3 -c "import hashlib; print(hashlib.sha256(b'test').hexdigest())"
run "py sha512" python3 -c "import hashlib; print(hashlib.sha512(b'test').hexdigest())"
run "py sha3_256" python3 -c "import hashlib; print(hashlib.sha3_256(b'test').hexdigest())"
run "py blake2b" python3 -c "import hashlib; print(hashlib.blake2b(b'test').hexdigest()[:32])"
run "py hmac" python3 -c "import hmac,hashlib,os; print(hmac.new(os.urandom(32),b'test',hashlib.sha256).hexdigest())"

# Encoding one-liners
run "py base64 enc" python3 -c "import base64; print(base64.b64encode(b'hello python').decode())"
run "py base64 dec" python3 -c "import base64; print(base64.b64decode('aGVsbG8gcHl0aG9u').decode())"
run "py base32" python3 -c "import base64; print(base64.b32encode(b'hello').decode())"
run "py hex" python3 -c "print(b'hello'.hex())"
run "py urlsafe_b64" python3 -c "import base64; print(base64.urlsafe_b64encode(b'hello+world/test').decode())"
run "py secrets" python3 -c "import secrets; print(secrets.token_hex(16),secrets.token_urlsafe(16))"

# Regex one-liners
run "py re.search" python3 -c "import re; print(re.search(r'\d+','abc123def').group())"
run "py re.findall" python3 -c "import re; print(re.findall(r'\b\w+\b','hello world 123'))"
run "py re.sub" python3 -c "import re; print(re.sub(r'\d+','NUM','abc 123 def 456'))"
run "py re.split" python3 -c "import re; print(re.split(r'[,;:]','a,b;c:d'))"

# JSON one-liners
run "py json.dumps" python3 -c "import json; print(json.dumps({'key':'value','list':[1,2,3]},indent=2))"
run "py json.loads" python3 -c 'import json; print(json.loads("{\"a\":1,\"b\":[2,3]}"))'

# Math/data one-liners
run "py math" python3 -c "import math; print(math.pi,math.e,math.sqrt(2),math.factorial(10))"
run "py random" python3 -c "import random; print([random.randint(1,100) for _ in range(10)])"
run "py collections.Counter" python3 -c "from collections import Counter; print(Counter('abracadabra'))"
run "py itertools" python3 -c "from itertools import combinations; print(list(combinations('ABCD',2)))"
run "py datetime" python3 -c "from datetime import datetime; print(datetime.now().isoformat())"
run "py struct pack" python3 -c "import struct; print(struct.pack('>I',12345).hex())"

# NumPy/Pandas (if available)
run "py numpy" python3 -c "import numpy as np; a=np.random.randn(100); print(f'mean={a.mean():.3f} std={a.std():.3f}')" 2>/dev/null
run "py numpy matmul" python3 -c "import numpy as np; a=np.random.randn(100,100); b=np.random.randn(100,100); print((a@b).shape)" 2>/dev/null
run "py pandas" python3 -c "import pandas as pd; df=pd.DataFrame({'a':range(10),'b':range(10,20)}); print(df.describe())" 2>/dev/null

# psutil (if available)
run "py psutil" python3 -c "import psutil; print(f'CPU={psutil.cpu_percent()}% MEM={psutil.virtual_memory().percent}%')" 2>/dev/null
run "py psutil processes" python3 -c "import psutil; [print(p.pid,p.name()) for p in list(psutil.process_iter())[:10]]" 2>/dev/null
run "py psutil network" python3 -c "import psutil; print(psutil.net_connections()[:5])" 2>/dev/null

# Cleanup
rm -f /tmp/py_one.txt /tmp/py_pathlib.txt

domain_end
