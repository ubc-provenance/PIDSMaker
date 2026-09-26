#!/bin/bash
# ============================================================================
# DOMAIN 05: COMPILERS & BUILD SYSTEMS
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "05 — COMPILERS & BUILD"

S=$(generate_sample_files)
SRC="$PROV_WORKDIR/src"
mkdir -p "$SRC"
export PATH="$HOME/.cargo/bin:$PATH"

# ============================================================================
section "Generate source files"
# ============================================================================

cat > "$SRC/hello.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
int main(int argc, char *argv[]) {
    printf("Hello from C! PID=%d PPID=%d\n", getpid(), getppid());
    FILE *f = fopen("/tmp/c_out.txt", "w");
    if (f) { fprintf(f, "C output\n"); fclose(f); }
    int fd = open("/etc/hostname", O_RDONLY);
    if (fd >= 0) { char buf[256]; read(fd, buf, sizeof(buf)); close(fd); }
    return 0;
}
EOF

cat > "$SRC/network.c" << 'EOF'
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {.sin_family=AF_INET, .sin_port=htons(80)};
    inet_pton(AF_INET, "93.184.216.34", &addr.sin_addr);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        write(sock, "GET / HTTP/1.0\r\nHost: example.com\r\n\r\n", 37);
        char buf[4096]; read(sock, buf, sizeof(buf));
    }
    close(sock);
    return 0;
}
EOF

cat > "$SRC/fork_test.c" << 'EOF'
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
int main() {
    for (int i = 0; i < 5; i++) {
        pid_t pid = fork();
        if (pid == 0) { printf("Child %d PID=%d\n", i, getpid()); _exit(0); }
        waitpid(pid, NULL, 0);
    }
    return 0;
}
EOF

cat > "$SRC/threads.c" << 'EOF'
#include <stdio.h>
#include <pthread.h>
void *worker(void *arg) { printf("Thread %ld\n", (long)arg); return NULL; }
int main() {
    pthread_t t[4];
    for (long i = 0; i < 4; i++) pthread_create(&t[i], NULL, worker, (void*)i);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);
    return 0;
}
EOF

cat > "$SRC/lib.c" << 'EOF'
#include <stdio.h>
void greet(const char *name) { printf("Hello, %s!\n", name); }
EOF

cat > "$SRC/lib.h" << 'EOF'
void greet(const char *name);
EOF

cat > "$SRC/use_lib.c" << 'EOF'
#include "lib.h"
int main() { greet("provenance"); return 0; }
EOF

cat > "$SRC/hello.cpp" << 'EOF'
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
int main() {
    std::cout << "Hello from C++!" << std::endl;
    std::vector<int> v = {5, 3, 1, 4, 2};
    std::sort(v.begin(), v.end());
    std::ofstream out("/tmp/cpp_out.txt");
    for (int x : v) out << x << " ";
    return 0;
}
EOF

cat > "$SRC/templates.cpp" << 'EOF'
#include <iostream>
#include <vector>
#include <map>
#include <memory>
template<typename T> T add(T a, T b) { return a + b; }
int main() {
    std::cout << add(1, 2) << " " << add(1.5, 2.5) << std::endl;
    auto p = std::make_unique<int>(42);
    std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
    for (auto& [k, v] : m) std::cout << k << "=" << v << " ";
    return 0;
}
EOF

cat > "$SRC/hello.asm" << 'EOF'
section .data
    msg db "Hello from NASM!", 10
    len equ $ - msg
section .text
    global _start
_start:
    mov rax, 1
    mov rdi, 1
    mov rsi, msg
    mov rdx, len
    syscall
    mov rax, 60
    xor rdi, rdi
    syscall
EOF

cat > "$SRC/hello.go" << 'EOF'
package main
import (
    "fmt"
    "os"
    "io/ioutil"
    "net/http"
)
func main() {
    fmt.Println("Hello from Go!")
    h, _ := os.Hostname()
    fmt.Println("Host:", h)
    resp, err := http.Get("https://example.com")
    if err == nil {
        defer resp.Body.Close()
        b, _ := ioutil.ReadAll(resp.Body)
        fmt.Printf("Got %d bytes\n", len(b))
    }
}
EOF

cat > "$SRC/hello.rs" << 'EOF'
use std::fs;
fn main() {
    println!("Hello from Rust!");
    let c = fs::read_to_string("/etc/hostname").unwrap_or_default();
    println!("Hostname: {}", c.trim());
}
EOF

cat > "$SRC/Hello.java" << 'EOF'
import java.io.*;
import java.net.*;
public class Hello {
    public static void main(String[] args) throws Exception {
        System.out.println("Hello from Java!");
        System.out.println("Host: " + InetAddress.getLocalHost().getHostName());
        FileWriter fw = new FileWriter("/tmp/java_out.txt");
        fw.write("Java output\n"); fw.close();
    }
}
EOF

cat > "$SRC/hello.py" << 'EOF'
import os, sys, socket
print(f"Hello from Python! PID={os.getpid()}")
print(f"Host: {socket.gethostname()}")
with open("/tmp/py_compile_out.txt", "w") as f:
    f.write("compiled python output\n")
EOF

cat > "$SRC/hello.pl" << 'EOF'
use strict;
use warnings;
print "Hello from Perl!\n";
open(my $fh, '<', '/etc/hostname') or die;
my $host = <$fh>; chomp $host; close $fh;
print "Host: $host\n";
EOF

cat > "$SRC/hello.rb" << 'EOF'
puts "Hello from Ruby!"
puts "Host: #{`hostname`.strip}"
File.write("/tmp/ruby_out.txt", "Ruby output\n")
EOF

cat > "$SRC/hello.lua" << 'EOF'
print("Hello from Lua!")
local f = io.open("/etc/hostname", "r")
if f then print("Host: " .. f:read("*l")); f:close() end
EOF

cat > "$SRC/hello.js" << 'EOF'
const fs = require('fs');
const os = require('os');
console.log("Hello from Node.js!");
console.log("Host:", os.hostname());
fs.writeFileSync("/tmp/node_out.txt", "Node output\n");
EOF

cat > "$SRC/hello.php" << 'EOF'
<?php
echo "Hello from PHP " . phpversion() . "\n";
echo "Host: " . gethostname() . "\n";
file_put_contents("/tmp/php_out.txt", "PHP output\n");
EOF

# ============================================================================
section "GCC — C compiler (all major flags)"
# ============================================================================
run "gcc basic" gcc -o "$SRC/hello" "$SRC/hello.c"
run "gcc run" "$SRC/hello"
run "gcc -O0 -g" gcc -O0 -g -o "$SRC/hello_O0" "$SRC/hello.c"
run "gcc -O1" gcc -O1 -o "$SRC/hello_O1" "$SRC/hello.c"
run "gcc -O2" gcc -O2 -o "$SRC/hello_O2" "$SRC/hello.c"
run "gcc -O3" gcc -O3 -o "$SRC/hello_O3" "$SRC/hello.c"
run "gcc -Os" gcc -Os -o "$SRC/hello_Os" "$SRC/hello.c"
run "gcc -Oz" gcc -Oz -o "$SRC/hello_Oz" "$SRC/hello.c" 2>/dev/null || true
run "gcc -Og" gcc -Og -g -o "$SRC/hello_Og" "$SRC/hello.c"
run "gcc -O3 -march=native" gcc -O3 -march=native -o "$SRC/hello_native" "$SRC/hello.c"
run "gcc -O2 -flto" gcc -O2 -flto -o "$SRC/hello_lto" "$SRC/hello.c"
run "gcc -S assembly" gcc -S -o "$SRC/hello.s" "$SRC/hello.c"
run "gcc -E preprocess" gcc -E -o "$SRC/hello.i" "$SRC/hello.c"
run "gcc -c object" gcc -c -o "$SRC/hello.o" "$SRC/hello.c"
run "gcc link object" gcc -o "$SRC/hello_linked" "$SRC/hello.o"
run "gcc -pipe" gcc -pipe -o "$SRC/hello_pipe" "$SRC/hello.c"
run "gcc -v verbose" gcc -v -o /dev/null "$SRC/hello.c" 2>/dev/null
run "gcc -Wall" gcc -Wall -o /dev/null "$SRC/hello.c"
run "gcc -Wall -Wextra" gcc -Wall -Wextra -o /dev/null "$SRC/hello.c"
run "gcc -Wall -Werror" gcc -Wall -Werror -o /dev/null "$SRC/hello.c" 2>/dev/null || true
run "gcc -Wpedantic" gcc -Wpedantic -o /dev/null "$SRC/hello.c"
run "gcc -std=c99" gcc -std=c99 -o /dev/null "$SRC/hello.c"
run "gcc -std=c11" gcc -std=c11 -o /dev/null "$SRC/hello.c"
run "gcc -std=c17" gcc -std=c17 -o /dev/null "$SRC/hello.c" 2>/dev/null || true
run "gcc -std=gnu11" gcc -std=gnu11 -o /dev/null "$SRC/hello.c"
run "gcc -m32" gcc -m32 -o /dev/null "$SRC/hello.c" 2>/dev/null || true
run "gcc -fPIC" gcc -fPIC -c -o "$SRC/hello_pic.o" "$SRC/hello.c"
run "gcc -fPIE -pie" gcc -fPIE -pie -o "$SRC/hello_pie" "$SRC/hello.c"
run "gcc -static" gcc -static -o "$SRC/hello_static" "$SRC/hello.c" 2>/dev/null || true
run "gcc -shared lib" gcc -fPIC -shared -o "$SRC/libhello.so" "$SRC/lib.c"
run "gcc use .so" gcc -o "$SRC/use_lib" "$SRC/use_lib.c" -L"$SRC" -lhello -I"$SRC" 2>/dev/null || true
run "gcc -fsanitize=address" gcc -fsanitize=address -g -o "$SRC/hello_asan" "$SRC/hello.c" 2>/dev/null || true
run "gcc -fsanitize=undefined" gcc -fsanitize=undefined -g -o "$SRC/hello_ubsan" "$SRC/hello.c" 2>/dev/null || true
run "gcc -pg profiling" gcc -pg -o "$SRC/hello_prof" "$SRC/hello.c"
run "gcc -fstack-protector" gcc -fstack-protector-strong -o "$SRC/hello_ssp" "$SRC/hello.c"
run "gcc -D define" gcc -DVERSION=42 -o /dev/null "$SRC/hello.c"
run "gcc -I include" gcc -I"$SRC" -o /dev/null "$SRC/hello.c"
run "gcc network" gcc -o "$SRC/network" "$SRC/network.c"
run "gcc run network" "$SRC/network" 2>/dev/null || true
run "gcc fork" gcc -o "$SRC/fork_test" "$SRC/fork_test.c"
run "gcc run fork" "$SRC/fork_test"
run "gcc threads" gcc -o "$SRC/threads" "$SRC/threads.c" -lpthread
run "gcc run threads" "$SRC/threads"
run "gcc -M deps" gcc -M "$SRC/hello.c"
run "gcc -MM deps" gcc -MM "$SRC/hello.c"
run "gcc -save-temps" gcc -save-temps -o /dev/null "$SRC/hello.c" 2>/dev/null || true

# ============================================================================
section "Clang — C/C++ compiler"
# ============================================================================
run "clang basic" clang -o "$SRC/hello_clang" "$SRC/hello.c" 2>/dev/null
run "clang -O2" clang -O2 -o "$SRC/hello_clang_O2" "$SRC/hello.c" 2>/dev/null
run "clang -O3" clang -O3 -o "$SRC/hello_clang_O3" "$SRC/hello.c" 2>/dev/null
run "clang -g" clang -g -o "$SRC/hello_clang_g" "$SRC/hello.c" 2>/dev/null
run "clang -S" clang -S -o "$SRC/hello_clang.s" "$SRC/hello.c" 2>/dev/null
run "clang -E" clang -E -o "$SRC/hello_clang.i" "$SRC/hello.c" 2>/dev/null
run "clang -emit-llvm" clang -S -emit-llvm -o "$SRC/hello.ll" "$SRC/hello.c" 2>/dev/null
run "clang -Wall" clang -Wall -Wextra -o /dev/null "$SRC/hello.c" 2>/dev/null
run "clang -fsanitize=address" clang -fsanitize=address -g -o "$SRC/hello_clang_asan" "$SRC/hello.c" 2>/dev/null || true
run "clang -std=c11" clang -std=c11 -o /dev/null "$SRC/hello.c" 2>/dev/null
run "clang++ basic" clang++ -o "$SRC/hellocpp_clang" "$SRC/hello.cpp" 2>/dev/null
run "clang++ -std=c++17" clang++ -std=c++17 -o "$SRC/hellocpp_clang17" "$SRC/hello.cpp" 2>/dev/null

# ============================================================================
section "G++ — C++ compiler"
# ============================================================================
run "g++ basic" g++ -o "$SRC/hellocpp" "$SRC/hello.cpp"
run "g++ run" "$SRC/hellocpp"
run "g++ -O2" g++ -O2 -o "$SRC/hellocpp_O2" "$SRC/hello.cpp"
run "g++ -O3" g++ -O3 -o "$SRC/hellocpp_O3" "$SRC/hello.cpp"
run "g++ -std=c++11" g++ -std=c++11 -o "$SRC/hellocpp11" "$SRC/hello.cpp"
run "g++ -std=c++14" g++ -std=c++14 -o "$SRC/hellocpp14" "$SRC/hello.cpp"
run "g++ -std=c++17" g++ -std=c++17 -o "$SRC/hellocpp17" "$SRC/hello.cpp"
run "g++ -std=c++20" g++ -std=c++20 -o "$SRC/hellocpp20" "$SRC/hello.cpp" 2>/dev/null || true
run "g++ templates" g++ -std=c++17 -O2 -o "$SRC/templates" "$SRC/templates.cpp"
run "g++ run templates" "$SRC/templates"
run "g++ -Wall" g++ -Wall -Wextra -o /dev/null "$SRC/hello.cpp"
run "g++ -g" g++ -g -o "$SRC/hellocpp_g" "$SRC/hello.cpp"
run "g++ -static" g++ -static -o "$SRC/hellocpp_static" "$SRC/hello.cpp" 2>/dev/null || true

# ============================================================================
section "NASM — assembler"
# ============================================================================
run "nasm elf64" nasm -f elf64 -o "$SRC/hello_asm.o" "$SRC/hello.asm" 2>/dev/null
run "nasm elf32" nasm -f elf32 -o "$SRC/hello_asm32.o" "$SRC/hello.asm" 2>/dev/null || true
run "ld link asm" ld -o "$SRC/hello_asm" "$SRC/hello_asm.o" 2>/dev/null
run "run asm" "$SRC/hello_asm" 2>/dev/null

# GNU as
run "gcc -S then as" gcc -S -o "$SRC/gas_test.s" "$SRC/hello.c" && as -o "$SRC/gas_test.o" "$SRC/gas_test.s" 2>/dev/null

# ============================================================================
section "Go compiler"
# ============================================================================
run "go build" go build -o "$SRC/hello_go" "$SRC/hello.go" 2>/dev/null
run_t 30 "go run" go run "$SRC/hello.go" 2>/dev/null
run "go vet" go vet "$SRC/hello.go" 2>/dev/null || true
run "go build -v" go build -v -o "$SRC/hello_go_v" "$SRC/hello.go" 2>/dev/null
run "go build -race" go build -race -o "$SRC/hello_go_race" "$SRC/hello.go" 2>/dev/null
run "go build -ldflags" go build -ldflags="-s -w" -o "$SRC/hello_go_small" "$SRC/hello.go" 2>/dev/null
run "go version" go version
run "go env" go env 2>/dev/null | head -10

# ============================================================================
section "Rust compiler"
# ============================================================================
run "rustc" rustc -o "$SRC/hello_rust" "$SRC/hello.rs" 2>/dev/null
run "rustc run" "$SRC/hello_rust" 2>/dev/null
run "rustc -O" rustc -O -o "$SRC/hello_rust_opt" "$SRC/hello.rs" 2>/dev/null
run "rustc -g" rustc -g -o "$SRC/hello_rust_g" "$SRC/hello.rs" 2>/dev/null
run "rustc --edition 2021" rustc --edition 2021 -o "$SRC/hello_rust_21" "$SRC/hello.rs" 2>/dev/null
run "rustc --emit asm" rustc --emit asm -o "$SRC/hello_rust.s" "$SRC/hello.rs" 2>/dev/null
run "rustc --emit llvm-ir" rustc --emit llvm-ir -o "$SRC/hello_rust.ll" "$SRC/hello.rs" 2>/dev/null
run "rustc version" rustc --version 2>/dev/null
run "cargo init" cargo init "$SRC/rustproj" 2>/dev/null
run "cargo build" cd "$SRC/rustproj" && cargo build 2>/dev/null; cd /
run "cargo build --release" cd "$SRC/rustproj" && cargo build --release 2>/dev/null; cd /
run "cargo run" cd "$SRC/rustproj" && cargo run 2>/dev/null; cd /
run "cargo test" cd "$SRC/rustproj" && cargo test 2>/dev/null; cd /
run "cargo check" cd "$SRC/rustproj" && cargo check 2>/dev/null; cd /
run "cargo clippy" cd "$SRC/rustproj" && cargo clippy 2>/dev/null; cd /

# ============================================================================
section "Java compiler"
# ============================================================================
run "javac" javac -d "$SRC" "$SRC/Hello.java" 2>/dev/null
run "java run" java -cp "$SRC" Hello 2>/dev/null
run "javac -verbose" javac -verbose -d "$SRC" "$SRC/Hello.java" 2>/dev/null
run "javac -source 8" javac -source 8 -target 8 -d "$SRC" "$SRC/Hello.java" 2>/dev/null || true
run "jar create" jar cf "$SRC/Hello.jar" -C "$SRC" Hello.class 2>/dev/null
run "jar list" jar tf "$SRC/Hello.jar" 2>/dev/null
run "java -jar" java -jar "$SRC/Hello.jar" 2>/dev/null || true
run "java version" java -version 2>/dev/null
run "javac version" javac -version 2>/dev/null

# ============================================================================
section "Interpreters — Python, Perl, Ruby, Lua, Node, PHP"
# ============================================================================
run "python3 hello.py" python3 "$SRC/hello.py"
run "python3 -c" python3 -c "print('inline python')"
run "python3 -m compileall" python3 -m compileall "$SRC/hello.py" 2>/dev/null
run "python3 -B" python3 -B "$SRC/hello.py"
run "python3 -O" python3 -O "$SRC/hello.py"
run "python3 -OO" python3 -OO "$SRC/hello.py"
run "python3 -u" python3 -u "$SRC/hello.py"
run "python3 -v" python3 -v -c "import os" 2>/dev/null | tail -5
run "python3 version" python3 --version

run "perl hello.pl" perl "$SRC/hello.pl"
run "perl -e" perl -e 'print "inline perl\n"'
run "perl -w" perl -w "$SRC/hello.pl"
run "perl -c check" perl -c "$SRC/hello.pl"
run "perl version" perl -v | head -5

run "ruby hello.rb" ruby "$SRC/hello.rb"
run "ruby -e" ruby -e 'puts "inline ruby"'
run "ruby -c check" ruby -c "$SRC/hello.rb"
run "ruby -w" ruby -w "$SRC/hello.rb"
run "ruby version" ruby --version

run "lua hello.lua" lua5.4 "$SRC/hello.lua" 2>/dev/null
run "lua -e" lua5.4 -e 'print("inline lua")' 2>/dev/null
run "lua version" lua5.4 -v 2>/dev/null

run "node hello.js" node "$SRC/hello.js"
run "node -e" node -e 'console.log("inline node")'
run "node --check" node --check "$SRC/hello.js"
run "node version" node --version

run "php hello.php" php "$SRC/hello.php"
run "php -r" php -r 'echo "inline php\n";'
run "php -l check" php -l "$SRC/hello.php"
run "php version" php --version

# ============================================================================
section "Make / CMake / Ninja"
# ============================================================================
cat > "$SRC/Makefile" << 'MKEOF'
CC=gcc
CFLAGS=-Wall -O2
all: hello_make fork_make
hello_make: hello.c
	$(CC) $(CFLAGS) -o $@ $<
fork_make: fork_test.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f hello_make fork_make
.PHONY: all clean
MKEOF

run "make" make -C "$SRC" 2>/dev/null
run "make run" "$SRC/hello_make"
run "make -n dry-run" make -n -C "$SRC" 2>/dev/null
run "make -B force" make -B -C "$SRC" 2>/dev/null
run "make -j4" make -j4 -C "$SRC" 2>/dev/null
run "make clean" make -C "$SRC" clean 2>/dev/null

cat > "$SRC/CMakeLists.txt" << 'CMEOF'
cmake_minimum_required(VERSION 3.10)
project(ProvTest C CXX)
add_executable(hello_cmake hello.c)
add_executable(hellocpp_cmake hello.cpp)
CMEOF

run "cmake generate" cmake -B "$SRC/build" -S "$SRC" 2>/dev/null
run "cmake build" cmake --build "$SRC/build" 2>/dev/null
run "cmake build verbose" cmake --build "$SRC/build" --verbose 2>/dev/null
run "cmake -G Ninja" cmake -G Ninja -B "$SRC/build_ninja" -S "$SRC" 2>/dev/null
run "ninja build" ninja -C "$SRC/build_ninja" 2>/dev/null

# ============================================================================
section "Binary analysis tools"
# ============================================================================
run "objdump -d" objdump -d "$SRC/hello" | head -40
run "objdump -D" objdump -D "$SRC/hello" | head -40
run "objdump -t symbols" objdump -t "$SRC/hello" | head -20
run "objdump -T dynamic" objdump -T "$SRC/hello" | head -10
run "objdump -x all headers" objdump -x "$SRC/hello" | head -30
run "objdump -h sections" objdump -h "$SRC/hello"
run "objdump -s full dump" objdump -s "$SRC/hello" | head -20
run "objdump -S source" objdump -S "$SRC/hello_O0" 2>/dev/null | head -30
run "readelf -h" readelf -h "$SRC/hello"
run "readelf -S sections" readelf -S "$SRC/hello"
run "readelf -l program" readelf -l "$SRC/hello"
run "readelf -s symbols" readelf -s "$SRC/hello" | head -20
run "readelf -d dynamic" readelf -d "$SRC/hello"
run "readelf -r reloc" readelf -r "$SRC/hello" | head -10
run "readelf -n notes" readelf -n "$SRC/hello"
run "readelf -a all" readelf -a "$SRC/hello" | head -50
run "nm" nm "$SRC/hello" | head -20
run "nm -D dynamic" nm -D "$SRC/hello" | head -10
run "nm -g global" nm -g "$SRC/hello" | head -10
run "nm -u undefined" nm -u "$SRC/hello"
run "nm --demangle" nm --demangle "$SRC/hellocpp" 2>/dev/null | head -10
run "ldd" ldd "$SRC/hello"
run "ldd verbose" ldd -v "$SRC/hello"
run "ldd cpp" ldd "$SRC/hellocpp"
run "size" size "$SRC/hello"
run "size cpp" size "$SRC/hellocpp"
run "strip" cp "$SRC/hello" "$SRC/hello_stripped" && strip "$SRC/hello_stripped"
run "strip --strip-all" cp "$SRC/hello" "$SRC/hello_strip_all" && strip --strip-all "$SRC/hello_strip_all"
run "strip --strip-debug" cp "$SRC/hello" "$SRC/hello_strip_debug" && strip --strip-debug "$SRC/hello_strip_debug"
run "file binary" file "$SRC/hello"
run "file .so" file "$SRC/libhello.so"
run "file .o" file "$SRC/hello.o"
run "file asm" file "$SRC/hello_asm" 2>/dev/null
run "file java" file "$SRC/Hello.class" 2>/dev/null
run "file go" file "$SRC/hello_go" 2>/dev/null
run "file rust" file "$SRC/hello_rust" 2>/dev/null
run "checksec" checksec --file="$SRC/hello" 2>/dev/null || true

# Cleanup
rm -f /tmp/c_out.txt /tmp/cpp_out.txt /tmp/java_out.txt /tmp/py_compile_out.txt /tmp/ruby_out.txt /tmp/node_out.txt /tmp/php_out.txt

domain_end
