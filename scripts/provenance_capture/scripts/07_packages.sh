#!/bin/bash
# ============================================================================
# DOMAIN 07: PACKAGE MANAGEMENT
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "07 — PACKAGE MANAGEMENT"

section "apt / apt-get / apt-cache"
run "apt list --installed" apt list --installed 2>/dev/null | wc -l
run "apt list --upgradable" apt list --upgradable 2>/dev/null | head -5
run "apt search web server" apt-cache search "web server" | head -10
run "apt search python" apt-cache search python3 | head -10
run "apt show nginx" apt-cache show nginx 2>/dev/null | head -15
run "apt show curl" apt-cache show curl 2>/dev/null | head -15
run "apt showpkg" apt-cache showpkg nginx 2>/dev/null | head -10
run "apt policy" apt-cache policy nginx 2>/dev/null
run "apt policy curl" apt-cache policy curl 2>/dev/null
run "apt depends nginx" apt-cache depends nginx 2>/dev/null | head -10
run "apt depends curl" apt-cache depends curl 2>/dev/null | head -10
run "apt rdepends" apt-cache rdepends curl 2>/dev/null | head -10
run "apt-cache stats" apt-cache stats 2>/dev/null
run "apt-cache pkgnames" apt-cache pkgnames | head -20
run "apt-get install cowsay" apt-get install -y -qq cowsay 2>/dev/null
run "cowsay test" cowsay "provenance test" 2>/dev/null
run "apt-get remove cowsay" apt-get remove -y -qq cowsay 2>/dev/null
run "apt-get autoremove" apt-get autoremove -y -qq 2>/dev/null
run "apt-get clean" apt-get clean 2>/dev/null
run "apt-get autoclean" apt-get autoclean 2>/dev/null

section "dpkg"
run "dpkg -l" dpkg -l | head -20
run "dpkg -l pattern" dpkg -l "lib*" | head -10
run "dpkg -L coreutils" dpkg -L coreutils | head -15
run "dpkg -L bash" dpkg -L bash | head -15
run "dpkg -s bash" dpkg -s bash
run "dpkg -s coreutils" dpkg -s coreutils
run "dpkg -S /usr/bin/ls" dpkg -S /usr/bin/ls
run "dpkg -S /usr/bin/curl" dpkg -S /usr/bin/curl 2>/dev/null
run "dpkg --get-selections" dpkg --get-selections | head -15
run "dpkg --print-architecture" dpkg --print-architecture
run "dpkg-query -W" dpkg-query -W -f='${Package} ${Version}\n' | head -15
run "dpkg-query -l" dpkg-query -l bash
run "dpkg --configure -a" dpkg --configure -a 2>/dev/null

section "pip"
run "pip list" pip list 2>/dev/null | head -20
run "pip list --outdated" pip list --outdated 2>/dev/null | head -5
run "pip list --format=json" pip list --format=json 2>/dev/null | head -5
run "pip show requests" pip show requests 2>/dev/null
run "pip freeze" pip freeze 2>/dev/null | head -15
run "pip check" pip check 2>/dev/null
run "pip install termcolor" pip install -q termcolor 2>/dev/null
run "pip uninstall termcolor" pip uninstall -y termcolor 2>/dev/null

section "npm"
run "npm list -g" npm list -g --depth=0 2>/dev/null
run "npm info express" npm info express version 2>/dev/null
run "npm info lodash" npm info lodash version 2>/dev/null
run "npm -v" npm -v 2>/dev/null
run "npm config list" npm config list 2>/dev/null
run "npm cache ls" npm cache ls 2>/dev/null | head -5

section "gem"
run "gem list" gem list --local 2>/dev/null | head -10
run "gem env" gem env 2>/dev/null | head -10
run "gem install colorize" gem install colorize --no-document 2>/dev/null
run "gem uninstall colorize" gem uninstall colorize -x 2>/dev/null

section "conda"
run "conda list" conda list 2>/dev/null | head -15
run "conda info" conda info 2>/dev/null
run "conda info --envs" conda info --envs 2>/dev/null
run "conda config --show" conda config --show 2>/dev/null | head -10

section "cargo"
run "cargo --version" cargo --version 2>/dev/null
run "cargo install --list" cargo install --list 2>/dev/null | head -5

section "go"
run "go version" go version 2>/dev/null
run "go env" go env 2>/dev/null | head -10

domain_end
