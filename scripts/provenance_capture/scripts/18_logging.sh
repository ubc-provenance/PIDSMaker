#!/bin/bash
# ============================================================================
# DOMAIN 18: LOGGING & AUDIT
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "18 — LOGGING"

run "logger" logger -t prov_test "provenance test"
run "logger -p" logger -p local0.info "info message"
run "logger -p warning" logger -p user.warning "warning message"
run "logger -p error" logger -p user.err "error message"
run "logger -s" logger -s "stderr test" 2>/dev/null
run "logger --id" logger --id=$$ "pid tagged message" 2>/dev/null || true
run "journalctl -n" journalctl -n 5 --no-pager 2>/dev/null
run "journalctl -n 20" journalctl -n 20 --no-pager 2>/dev/null
run "journalctl -u cron" journalctl -u cron -n 3 --no-pager 2>/dev/null
run "journalctl -u ssh" journalctl -u ssh -n 3 --no-pager 2>/dev/null
run "journalctl --since" journalctl --since "1 hour ago" -n 5 --no-pager 2>/dev/null
run "journalctl -p err" journalctl -p err -n 5 --no-pager 2>/dev/null
run "journalctl -k" journalctl -k -n 5 --no-pager 2>/dev/null
run "journalctl --disk-usage" journalctl --disk-usage 2>/dev/null
run "journalctl -f timeout" timeout 2 journalctl -f --no-pager 2>/dev/null || true
run "journalctl -o json" journalctl -n 2 -o json --no-pager 2>/dev/null
run "journalctl --list-boots" journalctl --list-boots 2>/dev/null
run "dmesg" dmesg 2>/dev/null | tail -10
run "dmesg -T" dmesg -T 2>/dev/null | tail -5
run "dmesg -H" dmesg -H 2>/dev/null | tail -5
run "dmesg --level=err" dmesg --level=err 2>/dev/null
run "dmesg -f kern" dmesg -f kern 2>/dev/null | tail -5
run "last" last -5 2>/dev/null
run "last -x" last -x -5 2>/dev/null
run "lastb" lastb -5 2>/dev/null
run "who -b" who -b 2>/dev/null
run "cat syslog" cat /var/log/syslog 2>/dev/null | tail -5
run "cat auth.log" cat /var/log/auth.log 2>/dev/null | tail -5
run "cat kern.log" cat /var/log/kern.log 2>/dev/null | tail -5
run "cat dpkg.log" cat /var/log/dpkg.log 2>/dev/null | tail -5
run "ls /var/log" ls -la /var/log/ | head -20

domain_end
