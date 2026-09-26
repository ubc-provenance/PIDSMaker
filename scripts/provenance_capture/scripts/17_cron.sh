#!/bin/bash
# ============================================================================
# DOMAIN 17: CRON & SCHEDULING
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "17 — SCHEDULING"

run "crontab -l" crontab -l 2>/dev/null || true
run "crontab set" bash -c 'echo "*/5 * * * * echo prov_test >> /tmp/prov_cron.log" | crontab -' 2>/dev/null
run "crontab -l after" crontab -l 2>/dev/null
run "crontab remove" crontab -r 2>/dev/null || true
run "cat /etc/crontab" cat /etc/crontab 2>/dev/null
run "ls /etc/cron.d" ls -la /etc/cron.d/ 2>/dev/null
run "ls /etc/cron.daily" ls -la /etc/cron.daily/ 2>/dev/null
run "ls /etc/cron.hourly" ls -la /etc/cron.hourly/ 2>/dev/null
run "ls /etc/cron.weekly" ls -la /etc/cron.weekly/ 2>/dev/null
run "ls /etc/cron.monthly" ls -la /etc/cron.monthly/ 2>/dev/null
run "at" bash -c 'echo "echo at_test" | at now + 1 minute' 2>/dev/null || true
run "atq" atq 2>/dev/null || true
run "atrm" atrm 1 2>/dev/null || true
run "batch" bash -c 'echo "echo batch_test" | batch' 2>/dev/null || true
run "systemctl list-timers" systemctl list-timers --no-pager 2>/dev/null | head -10

domain_end
