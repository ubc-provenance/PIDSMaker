#!/bin/bash
# ============================================================================
# DOMAIN 19: SYSTEMD & SERVICES
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "19 — SYSTEMD"

run "systemctl list-units" systemctl list-units --type=service --state=running --no-pager 2>/dev/null | head -20
run "systemctl list-units all" systemctl list-units --type=service --all --no-pager 2>/dev/null | head -20
run "systemctl list-units failed" systemctl list-units --failed --no-pager 2>/dev/null
run "systemctl status ssh" systemctl status ssh --no-pager 2>/dev/null
run "systemctl status cron" systemctl status cron --no-pager 2>/dev/null
run "systemctl status nginx" systemctl status nginx --no-pager 2>/dev/null
run "systemctl show ssh" systemctl show ssh --property=MainPID,ActiveState,SubState --no-pager 2>/dev/null
run "systemctl is-active ssh" systemctl is-active ssh 2>/dev/null
run "systemctl is-enabled ssh" systemctl is-enabled ssh 2>/dev/null
run "systemctl is-active cron" systemctl is-active cron 2>/dev/null
run "systemctl is-enabled cron" systemctl is-enabled cron 2>/dev/null
run "systemctl list-sockets" systemctl list-sockets --no-pager 2>/dev/null
run "systemctl list-timers" systemctl list-timers --no-pager 2>/dev/null
run "systemctl list-dependencies" systemctl list-dependencies ssh --no-pager 2>/dev/null | head -10
run "systemctl cat ssh" systemctl cat ssh --no-pager 2>/dev/null
run "systemctl show" systemctl show --no-pager 2>/dev/null | head -20
run "systemctl daemon-reload" systemctl daemon-reload 2>/dev/null
run "systemctl --version" systemctl --version 2>/dev/null

# Create test service
cat > /tmp/prov_test.service << 'EOF'
[Unit]
Description=Provenance Test Service
[Service]
ExecStart=/bin/echo "hello from service"
Type=oneshot
[Install]
WantedBy=multi-user.target
EOF
run "systemctl link" systemctl link /tmp/prov_test.service 2>/dev/null
run "systemctl enable" systemctl enable prov_test 2>/dev/null
run "systemctl start" systemctl start prov_test 2>/dev/null
run "systemctl status test" systemctl status prov_test --no-pager 2>/dev/null
run "systemctl stop" systemctl stop prov_test 2>/dev/null
run "systemctl disable" systemctl disable prov_test 2>/dev/null
rm -f /tmp/prov_test.service

# Service management
run "service --status-all" service --status-all 2>/dev/null | head -15

domain_end
