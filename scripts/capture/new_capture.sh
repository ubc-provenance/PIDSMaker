#!/usr/bin/env bash
# new_capture.sh - capture this machine's provenance with SPADE and stream it to a Kafka topic.
#
# USAGE
#   scripts/capture/new_capture.sh --topic NAME [options]
#
# While it runs, every system call the Linux audit subsystem records is turned into
# provenance by SPADE and published to the topic as it happens. Stop with Ctrl-C.
# Anything that reads the topic sees the provenance live - typically PIDSMaker's
# real-time detector, in another terminal:
#
#   docker compose -f compose-pidsmaker.yml exec pids \
#     python pidsmaker/stream_detect.py orthrus SPADE_AUDIT \
#       --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml \
#       --stream_topic=NAME --stream_from_beginning=False --emit_viz=True
#
# Run it on the host, as a user who can sudo (the audit log is root-only). SPADE must be
# built (see docs/docs/features/streaming.md); the script starts it when it is not running.
#
# REQUIRED
#   --topic NAME       Kafka topic to publish to. Reuse a name to keep adding to it;
#                      pick a new one (or --fresh) to start over.
#
# WHAT IS CAPTURED (default: every process of this login session)
#   --user NAME        Every process of that user, in all of their sessions.
#   --system-wide      Every process on the machine.
#   --ignore NAMES     Drop these processes (comma-separated names, e.g. node,code) -
#                      for an IDE or agent whose constant I/O would swamp the capture.
#
# OPTIONS
#   --fresh            Empty the topic first.
#   --dataset NAME     Dataset name used in the printed PIDSMaker commands (default SPADE_AUDIT;
#                      any name works - its database and dataset.yml take the lowercased name).
#   --from-log FILE    Publish an audit log (ausearch --raw output) instead of capturing live.
#   --check            Only run the checks, change nothing.
#   --spade-home DIR   SPADE checkout (default: $SPADE_HOME, or ~/SPADE).
#   --java-home DIR    JDK 21+ for SPADE (default: $JAVA_HOME, then the usual locations).
#   --kafka-server H:P Broker as seen from the host (default: localhost:KAFKA_PORT or 29092).
#   --audit-log FILE   auditd's log file (default: log_file in /etc/audit/auditd.conf).
#   --audit-key KEY    Key tagging this run's audit rules (default: pidsmaker_<timestamp>).
#   -h, --help         This text.
set -uo pipefail

# ------------------------------------------------------------------ options ---
PIDS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KAFKA_COMPOSE="$PIDS_HOME/compose-kafka.yml"
ENV_FILE="$PIDS_HOME/.env"
# Under sudo, "home" must still be the invoking user's.
USER_HOME="$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)"; USER_HOME="${USER_HOME:-$HOME}"

TOPIC="" FRESH=0 SCOPE=session SCOPE_USER="" FROM_LOG="" CHECK_ONLY=0
SPADE_HOME="${SPADE_HOME:-$USER_HOME/SPADE}"
JAVA_HOME_OPT="${JAVA_HOME:-}"
KAFKA_SERVER="" AUDIT_LOG="" AUDIT_KEY="" IGNORE=""
DATASET="SPADE_AUDIT" SYSTEM="orthrus"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "${BASH_SOURCE[0]}"; exit 0; }
need_value() { [ -n "${2:-}" ] && [ "${2#-}" = "$2" ] || { echo "error: $1 needs a value" >&2; exit 2; }; }

while [ $# -gt 0 ]; do
  case "$1" in
    --topic)        need_value "$1" "${2:-}"; TOPIC="$2"; shift ;;
    --fresh)        FRESH=1 ;;
    --dataset)      need_value "$1" "${2:-}"; DATASET="$2"; shift ;;
    --user)         need_value "$1" "${2:-}"; SCOPE=user; SCOPE_USER="$2"; shift ;;
    --system-wide)  SCOPE=host ;;
    --ignore)       need_value "$1" "${2:-}"; IGNORE="$2"; shift ;;
    --from-log)     need_value "$1" "${2:-}"; FROM_LOG="$2"; shift ;;
    --check)        CHECK_ONLY=1 ;;
    --spade-home)   need_value "$1" "${2:-}"; SPADE_HOME="$2"; shift ;;
    --java-home)    need_value "$1" "${2:-}"; JAVA_HOME_OPT="$2"; shift ;;
    --kafka-server) need_value "$1" "${2:-}"; KAFKA_SERVER="$2"; shift ;;
    --audit-log)    need_value "$1" "${2:-}"; AUDIT_LOG="$2"; shift ;;
    --audit-key)    need_value "$1" "${2:-}"; AUDIT_KEY="$2"; shift ;;
    -h|--help)      usage ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ -t 1 ]; then c_g=$'\033[32m'; c_y=$'\033[33m'; c_r=$'\033[31m'; c_b=$'\033[1;36m'; c_0=$'\033[0m'
else c_g=""; c_y=""; c_r=""; c_b=""; c_0=""; fi
step() { echo; echo "${c_b}==> $*${c_0}"; }
say()  { echo "    $*"; }
ok()   { echo "  ${c_g}✓${c_0} $*"; }
warn() { echo "  ${c_y}!${c_0} $*"; }
hint() { echo "  ${c_y}→${c_0} $*"; }
die()  { echo; echo "${c_r}error:${c_0} $*" >&2; exit 1; }

if [ -z "$TOPIC" ]; then
  echo "error: --topic NAME is required." >&2
  echo "       The topic is where the provenance is published and what the detector reads" >&2
  echo "       (--stream_topic=NAME). Reuse a name to keep adding to it, or pick a new one." >&2
  echo "       Example:  $0 --topic benign" >&2
  exit 2
fi
[[ "$TOPIC" =~ ^[A-Za-z0-9._-]+$ ]] || die "topic '$TOPIC' may only contain letters, digits, '.', '_' and '-'."
[ -z "$FROM_LOG" ] || [ -f "$FROM_LOG" ] || die "--from-log: no such file: $FROM_LOG"

env_value() { [ -f "$ENV_FILE" ] && sed -n "s/^$1=//p" "$ENV_FILE" | tail -1 | tr -d '"'"'"; }
[ -n "$KAFKA_SERVER" ] || KAFKA_SERVER="localhost:$(env_value KAFKA_PORT)"; KAFKA_SERVER="${KAFKA_SERVER%:}"
[ "$KAFKA_SERVER" != "localhost" ] || KAFKA_SERVER="localhost:29092"
[ -n "$AUDIT_KEY" ] || AUDIT_KEY="pidsmaker_$(date +%Y%m%d_%H%M%S)"
DB_NAME="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')"
DATASET_CONFIG="/home/artifacts/streaming/$DB_NAME/dataset.yml"   # path inside the container
LIVE=1; [ -z "$FROM_LOG" ] || LIVE=0

# ------------------------------------------------------------------ helpers ---
if [ "$(id -u)" = 0 ]; then as_root() { "$@"; }; as_root_quiet() { "$@"; }
else as_root() { sudo "$@"; }; as_root_quiet() { sudo -n "$@"; }; fi

java_major() { "$1" -version 2>&1 | head -1 | sed -E 's/.*version "([0-9]+).*/\1/'; }
find_java() {
  shopt -s nullglob
  local c candidates=()
  [ -n "$JAVA_HOME_OPT" ] && candidates+=("$JAVA_HOME_OPT/bin/java")
  command -v java >/dev/null 2>&1 && candidates+=("$(command -v java)")
  candidates+=(/usr/lib/jvm/*/bin/java /usr/lib/jvm/*/*/bin/java /opt/*/bin/java /usr/local/*/bin/java "$USER_HOME"/.sdkman/candidates/java/*/bin/java)
  shopt -u nullglob
  for c in "${candidates[@]}"; do
    [ -x "$c" ] || continue
    local v; v="$(java_major "$c")"
    [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -ge 21 ] && { readlink -f "$c"; return 0; }
  done
  return 1
}
SPADE_JAVA_HOME=""
CTL_TIMEOUT="${CTL_TIMEOUT:-45}"
spade_env() { JAVA_HOME="$SPADE_JAVA_HOME" PATH="$SPADE_JAVA_HOME/bin:$PATH" "$@"; }
_kill_tree() { local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do _kill_tree "$c"; done; kill -KILL "$p" 2>/dev/null; }
# SPADE's control client, bounded: `bin/spade` is a wrapper around a java child, and a
# plain `timeout` would leave that child orphaned and wedged, so the whole tree is killed.
spade_ctl() {
  local input tmp pid waited=0
  input="$(cat)"; tmp="$(mktemp)"
  ( cd "$SPADE_HOME" && spade_env bin/spade control >"$tmp" 2>&1 <<<"$input" ) &
  pid=$!
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$CTL_TIMEOUT" ]; do sleep 1; waited=$((waited + 1)); done
  kill -0 "$pid" 2>/dev/null && _kill_tree "$pid"
  wait "$pid" 2>/dev/null
  cat "$tmp"; rm -f "$tmp"
}
spade_list() { printf 'list all\nexit\n' | spade_ctl; }
spade_reachable() { spade_list | grep -q "storage(s) added\|No storages added\|reporter(s) added\|No reporters added"; }
spade_up() { local i; for i in 1 2 3; do spade_reachable && return 0; sleep 2; done; return 1; }
spade_kernel_pid() { local p; p="$(cat "$SPADE_HOME/spade.pid" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }
port_19999_pid() { ss -ltnp 2>/dev/null | sed -n 's/.*:19999 .*pid=\([0-9]\+\).*/\1/p' | head -1; }
spade_eof_logged() { [ -f "$SPADE_LOG" ] && tail -n +"$((LOG_MARK + 1))" "$SPADE_LOG" 2>/dev/null | grep -q "Reached the end of file"; }

kafka_cli() { docker compose -f "$KAFKA_COMPOSE" exec -T kafka /opt/kafka/bin/"$1" --bootstrap-server localhost:9092 "${@:2}" 2>/dev/null; }
topic_exists() { kafka_cli kafka-topics.sh --list | grep -qx "$TOPIC"; }
topic_offset() { kafka_cli kafka-get-offsets.sh --topic "$TOPIC" | awk -F: '{s+=$3} END{print s+0}'; }
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
fmt() { printf "%'d" "$1" 2>/dev/null || echo "$1"; }

# ------------------------------------------------------------------ cleanup ---
ARMED=0 FEED_PID="" SENTINEL_PID="" KEEPALIVE_PID="" REPORTER_ADDED=0 FIFO_DIR=""
SPADE_LOG="" LOG_MARK=0
disarm_rules() {
  [ "$ARMED" = 1 ] || return 0
  as_root auditctl -D -k "$AUDIT_KEY" >/dev/null 2>&1 && ok "audit rules removed" || warn "could not remove the audit rules; run:  sudo auditctl -D -k $AUDIT_KEY"
  ARMED=0
}
stop_feed() {
  [ -n "$SENTINEL_PID" ] && { kill "$SENTINEL_PID" 2>/dev/null; SENTINEL_PID=""; }
  [ -n "$FEED_PID" ] && { wait "$FEED_PID" 2>/dev/null; FEED_PID=""; }
}
detach_reporter() {
  [ "$REPORTER_ADDED" = 1 ] || return 0
  printf 'remove reporter Audit\nexit\n' | spade_ctl >/dev/null
  REPORTER_ADDED=0
}
cleanup() {
  disarm_rules
  stop_feed
  detach_reporter
  [ -n "$KEEPALIVE_PID" ] && kill "$KEEPALIVE_PID" 2>/dev/null
  [ -n "$FIFO_DIR" ] && rm -rf "$FIFO_DIR" 2>/dev/null
  return 0
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---------------------------------------------------------------- preflight ---
preflight() {
  step "Checking the setup"
  local fail=0

  if [ "$LIVE" = 1 ]; then
    if [ -z "$AUDIT_LOG" ]; then
      AUDIT_LOG="$(as_root_quiet sed -n 's/^log_file *= *//p' /etc/audit/auditd.conf 2>/dev/null | tail -1)"
      AUDIT_LOG="${AUDIT_LOG:-/var/log/audit/audit.log}"
    fi
    # Reading the audit log and managing rules both need root. One prompt now, if any.
    if [ "$(id -u)" != 0 ] && [ ! -r "$AUDIT_LOG" ] && ! sudo -n tail -c0 "$AUDIT_LOG" >/dev/null 2>&1; then
      say "sudo is needed to read $AUDIT_LOG and to manage audit rules."
      if [ -t 0 ]; then sudo -v || true; fi
      sudo -n tail -c0 "$AUDIT_LOG" >/dev/null 2>&1 || {
        warn "cannot read $AUDIT_LOG as root without a password prompt"
        hint "run  sudo -v  first (or run this script with sudo), then try again"; fail=1; }
    fi
    if command -v auditctl >/dev/null 2>&1 || [ -x /usr/sbin/auditctl ] || [ -x /sbin/auditctl ]; then
      ok "audit tools installed"
    else
      warn "auditctl not found - install the audit package (e.g. sudo apt install auditd)"; fail=1
    fi
    if systemctl is-active auditd >/dev/null 2>&1; then ok "auditd is running"
    else warn "auditd is not running - start it:  sudo systemctl restart auditd"; fail=1; fi
    # A daemon that suspended on low disk stays "active" but writes nothing until restarted.
    if command -v journalctl >/dev/null 2>&1; then
      local jstate
      jstate="$(as_root_quiet journalctl -u auditd -n 60 --no-pager 2>/dev/null \
        | grep -ioE 'suspending logging|resuming logging|rotating logs|Started Security Auditing Service' | tail -1)"
      if echo "$jstate" | grep -qi suspend; then
        warn "auditd has SUSPENDED logging (usually low disk): it records nothing until restarted"
        hint "free space if needed, then:  sudo systemctl restart auditd"; fail=1
      fi
    fi
    local enabled; enabled="$(as_root auditctl -s 2>/dev/null | awk '/^enabled/{print $2}')"
    case "${enabled:-?}" in
      1) ok "kernel auditing enabled" ;;
      2) warn "audit rules are locked until reboot (auditctl -e 2): no rule can be added"; fail=1 ;;
      0) warn "kernel auditing is disabled - enable it:  sudo auditctl -e 1"; fail=1 ;;
      *) warn "cannot query the audit status (sudo auditctl -s failed)"; fail=1 ;;
    esac
    case "$SCOPE" in
      session)
        local sid; sid="$(cat /proc/self/sessionid 2>/dev/null)"
        if [ -z "$sid" ] || [ "$sid" = 4294967295 ]; then
          warn "this shell has no login session id, so a session-scoped capture would be empty"
          hint "use --user NAME or --system-wide"; fail=1
        else ok "capturing this login session ($sid); --user NAME or --system-wide widen it"; fi ;;
      user)
        if id -u "$SCOPE_USER" >/dev/null 2>&1; then ok "capturing every process of user $SCOPE_USER"
        else warn "unknown user: $SCOPE_USER"; fail=1; fi ;;
      host) ok "capturing every process on the machine (--system-wide)" ;;
    esac
    local free; free="$(free_gb "$(dirname "$AUDIT_LOG")")"
    if [ -n "$free" ] && [ "$free" -ge 1 ]; then ok "${free}G free where auditd writes"
    else warn "less than 1G free on $(dirname "$AUDIT_LOG") - auditd stops logging when its disk runs low"; fail=1; fi
    local k
    for k in $(as_root auditctl -l 2>/dev/null | grep -o 'key=pidsmaker_[A-Za-z0-9_]*' | sort -u | cut -d= -f2); do
      [ "$k" = "$AUDIT_KEY" ] && continue
      as_root auditctl -D -k "$k" >/dev/null 2>&1 && warn "removed audit rules left by an earlier run ($k)"
    done
  else
    ok "publishing $FROM_LOG ($(grep -c '^type=' "$FROM_LOG" 2>/dev/null || echo 0) audit records)"
  fi

  # SPADE
  [ -f "$SPADE_HOME/lib/spade.jar" ] && ok "SPADE build found in $SPADE_HOME" \
    || { warn "no SPADE build in $SPADE_HOME (expected lib/spade.jar)"; hint "build SPADE first (see the real-time detection docs), or pass --spade-home DIR"; fail=1; }
  local java; java="$(find_java)"
  if [ -n "$java" ]; then SPADE_JAVA_HOME="$(dirname "$(dirname "$java")")"; ok "JDK $(java_major "$java") found for SPADE"
  else warn "no JDK 21 or newer found - SPADE needs one"; hint "install one (e.g. sudo apt install openjdk-21-jdk) or pass --java-home DIR"; fail=1; fi
  if [ -n "$java" ] && [ -f "$SPADE_HOME/lib/spade.jar" ]; then
    if spade_up; then ok "SPADE server is running"
    else
      local kpid ppid; kpid="$(spade_kernel_pid)"
      if [ -n "$kpid" ]; then
        ppid="$(port_19999_pid)"
        if [ -n "$ppid" ] && [ "$ppid" != "$kpid" ]; then
          warn "SPADE control port 19999 is held by pid $ppid, not this build's kernel (pid $kpid)"
          hint "stop the stray one:  kill $ppid   then  (cd $SPADE_HOME && bin/spade start)"; fail=1
        else
          warn "a SPADE kernel (pid $kpid) is running but its control client did not answer"
          hint "restart it:  (cd $SPADE_HOME && bin/spade stop && bin/spade start)"; fail=1
        fi
      else say "SPADE is not running: it will be started."; fi
    fi
  fi

  # Kafka
  if [ -f "$KAFKA_COMPOSE" ] && kafka_cli kafka-topics.sh --list >/dev/null; then ok "Kafka broker is up"
  else warn "cannot reach the Kafka broker"; hint "start it with:  docker compose -f compose-kafka.yml up -d"; fail=1; fi
  if (exec 3<>"/dev/tcp/${KAFKA_SERVER%:*}/${KAFKA_SERVER##*:}") 2>/dev/null; then ok "broker reachable from the host at $KAFKA_SERVER"
  else warn "nothing listens on $KAFKA_SERVER, where SPADE will publish"; hint "pass --kafka-server HOST:PORT (the broker's EXTERNAL listener)"; fail=1; fi

  [ "$fail" = 0 ] || die "fix the items marked '!' above, then run again (--check only runs these checks)."
  ok "all checks passed"
}

# --------------------------------------------------------------------- SPADE ---
ensure_spade() {
  spade_up && return 0
  say "Starting SPADE..."
  ( cd "$SPADE_HOME" && spade_env bin/spade start >/dev/null 2>&1 )
  local i; for i in $(seq 1 45); do sleep 2; spade_up && { ok "SPADE server started"; return 0; }; done
  die "SPADE did not come up; see $SPADE_HOME/log/current.log (a JDK older than 21 on PATH is the usual cause)."
}
ensure_topic() {
  if [ "$FRESH" = 1 ] && topic_exists; then
    say "Emptying topic '$TOPIC' (--fresh)..."
    kafka_cli kafka-topics.sh --delete --topic "$TOPIC" >/dev/null
    local i; for i in $(seq 1 30); do topic_exists || break; sleep 1; done
    topic_exists && die "could not delete topic '$TOPIC'"
  fi
  if ! topic_exists; then
    kafka_cli kafka-topics.sh --create --if-not-exists --topic "$TOPIC" --partitions 1 --replication-factor 1 >/dev/null \
      || die "could not create topic '$TOPIC'"
    ok "topic '$TOPIC' created"
  else
    ok "topic '$TOPIC' already holds $(fmt "$(topic_offset)") records; this capture adds to them (--fresh starts over)"
  fi
}
ensure_storage() {
  # Exactly one Kafka storage, pointed at this topic; settings go as arguments, so
  # nothing in SPADE's cfg/ is touched.
  local storage want="kafka.output.server=$KAFKA_SERVER kafka.output.topic=$TOPIC"
  storage="$(spade_list | grep -m1 -E '^[[:space:]]*[0-9]+\. Kafka')"
  if [ "$FRESH" = 0 ] && [[ "$storage" == *"$want"* ]]; then ok "SPADE publishes to '$TOPIC'"; return 0; fi
  [ -z "$storage" ] || printf 'remove storage Kafka\nexit\n' | spade_ctl >/dev/null
  printf 'add storage Kafka %s\nexit\n' "$want" | spade_ctl | grep -q "Adding storage Kafka... done" \
    || die "SPADE could not attach its Kafka storage (see $SPADE_HOME/log/current.log)."
  ok "SPADE publishes to '$TOPIC' via $KAFKA_SERVER"
}
attach_reporter() {  # $1 = path SPADE reads (the pipe, or a log file)
  if spade_list | grep -qE '^[[:space:]]*[0-9]+\. Audit'; then
    say "Detaching an Audit reporter left by a previous run..."
    printf 'remove reporter Audit\nexit\n' | spade_ctl >/dev/null
  fi
  SPADE_LOG="$SPADE_HOME/log/current.log"; LOG_MARK=0
  [ -f "$SPADE_LOG" ] && LOG_MARK="$(wc -l < "$SPADE_LOG")"
  local out
  local ignore="kauditd,auditd,audispd,ausearch,auditctl${IGNORE:+,$IGNORE}"
  out="$(printf 'add reporter Audit inputLog=%s fileIO=true netIO=true ignoreProcesses=%s\nexit\n' "$1" "$ignore" | spade_ctl)"
  echo "$out" | grep -q "Adding reporter Audit... done" || {
    echo "$out" | tail -5 | sed 's/^/      /'
    die "SPADE could not start its Audit reporter (see $SPADE_HOME/log/current.log)."
  }
  REPORTER_ADDED=1
}
wait_for_spade_drain() {  # after the input ended: until SPADE logged EOF and the topic is flat
  local prev flat=0 i now
  prev="$(topic_offset)"
  for i in $(seq 1 30); do
    sleep 2; now="$(topic_offset)"
    if [ "$now" = "$prev" ]; then flat=$((flat + 1)); else flat=0; fi
    prev="$now"
    spade_eof_logged && [ "$flat" -ge 2 ] && return 0
    [ "$flat" -ge 10 ] && return 0
  done
}

# ------------------------------------------------------------------ capture ---
print_consumers() {
  say "Read it live from another terminal - score it with a trained model:"
  echo "      docker compose -f compose-pidsmaker.yml exec pids \\"
  echo "        python pidsmaker/stream_detect.py $SYSTEM $DATASET --dataset_config=$DATASET_CONFIG \\"
  echo "        --stream_topic=$TOPIC --stream_from_beginning=False --emit_viz=True"
  say "or turn it into a dataset to train on (reads the whole topic, stops when it is quiet):"
  echo "      docker compose -f compose-pidsmaker.yml exec pids \\"
  echo "        python pidsmaker/stream_ingest.py $DATASET --stream_topic=$TOPIC --stream_idle_timeout=25"
}

start_feed() {
  # audit log -> (this run's records only) -> named pipe, which SPADE reads as its input
  # log. A pipe blocks SPADE until data arrives, so the feed is continuous and ends
  # only when the writer closes it - that is how Ctrl-C stops the capture cleanly.
  FIFO_DIR="$(mktemp -d /tmp/pidsmaker_capture.XXXXXX)"; chmod 755 "$FIFO_DIR"
  FIFO="$FIFO_DIR/audit.pipe"; mkfifo "$FIFO"; chmod 644 "$FIFO"
  sleep infinity & SENTINEL_PID=$!         # tail follows this; killing it ends the feed
  local reader=(tail -n 0 -F --pid="$SENTINEL_PID" "$AUDIT_LOG")
  [ -r "$AUDIT_LOG" ] || reader=(sudo -n "${reader[@]}")
  # Records are grouped by their audit serial; the SYSCALL record comes first and carries
  # the key, so the decision made on it is applied to the rest of its event.
  (
    exec > "$FIFO"                          # blocks until SPADE opens the pipe for reading
    "${reader[@]}" 2>/dev/null | awk -v key="$AUDIT_KEY" '
      BEGIN { sep = sprintf("%c", 1) }
      { if (match($0, /msg=audit\([0-9.]+:[0-9]+\)/)) s = substr($0, RSTART, RLENGTH); else next }
      !(s in keep) {
        keep[s] = ($0 ~ /^type=SYSCALL / && (index($0, "key=\"" key "\"") || index($0, sep key sep) || index($0, "\"" key sep) || index($0, sep key "\"")))
        order[++n] = s
        if (n - head > 20000) { delete keep[order[++head]]; delete order[head] }
      }
      keep[s] { print; fflush() }'
  ) & FEED_PID=$!
}

arm_rules() {
  # Exclude, by pid, everything that moves the feed itself: otherwise SPADE reading the
  # pipe and tail reading the log would record each other forever.
  local excl="" p c
  for p in $$ "$FEED_PID" $(pgrep -P "$FEED_PID" 2>/dev/null) "$(spade_kernel_pid)" $(pgrep -f "spadeAuditBridge" 2>/dev/null); do
    [ -n "$p" ] && excl="$excl -F pid!=$p"
  done
  for p in $(pgrep -P "$FEED_PID" 2>/dev/null); do    # tail and awk live one level below the feed
    for c in $(pgrep -P "$p" 2>/dev/null); do excl="$excl -F pid!=$c"; done
  done
  local filters="-F arch=b64 -F ppid!=$$ $excl"
  case "$SCOPE" in
    session) filters="$filters -F sessionid=$(cat /proc/self/sessionid)" ;;
    user)    filters="$filters -F auid=$(id -u "$SCOPE_USER")" ;;
  esac
  # SPADE's own syscall set with file and network I/O; the -S list must precede -k.
  # shellcheck disable=SC2086
  as_root auditctl -a exit,always $filters -F success=1 \
    -S read,readv,pread,preadv,write,writev,pwrite,pwritev,sendmsg,sendto,recvmsg,recvfrom,\
bind,accept,accept4,socket,clone,fork,vfork,execve,open,openat,creat,close,dup,dup2,dup3,\
rename,renameat,unlink,unlinkat,chmod,fchmod,pipe,pipe2,socketpair,truncate,ftruncate,\
link,symlink,ptrace \
    -k "$AUDIT_KEY" >/dev/null || die "could not add the audit rules (sudo auditctl failed)"
  ARMED=1
  # shellcheck disable=SC2086
  as_root auditctl -a exit,always $filters -S exit,exit_group,connect,kill -k "$AUDIT_KEY" >/dev/null \
    || die "could not add the audit rules (sudo auditctl failed)"
}

capture_live() {
  step "Starting the capture"
  if [ "$(id -u)" != 0 ]; then ( while sudo -n -v 2>/dev/null; do sleep 60; done ) & KEEPALIVE_PID=$!; fi
  ensure_spade; ensure_topic; ensure_storage
  start_feed
  attach_reporter "$FIFO"
  local i; for i in $(seq 1 20); do [ -n "$(pgrep -P "$FEED_PID" 2>/dev/null)" ] && break; sleep 0.5; done
  [ -n "$(pgrep -P "$FEED_PID" 2>/dev/null)" ] || die "SPADE did not open the pipe; see $SPADE_HOME/log/current.log"
  ok "SPADE is reading the audit stream"
  arm_rules
  ok "audit rules armed (key $AUDIT_KEY)"
  [ -z "$IGNORE" ] || ok "SPADE drops processes named: $IGNORE"

  local before; before="$(topic_offset)"
  step "Capturing to topic '$TOPIC' - Ctrl-C to stop"
  print_consumers
  echo
  local stop=0 start now last=0 count
  start="$(date +%s)"
  trap 'stop=1' INT TERM
  while [ "$stop" = 0 ]; do
    sleep 1
    if ! kill -0 "$FEED_PID" 2>/dev/null; then
      warn "the feed stopped (SPADE closed the pipe?) - see $SPADE_HOME/log/current.log"; break
    fi
    now="$(date +%s)"
    if [ $((now - last)) -ge 30 ]; then
      last=$now; count="$(topic_offset)"
      say "$(date +%H:%M:%S)  topic '$TOPIC': $(fmt "$count") records (+$(fmt $((count - before))) this capture, $(( (now - start) / 60 )) min)"
    fi
  done
  trap 'exit 130' INT TERM

  step "Stopping"
  disarm_rules
  stop_feed                                     # closes the pipe -> SPADE reaches EOF
  say "Waiting for SPADE to publish what it has..."
  wait_for_spade_drain
  detach_reporter
  local after; after="$(topic_offset)"
  ok "topic '$TOPIC' holds $(fmt "$after") records (+$(fmt $((after - before))) from this capture)"
}

publish_log() {
  step "Publishing $FROM_LOG"
  ensure_spade; ensure_topic; ensure_storage
  local before; before="$(topic_offset)"
  attach_reporter "$(readlink -f "$FROM_LOG")"
  say "SPADE is parsing the log..."
  local prev="$before" now flat=0 grew=0 tick=0
  while :; do
    sleep 3; tick=$((tick + 3)); now="$(topic_offset)"
    [ "${now:-0}" -gt "$before" ] && grew=1
    if [ "$now" = "$prev" ]; then flat=$((flat + 1)); else flat=0; fi; prev="$now"
    [ $((tick % 15)) = 0 ] && [ "$grew" = 1 ] && say "topic '$TOPIC': $(fmt "$now") records (+$(fmt $((now - before))))"
    if [ "$grew" = 1 ]; then
      { spade_eof_logged && [ "$flat" -ge 2 ]; } && break
      [ "$flat" -ge 10 ] && break
    elif [ "$tick" -ge 90 ]; then
      warn "SPADE published nothing in 90s; giving up"; hint "see $SPADE_HOME/log/current.log"; break
    fi
  done
  detach_reporter
  local after; after="$(topic_offset)"
  [ "$after" = "$before" ] && warn "the topic did not grow: SPADE published nothing from this log (see $SPADE_HOME/log/current.log)" \
    || ok "topic '$TOPIC' holds $(fmt "$after") records (+$(fmt $((after - before))) from this log)"
}

# --------------------------------------------------------------------- main ---
preflight
[ "$CHECK_ONLY" = 0 ] || { echo; say "Check only: nothing was changed."; exit 0; }
if [ "$LIVE" = 1 ]; then capture_live; else publish_log; fi
step "Done"
say "SPADE keeps running; run this again with --topic $TOPIC to keep adding to the topic."
echo
print_consumers
