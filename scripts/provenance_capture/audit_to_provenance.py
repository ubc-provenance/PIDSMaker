#!/usr/bin/env python3
"""
Convert auditd logs to provenance graph format — v2.

Handles the expanded rule set including:
  - prov_exec → EVENT_EXECUTE
  - prov_clone → EVENT_CLONE
  - prov_connect → EVENT_CONNECT
  - prov_accept → EVENT_ACCEPT (incoming connections)
  - prov_sendto → EVENT_SENDTO
  - prov_sendmsg → EVENT_SENDMSG
  - prov_recvfrom → EVENT_RECVFROM
  - prov_recvmsg → EVENT_RECVMSG
  - prov_openat → EVENT_READ / EVENT_WRITE (based on flags)
  - prov_create → EVENT_WRITE (file/dir creation)
  - prov_delete → EVENT_WRITE (file deletion)
  - prov_rename → EVENT_WRITE (file rename)
  - prov_chmod / prov_chown → EVENT_WRITE (attribute change)
  - prov_ptrace → EVENT_OPEN (process inspection)
  - prov_pipe / prov_dup → (tracked for process linkage)
  - prov_file_* → EVENT_READ / EVENT_WRITE (legacy file watches, if present)

Reads audit.log and produces:
  - entities.tsv: (entity_id, entity_type, entity_text)
  - edges.tsv:    (src_id, event_type, dst_id, timestamp)

Usage:
    # Parse all rotated + current logs
    sudo bash -c 'cat /var/log/audit/audit.log.* /var/log/audit/audit.log 2>/dev/null' | \
        python3 audit_to_provenance_v2.py --input - --output provenance_data/

    # Parse single file
    python3 audit_to_provenance_v2.py --input /var/log/audit/audit.log --output provenance_data/
"""

import argparse
import os
import re
import struct
import socket
import sys
from collections import defaultdict


def parse_audit_line(line):
    """Parse a single auditd log line into a dict."""
    result = {}

    m = re.match(r'type=(\w+)', line)
    if m:
        result['type'] = m.group(1)

    m = re.search(r'msg=audit\((\d+\.\d+):(\d+)\)', line)
    if m:
        result['timestamp'] = float(m.group(1))
        result['event_id'] = m.group(2)

    # Handle multiple keys: key="prov_openat" or key=(null)
    # The 'key' field can appear as key="val" or key=(null)
    key_match = re.search(r'\bkey="([^"]*)"', line)
    if key_match:
        result['_audit_key'] = key_match.group(1)

    for m in re.finditer(r'(\w+)=("([^"]*)"|(\S+))', line):
        key = m.group(1)
        val = m.group(3) if m.group(3) is not None else m.group(4)
        if key not in result:
            result[key] = val

    return result


def parse_sockaddr(saddr_hex):
    """Parse the saddr field from a SOCKADDR audit record."""
    if not saddr_hex or len(saddr_hex) < 8:
        return None

    try:
        raw = bytes.fromhex(saddr_hex)
    except ValueError:
        return None

    if len(raw) < 2:
        return None

    family = struct.unpack_from('<H', raw, 0)[0]

    if family == 2 and len(raw) >= 8:  # AF_INET
        port = struct.unpack_from('>H', raw, 2)[0]
        ip = socket.inet_ntoa(raw[4:8])
        return ('ipv4', ip, port)

    if family == 10 and len(raw) >= 28:  # AF_INET6
        port = struct.unpack_from('>H', raw, 2)[0]
        ip = socket.inet_ntop(socket.AF_INET6, raw[8:24])
        return ('ipv6', ip, port)

    if family == 1:  # AF_UNIX
        path = raw[2:].split(b'\x00')[0].decode('utf-8', errors='replace')
        return ('unix', path, 0)

    return None


# Paths to skip in openat events (too noisy, no semantic value)
SKIP_PATHS = {'.', '..', '/', '', '/proc', '/sys', '/dev/null'}
SKIP_PREFIXES = ('/proc/', '/sys/', '/dev/pts/', '/dev/fd/')


def should_skip_path(path):
    """Return True if this file path should be excluded from the graph."""
    if path in SKIP_PATHS:
        return True
    for prefix in SKIP_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


class ProvenanceGraphBuilder:
    """Build a provenance graph from auditd events."""

    def __init__(self):
        self.entities = {}
        self.edges = []
        self.pid_to_entity = {}
        self.process_info = {}
        self._entity_counter = 0
        self._file_entity_cache = {}  # path -> eid (dedup files)

        # Stats
        self.event_key_counts = defaultdict(int)
        self.skipped_counts = defaultdict(int)

    def _new_entity_id(self, prefix):
        self._entity_counter += 1
        return f"{prefix}_{self._entity_counter}"

    def _get_or_create_proc(self, pid, exe=None, cmdline=None):
        if pid in self.pid_to_entity:
            return self.pid_to_entity[pid]

        eid = self._new_entity_id("proc")
        text = cmdline or exe or f"pid={pid}"
        self.entities[eid] = ("PROC", text)
        self.pid_to_entity[pid] = eid
        if exe or cmdline:
            self.process_info[pid] = {'exe': exe, 'cmdline': cmdline}
        return eid

    def _get_or_create_file(self, path):
        """Get or create a file entity, deduplicating by path."""
        if path in self._file_entity_cache:
            return self._file_entity_cache[path]
        eid = self._new_entity_id("file")
        self.entities[eid] = ("FILE", path)
        self._file_entity_cache[path] = eid
        return eid

    def _create_sock_entity(self, local_ip, local_port, remote_ip, remote_port):
        eid = self._new_entity_id("sock")
        text = f"{local_ip} {local_port} {remote_ip} {remote_port}"
        self.entities[eid] = ("SOCK", text)
        return eid

    def _decode_execve_args(self, execve_rec):
        """Build command line from EXECVE record."""
        argc = int(execve_rec.get('argc', '0'))
        args = []
        for i in range(argc):
            arg = execve_rec.get(f'a{i}', '')
            if arg:
                # Decode hex-encoded arguments
                if len(arg) > 2 and all(c in '0123456789abcdef' for c in arg.lower()):
                    try:
                        arg = bytes.fromhex(arg).decode('utf-8', errors='replace')
                    except:
                        pass
                args.append(arg)
        return ' '.join(args) if args else None

    def _parse_open_flags(self, flags_str):
        """Parse openat flags to determine read/write intent."""
        try:
            if flags_str.startswith('0x') or flags_str.startswith('0X'):
                flags = int(flags_str, 16)
            else:
                flags = int(flags_str, 16)
        except (ValueError, TypeError):
            return 'read'

        # O_WRONLY=0x1, O_RDWR=0x2, O_CREAT=0x40, O_TRUNC=0x200, O_APPEND=0x400
        O_WRONLY = 0x1
        O_RDWR = 0x2
        O_CREAT = 0x40
        O_TRUNC = 0x200
        O_APPEND = 0x400

        if flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND):
            return 'write'
        return 'read'

    def process_event(self, records):
        """Process a complete auditd event (group of records with same event_id)."""
        syscall_rec = None
        execve_rec = None
        path_recs = []
        sockaddr_rec = None
        cwd_rec = None

        for rec in records:
            rtype = rec.get('type', '')
            if rtype == 'SYSCALL':
                syscall_rec = rec
            elif rtype == 'EXECVE':
                execve_rec = rec
            elif rtype == 'PATH':
                path_recs.append(rec)
            elif rtype == 'SOCKADDR':
                sockaddr_rec = rec
            elif rtype == 'CWD':
                cwd_rec = rec

        if not syscall_rec:
            return

        # Use _audit_key if available (more reliable), fall back to 'key'
        key = syscall_rec.get('_audit_key', '') or syscall_rec.get('key', '')
        if key == '(null)' or not key:
            return

        pid = syscall_rec.get('pid', '')
        ppid = syscall_rec.get('ppid', '')
        exe = syscall_rec.get('exe', '')
        comm = syscall_rec.get('comm', '')
        success = syscall_rec.get('success', '')
        timestamp = syscall_rec.get('timestamp', 0)

        self.event_key_counts[key] += 1

        if success != 'yes':
            self.skipped_counts['failed_syscall'] += 1
            return

        # ── EVENT_EXECUTE (execve) ──
        if key == 'prov_exec' and execve_rec:
            cmdline = self._decode_execve_args(execve_rec) or exe
            parent_eid = self._get_or_create_proc(ppid)

            child_eid = self._new_entity_id("proc")
            self.entities[child_eid] = ("PROC", cmdline)
            self.pid_to_entity[pid] = child_eid
            self.process_info[pid] = {'exe': exe, 'cmdline': cmdline}

            # Binary file → process edge
            if path_recs:
                binary_path = path_recs[0].get('name', exe)
                if not should_skip_path(binary_path):
                    bin_eid = self._get_or_create_file(binary_path)
                    self.edges.append((bin_eid, "EVENT_EXECUTE", child_eid, timestamp))

            # Parent → child execute edge
            self.edges.append((parent_eid, "EVENT_EXECUTE", child_eid, timestamp))

        # ── EVENT_CLONE (fork/clone/vfork) ──
        elif key == 'prov_clone':
            parent_eid = self._get_or_create_proc(ppid, exe=exe)
            child_eid = self._get_or_create_proc(pid, exe=exe)
            self.edges.append((parent_eid, "EVENT_CLONE", child_eid, timestamp))

        # ── EVENT_CONNECT ──
        elif key == 'prov_connect' and sockaddr_rec:
            saddr = sockaddr_rec.get('saddr', '')
            parsed = parse_sockaddr(saddr)
            if parsed and parsed[0] in ('ipv4', 'ipv6'):
                _, remote_ip, remote_port = parsed
                if remote_port == 0:
                    return
                proc_eid = self._get_or_create_proc(pid, exe=exe)
                sock_eid = self._create_sock_entity("0.0.0.0", 0, remote_ip, remote_port)
                self.edges.append((proc_eid, "EVENT_CONNECT", sock_eid, timestamp))
            elif parsed and parsed[0] == 'unix':
                _, path, _ = parsed
                if path and not should_skip_path(path):
                    proc_eid = self._get_or_create_proc(pid, exe=exe)
                    file_eid = self._get_or_create_file(path)
                    self.edges.append((proc_eid, "EVENT_CONNECT", file_eid, timestamp))

        # ── EVENT_ACCEPT (incoming connection) ──
        elif key == 'prov_accept' and sockaddr_rec:
            saddr = sockaddr_rec.get('saddr', '')
            parsed = parse_sockaddr(saddr)
            if parsed and parsed[0] in ('ipv4', 'ipv6'):
                _, remote_ip, remote_port = parsed
                proc_eid = self._get_or_create_proc(pid, exe=exe)
                sock_eid = self._create_sock_entity(remote_ip, remote_port, "0.0.0.0", 0)
                self.edges.append((sock_eid, "EVENT_CONNECT", proc_eid, timestamp))

        # ── EVENT_SENDTO / EVENT_SENDMSG ──
        elif key in ('prov_sendto', 'prov_sendmsg') and sockaddr_rec:
            saddr = sockaddr_rec.get('saddr', '')
            parsed = parse_sockaddr(saddr)
            if parsed and parsed[0] in ('ipv4', 'ipv6'):
                _, remote_ip, remote_port = parsed
                if remote_port == 0:
                    return
                proc_eid = self._get_or_create_proc(pid, exe=exe)
                sock_eid = self._create_sock_entity("0.0.0.0", 0, remote_ip, remote_port)
                event = "EVENT_SENDTO" if key == 'prov_sendto' else "EVENT_SENDMSG"
                self.edges.append((proc_eid, event, sock_eid, timestamp))

        # ── EVENT_RECVFROM / EVENT_RECVMSG ──
        elif key in ('prov_recvfrom', 'prov_recvmsg') and sockaddr_rec:
            saddr = sockaddr_rec.get('saddr', '')
            parsed = parse_sockaddr(saddr)
            if parsed and parsed[0] in ('ipv4', 'ipv6'):
                _, remote_ip, remote_port = parsed
                if remote_port == 0:
                    return
                proc_eid = self._get_or_create_proc(pid, exe=exe)
                sock_eid = self._create_sock_entity(remote_ip, remote_port, "0.0.0.0", 0)
                event = "EVENT_RECVFROM" if key == 'prov_recvfrom' else "EVENT_RECVMSG"
                self.edges.append((sock_eid, event, proc_eid, timestamp))

        # ── EVENT_READ / EVENT_WRITE (openat syscall) ──
        elif key == 'prov_openat':
            proc_eid = self._get_or_create_proc(pid, exe=exe, cmdline=comm)

            # Get file path from PATH records
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if should_skip_path(filepath):
                    continue
                # Resolve relative paths using CWD
                if not filepath.startswith('/') and cwd_rec:
                    cwd = cwd_rec.get('cwd', '')
                    if cwd:
                        filepath = os.path.join(cwd, filepath)

                file_eid = self._get_or_create_file(filepath)

                # Determine read vs write from flags (a2 field for openat)
                flags_str = syscall_rec.get('a2', '0')
                direction = self._parse_open_flags(flags_str)

                if direction == 'write':
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))
                else:
                    self.edges.append((file_eid, "EVENT_READ", proc_eid, timestamp))

        # ── EVENT_WRITE (file creation: mkdir) ──
        elif key == 'prov_create':
            proc_eid = self._get_or_create_proc(pid, exe=exe)
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if filepath and not should_skip_path(filepath):
                    file_eid = self._get_or_create_file(filepath)
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))

        # ── EVENT_WRITE (file deletion) ──
        elif key == 'prov_delete':
            proc_eid = self._get_or_create_proc(pid, exe=exe)
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if filepath and not should_skip_path(filepath):
                    file_eid = self._get_or_create_file(filepath)
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))

        # ── EVENT_WRITE (rename) ──
        elif key == 'prov_rename':
            proc_eid = self._get_or_create_proc(pid, exe=exe)
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if filepath and not should_skip_path(filepath):
                    file_eid = self._get_or_create_file(filepath)
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))

        # ── EVENT_WRITE (chmod/chown — attribute change) ──
        elif key in ('prov_chmod', 'prov_chown'):
            proc_eid = self._get_or_create_proc(pid, exe=exe)
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if filepath and not should_skip_path(filepath):
                    file_eid = self._get_or_create_file(filepath)
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))

        # ── EVENT_OPEN (ptrace — process inspection) ──
        elif key == 'prov_ptrace':
            proc_eid = self._get_or_create_proc(pid, exe=exe)
            target_pid = syscall_rec.get('a1', '')
            if target_pid:
                try:
                    target_pid = str(int(target_pid, 16))
                except:
                    pass
                target_eid = self._get_or_create_proc(target_pid)
                self.edges.append((proc_eid, "EVENT_OPEN", target_eid, timestamp))

        # ── Legacy file watch events (prov_file_*) ──
        elif key.startswith('prov_file'):
            proc_eid = self._get_or_create_proc(pid, exe=exe, cmdline=comm)
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if not filepath or should_skip_path(filepath):
                    continue
                file_eid = self._get_or_create_file(filepath)
                perm = syscall_rec.get('perm', '')
                if 'w' in perm or 'a' in perm:
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))
                elif 'r' in perm:
                    self.edges.append((file_eid, "EVENT_READ", proc_eid, timestamp))
                elif 'x' in perm:
                    self.edges.append((file_eid, "EVENT_EXECUTE", proc_eid, timestamp))
                else:
                    self.edges.append((file_eid, "EVENT_READ", proc_eid, timestamp))

        # ── Legacy prov_open / prov_container_open ──
        elif key in ('prov_open', 'prov_container_open'):
            proc_eid = self._get_or_create_proc(pid, exe=exe, cmdline=comm)
            for path_rec in path_recs:
                filepath = path_rec.get('name', '')
                if should_skip_path(filepath):
                    continue
                file_eid = self._get_or_create_file(filepath)
                flags_str = syscall_rec.get('a2', syscall_rec.get('a1', '0'))
                direction = self._parse_open_flags(flags_str)
                if direction == 'write':
                    self.edges.append((proc_eid, "EVENT_WRITE", file_eid, timestamp))
                else:
                    self.edges.append((file_eid, "EVENT_READ", proc_eid, timestamp))

        else:
            self.skipped_counts[f'unhandled_key:{key}'] += 1

    def export(self, output_dir):
        """Export entities and edges to TSV files."""
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'entities.tsv'), 'w') as f:
            f.write("entity_id\tentity_type\tentity_text\n")
            for eid, (etype, etext) in sorted(self.entities.items()):
                f.write(f"{eid}\t{etype}\t{etext}\n")

        with open(os.path.join(output_dir, 'edges.tsv'), 'w') as f:
            f.write("src_id\tevent_type\tdst_id\ttimestamp\n")
            for src, evt, dst, ts in self.edges:
                f.write(f"{src}\t{evt}\t{dst}\t{ts}\n")

        # Stats
        type_counts = defaultdict(int)
        for _, (t, _) in self.entities.items():
            type_counts[t] += 1
        event_counts = defaultdict(int)
        for _, evt, _, _ in self.edges:
            event_counts[evt] += 1

        print(f"\nExported {len(self.entities)} entities, {len(self.edges)} edges to {output_dir}/")
        print(f"  Entities: {dict(type_counts)}")
        print(f"  Events:   {dict(sorted(event_counts.items(), key=lambda x: -x[1]))}")

        print(f"\n  Audit key distribution:")
        for key, count in sorted(self.event_key_counts.items(), key=lambda x: -x[1]):
            print(f"    {key:30s} {count:>10,}")

        if self.skipped_counts:
            print(f"\n  Skipped:")
            for reason, count in sorted(self.skipped_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"    {reason:30s} {count:>10,}")


def main():
    parser = argparse.ArgumentParser(description='Convert auditd logs to provenance graph (v2)')
    parser.add_argument('--input', required=True, help='Path to audit.log (or - for stdin)')
    parser.add_argument('--output', default='provenance_data', help='Output directory')
    args = parser.parse_args()

    builder = ProvenanceGraphBuilder()

    current_event_id = None
    current_records = []

    def flush_event():
        if current_records:
            builder.process_event(current_records)

    if args.input == '-':
        infile = sys.stdin
    else:
        infile = open(args.input, 'r')

    line_count = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue

        line_count += 1
        rec = parse_audit_line(line)
        eid = rec.get('event_id')

        if eid != current_event_id:
            flush_event()
            current_event_id = eid
            current_records = [rec]
        else:
            current_records.append(rec)

    flush_event()

    if args.input != '-':
        infile.close()

    print(f"Processed {line_count:,} audit lines")
    builder.export(args.output)


if __name__ == '__main__':
    main()