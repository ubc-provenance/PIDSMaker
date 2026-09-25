#!/bin/bash
# ============================================================================
# DOMAIN 16: DISK & FILESYSTEM
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "16 — DISK & FILESYSTEM"
S=$(generate_sample_files)

section "Filesystem info"
run "df" df
run "df -h" df -h
run "df -i" df -i
run "df -T" df -T
run "df -a" df -a | head -10
run "mount" mount | head -15
run "mount -l" mount -l | head -10
run "findmnt" findmnt 2>/dev/null | head -15
run "findmnt -t" findmnt -t ext4 2>/dev/null
run "blkid" blkid 2>/dev/null
run "lsblk" lsblk 2>/dev/null
run "cat /proc/mounts" cat /proc/mounts | head -10
run "cat /proc/filesystems" cat /proc/filesystems

section "Loopback filesystem"
run "dd create image" dd if=/dev/zero of="$S/disk.img" bs=1M count=10 2>/dev/null
run "mkfs.ext4" mkfs.ext4 -F "$S/disk.img" 2>/dev/null
mkdir -p "$S/mnt"
run "mount loopback" mount -o loop "$S/disk.img" "$S/mnt" 2>/dev/null
run "write to mount" echo "mounted test" > "$S/mnt/test.txt" 2>/dev/null
run "read from mount" cat "$S/mnt/test.txt" 2>/dev/null
run "ls mount" ls -la "$S/mnt/" 2>/dev/null
run "df mount" df -h "$S/mnt" 2>/dev/null
run "umount" umount "$S/mnt" 2>/dev/null
run "mkfs.ext2" mkfs.ext2 -F "$S/disk.img" 2>/dev/null
run "file fs image" file "$S/disk.img"
rm -f "$S/disk.img"

section "rsync variants"
mkdir -p "$S/rsrc" "$S/rdst"
cp "$S/tiny.txt" "$S/small.txt" "$S/multiline.txt" "$S/rsrc/"
run "rsync -a" rsync -a "$S/rsrc/" "$S/rdst/"
run "rsync -av" rsync -av "$S/rsrc/" "$S/rdst/"
run "rsync -avz" rsync -avz "$S/rsrc/" "$S/rdst/"
run "rsync --delete" rsync -av --delete "$S/rsrc/" "$S/rdst/"
run "rsync -n dry" rsync -avn "$S/rsrc/" "$S/rdst/"
run "rsync --exclude" rsync -av --exclude="*.bin" "$S/rsrc/" "$S/rdst/"
run "rsync --checksum" rsync -avc "$S/rsrc/" "$S/rdst/"
run "rsync --backup" rsync -av --backup --suffix=.old "$S/rsrc/" "$S/rdst/"
run "rsync --progress" rsync -av --progress "$S/rsrc/" "$S/rdst/"
run "rsync --compress" rsync -avz --compress-level=9 "$S/rsrc/" "$S/rdst/"
rm -rf "$S/rsrc" "$S/rdst"

section "File attributes"
run "lsattr" lsattr "$S/tiny.txt" 2>/dev/null
run "chattr +i" chattr +i "$S/tiny.txt" 2>/dev/null
run "chattr -i" chattr -i "$S/tiny.txt" 2>/dev/null
run "chattr +a" chattr +a "$S/tiny.txt" 2>/dev/null
run "chattr -a" chattr -a "$S/tiny.txt" 2>/dev/null

domain_end
