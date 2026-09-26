#!/bin/bash
# ============================================================================
# DOMAIN 01: CORE SYSTEM UTILITIES — Exhaustive flag coverage
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "01 — CORE SYSTEM UTILITIES"

S=$(generate_sample_files)

# ============================================================================
section "ls — list directory contents"
# ============================================================================
run "ls" ls
run "ls /" ls /
run "ls /etc" ls /etc
run "ls /usr/bin" ls /usr/bin
run "ls /var/log" ls /var/log
run "ls /tmp" ls /tmp
run "ls /dev" ls /dev
run "ls /proc" ls /proc
run "ls -l" ls -l /etc
run "ls -la" ls -la /etc
run "ls -lh" ls -lh /usr/bin
run "ls -lah" ls -lah /var
run "ls -R" ls -R "$S/tree" 
run "ls -lR" ls -lR "$S/tree"
run "ls -1" ls -1 /usr/bin | head -30
run "ls -S" ls -S /usr/bin | head -10
run "ls -lS" ls -lS /usr/bin | head -10
run "ls -lSr" ls -lSr /usr/bin | head -10
run "ls -t" ls -t /var/log | head -10
run "ls -lt" ls -lt /var/log | head -10
run "ls -ltr" ls -ltr /var/log | head -10
run "ls -i" ls -i /etc | head -10
run "ls -li" ls -li /etc | head -10
run "ls -lai" ls -lai /etc | head -10
run "ls -d" ls -d /etc/*/
run "ls -ld" ls -ld /etc /var /tmp
run "ls -F" ls -F /usr/bin | head -10
run "ls -p" ls -p /usr/bin | head -10
run "ls --color" ls --color=always /etc | head -10
run "ls --sort=size" ls --sort=size /usr/bin | head -10
run "ls --sort=time" ls --sort=time /var/log | head -10
run "ls --sort=extension" ls --sort=extension /etc | head -10
run "ls -n" ls -n /etc | head -10
run "ls -g" ls -g /etc | head -10
run "ls -o" ls -o /etc | head -10
run "ls -A" ls -A /root 2>/dev/null || ls -A /tmp
run "ls -B" ls -B /tmp
run "ls -C" ls -C /usr/bin | head -5
run "ls -m" ls -m /usr/bin | head -3
run "ls -x" ls -x /usr/bin | head -3
run "ls -Q" ls -Q /etc | head -10
run "ls --block-size=M" ls -l --block-size=M /usr/bin | head -5
run "ls --full-time" ls --full-time /etc | head -5
run "ls -lhSra /usr/lib" ls -lhSra /usr/lib | head -20
run "ls hidden" ls -la "$S/"
run "ls symlink" ln -sf "$S/tiny.txt" "$S/link.txt" && ls -la "$S/link.txt" && rm -f "$S/link.txt"
run "ls glob txt" ls "$S"/*.txt
run "ls glob bin" ls "$S"/*.bin

# ============================================================================
section "cat — concatenate and display"
# ============================================================================
run "cat file" cat "$S/tiny.txt"
run "cat -n" cat -n "$S/multiline.txt"
run "cat -b" cat -b "$S/multiline.txt"
run "cat -s" cat -s "$S/multiline.txt"
run "cat -A" cat -A "$S/tiny.txt"
run "cat -E" cat -E "$S/multiline.txt" | head -5
run "cat -T" cat -T "$S/multiline.txt" | head -5
run "cat -v" cat -v "$S/random_1k.bin" | head -5
run "cat -e" cat -e "$S/multiline.txt" | head -5
run "cat -t" cat -t "$S/multiline.txt" | head -5
run "cat /etc/passwd" cat /etc/passwd
run "cat /etc/group" cat /etc/group
run "cat /etc/hostname" cat /etc/hostname
run "cat /etc/hosts" cat /etc/hosts
run "cat /etc/os-release" cat /etc/os-release
run "cat /etc/fstab" cat /etc/fstab 2>/dev/null
run "cat /etc/resolv.conf" cat /etc/resolv.conf 2>/dev/null
run "cat /etc/shells" cat /etc/shells 2>/dev/null
run "cat /etc/profile" cat /etc/profile 2>/dev/null
run "cat /etc/environment" cat /etc/environment 2>/dev/null
run "cat /etc/timezone" cat /etc/timezone 2>/dev/null
run "cat /etc/shadow" cat /etc/shadow 2>/dev/null
run "cat /etc/sudoers" cat /etc/sudoers 2>/dev/null
run "cat /etc/ssh/sshd_config" cat /etc/ssh/sshd_config 2>/dev/null
run "cat /etc/nginx/nginx.conf" cat /etc/nginx/nginx.conf 2>/dev/null
run "cat /etc/apt/sources.list" cat /etc/apt/sources.list 2>/dev/null
run "cat /proc/version" cat /proc/version
run "cat /proc/cpuinfo" cat /proc/cpuinfo
run "cat /proc/meminfo" cat /proc/meminfo
run "cat /proc/loadavg" cat /proc/loadavg
run "cat /proc/uptime" cat /proc/uptime
run "cat /proc/mounts" cat /proc/mounts
run "cat /proc/self/status" cat /proc/self/status
run "cat /proc/self/cmdline" cat /proc/self/cmdline | tr '\0' ' '; echo
run "cat /proc/self/maps" cat /proc/self/maps | head -10
run "cat /proc/net/tcp" cat /proc/net/tcp | head -5
run "cat /proc/net/udp" cat /proc/net/udp | head -5
run "cat /proc/filesystems" cat /proc/filesystems
run "cat /proc/partitions" cat /proc/partitions 2>/dev/null
run "cat /proc/diskstats" cat /proc/diskstats 2>/dev/null | head -5
run "cat multiple" cat /etc/hostname /etc/hosts "$S/tiny.txt"
run "cat concat" cat "$S/tiny.txt" "$S/small.txt" > "$S/concatenated.txt"
run "cat heredoc" cat << 'EOF'
heredoc line 1
heredoc line 2
EOF
run "cat /dev/null" cat /dev/null

# ============================================================================
section "cp — copy"
# ============================================================================
run "cp basic" cp "$S/tiny.txt" "$S/tiny_cp.txt"
run "cp -v" cp -v "$S/tiny.txt" "$S/tiny_cpv.txt"
run "cp -i force" yes | cp -i "$S/tiny.txt" "$S/tiny_cpv.txt" 2>/dev/null
run "cp -f" cp -f "$S/tiny.txt" "$S/tiny_cpv.txt"
run "cp -n" cp -n "$S/tiny.txt" "$S/tiny_new.txt" 2>/dev/null
run "cp -u" cp -u "$S/tiny.txt" "$S/tiny_cpv.txt"
run "cp -p" cp -p "$S/tiny.txt" "$S/tiny_preserved.txt"
run "cp -a" cp -a "$S/tiny.txt" "$S/tiny_archive.txt"
run "cp --preserve=all" cp --preserve=all "$S/tiny.txt" "$S/tiny_pall.txt"
run "cp --preserve=mode" cp --preserve=mode "$S/tiny.txt" "$S/tiny_pmode.txt"
run "cp --preserve=timestamps" cp --preserve=timestamps "$S/tiny.txt" "$S/tiny_pts.txt"
run "cp -r dir" cp -r "$S/tree" "$S/tree_copy"
run "cp -a dir" cp -a "$S/tree" "$S/tree_archive"
run "cp -l hardlink" cp -l "$S/tiny.txt" "$S/tiny_hard.txt" 2>/dev/null
run "cp -s symlink" cp -s "$S/tiny.txt" "$S/tiny_sym.txt" 2>/dev/null
run "cp --reflink" cp --reflink=auto "$S/tiny.txt" "$S/tiny_ref.txt" 2>/dev/null
run "cp --backup" cp --backup=numbered "$S/tiny.txt" "$S/tiny_cpv.txt"
run "cp --suffix" cp --suffix=.bak "$S/tiny.txt" "$S/tiny_cpv.txt"
run "cp multiple" cp "$S/tiny.txt" "$S/small.txt" "$S/tree/"
rm -rf "$S/tree_copy" "$S/tree_archive" "$S/tiny_cp.txt" "$S/tiny_cpv.txt" "$S/tiny_new.txt" \
       "$S/tiny_preserved.txt" "$S/tiny_archive.txt" "$S/tiny_pall.txt" "$S/tiny_pmode.txt" \
       "$S/tiny_pts.txt" "$S/tiny_hard.txt" "$S/tiny_sym.txt" "$S/tiny_ref.txt"

# ============================================================================
section "mv — move/rename"
# ============================================================================
cp "$S/tiny.txt" "$S/mv_test.txt"
run "mv rename" mv "$S/mv_test.txt" "$S/mv_renamed.txt"
run "mv back" mv "$S/mv_renamed.txt" "$S/mv_test.txt"
run "mv -v" mv -v "$S/mv_test.txt" "$S/mv_v.txt"
run "mv -f" mv -f "$S/mv_v.txt" "$S/mv_test.txt"
run "mv -n" cp "$S/tiny.txt" "$S/mv_nolobber.txt" && mv -n "$S/mv_test.txt" "$S/mv_nolobber.txt" 2>/dev/null
run "mv --backup" cp "$S/tiny.txt" "$S/mv_bak.txt" && mv --backup=numbered "$S/mv_nolobber.txt" "$S/mv_bak.txt"
run "mv to dir" cp "$S/tiny.txt" "$S/mv_todir.txt" && mv "$S/mv_todir.txt" "$S/tree/"
rm -f "$S/mv_test.txt" "$S/mv_bak.txt" "$S/mv_bak.txt.~1~" "$S/tree/mv_todir.txt"

# ============================================================================
section "rm — remove"
# ============================================================================
touch "$S/rm1.txt" "$S/rm2.txt" "$S/rm3.txt"
run "rm basic" rm "$S/rm1.txt"
run "rm -f" rm -f "$S/rm2.txt"
run "rm -v" rm -v "$S/rm3.txt"
mkdir -p "$S/rmdir/sub" && touch "$S/rmdir/sub/file.txt"
run "rm -r" rm -r "$S/rmdir"
mkdir -p "$S/rmdir2/sub" && touch "$S/rmdir2/sub/file.txt"
run "rm -rf" rm -rf "$S/rmdir2"
touch "$S/rm_glob1.tmp" "$S/rm_glob2.tmp" "$S/rm_glob3.tmp"
run "rm glob" rm -f "$S"/*.tmp

# ============================================================================
section "chmod — change permissions"
# ============================================================================
cp "$S/tiny.txt" "$S/chmod_test.txt"
for mode in 000 111 222 333 444 555 644 664 700 711 755 775 777; do
    run "chmod $mode" chmod $mode "$S/chmod_test.txt"
done
run "chmod u+x" chmod u+x "$S/chmod_test.txt"
run "chmod u-x" chmod u-x "$S/chmod_test.txt"
run "chmod g+w" chmod g+w "$S/chmod_test.txt"
run "chmod g-w" chmod g-w "$S/chmod_test.txt"
run "chmod o+r" chmod o+r "$S/chmod_test.txt"
run "chmod o-r" chmod o-r "$S/chmod_test.txt"
run "chmod a+x" chmod a+x "$S/chmod_test.txt"
run "chmod a-x" chmod a-x "$S/chmod_test.txt"
run "chmod u+rwx" chmod u+rwx "$S/chmod_test.txt"
run "chmod go-rwx" chmod go-rwx "$S/chmod_test.txt"
run "chmod u=rw,g=r,o=" chmod u=rw,g=r,o= "$S/chmod_test.txt"
run "chmod +t" chmod +t "$S/chmod_test.txt"
run "chmod -t" chmod -t "$S/chmod_test.txt"
run "chmod +s" chmod u+s "$S/chmod_test.txt" 2>/dev/null
run "chmod -R" chmod -R 755 "$S/tree"
run "chmod --reference" chmod --reference=/etc/passwd "$S/chmod_test.txt"
rm -f "$S/chmod_test.txt"

# ============================================================================
section "chown/chgrp — change ownership"
# ============================================================================
cp "$S/tiny.txt" "$S/chown_test.txt"
run "chown root" chown root "$S/chown_test.txt" 2>/dev/null
run "chown root:root" chown root:root "$S/chown_test.txt" 2>/dev/null
run "chown :root" chown :root "$S/chown_test.txt" 2>/dev/null
run "chown -R" chown -R root:root "$S/tree" 2>/dev/null
run "chown --reference" chown --reference=/etc/passwd "$S/chown_test.txt" 2>/dev/null
run "chgrp root" chgrp root "$S/chown_test.txt" 2>/dev/null
run "chgrp -R" chgrp -R root "$S/tree" 2>/dev/null
rm -f "$S/chown_test.txt"

# ============================================================================
section "find — search for files"
# ============================================================================
run "find all" find "$S" 2>/dev/null | head -30
run "find -name txt" find "$S" -name "*.txt"
run "find -name bin" find "$S" -name "*.bin"
run "find -iname" find "$S" -iname "*.TXT"
run "find -type f" find "$S" -type f | head -20
run "find -type d" find "$S" -type d
run "find -type l" find "$S" -type l
run "find -maxdepth 1" find "$S" -maxdepth 1 -type f
run "find -maxdepth 2" find "$S" -maxdepth 2 -type f
run "find -mindepth 2" find "$S" -mindepth 2 -type f
run "find -size +1k" find "$S" -size +1k -type f
run "find -size -1k" find "$S" -size -1k -type f
run "find -size +10k" find "$S" -size +10k -type f
run "find -empty" find "$S" -empty
run "find -newer" find "$S" -newer "$S/tiny.txt"
run "find -mmin -60" find "$S" -mmin -60 -type f | head -10
run "find -mtime 0" find "$S" -mtime 0 -type f | head -10
run "find -perm 644" find "$S" -perm 644 -type f | head -5
run "find -perm -u+x" find "$S" -perm -u+x -type f
run "find -user root" find "$S" -user root -type f 2>/dev/null | head -5
run "find -exec ls" find "$S" -name "*.txt" -exec ls -l {} \;
run "find -exec cat" find "$S" -name "tiny.txt" -exec cat {} \;
run "find -exec rm" touch "$S/todelete1.tmp" "$S/todelete2.tmp" && find "$S" -name "*.tmp" -exec rm {} \;
run "find -print0" find "$S" -name "*.txt" -print0 | xargs -0 ls -la
run "find -printf" find "$S" -type f -printf "%s %p\n" | head -10
run "find -delete" touch "$S/del1.tmp" "$S/del2.tmp" && find "$S" -name "*.tmp" -delete
run "find -or" find "$S" -name "*.txt" -o -name "*.bin" | head -10
run "find -and" find "$S" -name "*.txt" -size +0c
run "find -not" find "$S" -not -name "*.txt" -type f | head -10
run "find -regex" find "$S" -regex ".*\.\(txt\|bin\)" | head -10
run "find / SUID" find / -perm -4000 -type f 2>/dev/null | head -20
run "find / SGID" find / -perm -2000 -type f 2>/dev/null | head -10
run "find / writable" find /etc -writable -type f 2>/dev/null | head -10
run "find / world-writable" find /tmp -perm -0002 -type f 2>/dev/null | head -10
run "find /etc -name conf" find /etc -name "*.conf" -maxdepth 2 2>/dev/null | head -20
run "find /usr -type f" find /usr/bin -type f -maxdepth 1 2>/dev/null | wc -l

# ============================================================================
section "grep — search text patterns"
# ============================================================================
run "grep basic" grep root /etc/passwd
run "grep -i" grep -i ROOT /etc/passwd
run "grep -v" grep -v "^#" /etc/hosts
run "grep -c" grep -c : /etc/passwd
run "grep -n" grep -n bash /etc/passwd
run "grep -l" grep -rl root /etc/passwd /etc/group
run "grep -L" grep -rL root "$S"/*.txt 2>/dev/null
run "grep -w" grep -w root /etc/passwd
run "grep -x" grep -x "root:x:0:0:root:/root:/bin/bash" /etc/passwd 2>/dev/null || true
run "grep -m 1" grep -m 1 root /etc/passwd
run "grep -o" grep -o "[0-9]\+" /etc/passwd | head -10
run "grep -A 2" grep -A 2 root /etc/passwd
run "grep -B 2" grep -B 2 root /etc/passwd
run "grep -C 2" grep -C 2 root /etc/passwd
run "grep -E extended" grep -E "^(root|daemon|nobody)" /etc/passwd
run "grep -E or" grep -E "root|admin|user" /etc/passwd
run "grep -E quantifier" grep -E "[0-9]{2,}" /etc/passwd | head -5
run "grep -P perl" grep -P "\d+\.\d+" /etc/hosts 2>/dev/null || true
run "grep -P lookahead" grep -P "root(?=:)" /etc/passwd 2>/dev/null || true
run "grep -r recursive" grep -r "root" /etc/passwd /etc/group /etc/hosts 2>/dev/null
run "grep -r /etc" grep -rl "127.0.0" /etc/ 2>/dev/null | head -10
run "grep -Z" grep -rlZ "root" "$S" 2>/dev/null | xargs -0 ls -la 2>/dev/null
run "grep --include" grep -r --include="*.txt" "line" "$S" | head -5
run "grep --exclude" grep -r --exclude="*.bin" "." "$S" | head -5 2>/dev/null
run "grep --exclude-dir" grep -r --exclude-dir=tree "." "$S" | head -5 2>/dev/null
run "grep -f pattern file" echo -e "root\ndaemon" > "$S/patterns.txt" && grep -f "$S/patterns.txt" /etc/passwd
run "grep binary" grep -a "ELF" /usr/bin/ls | head -1
run "grep pipe" cat /etc/passwd | grep root
run "grep chain" cat /etc/passwd | grep -v "^#" | grep -c ":"
run "grep color" grep --color=always root /etc/passwd | head -3

# ============================================================================
section "sed — stream editor"
# ============================================================================
run "sed s basic" sed 's/root/ROOT/' /etc/passwd | head -3
run "sed s global" sed 's/root/ROOT/g' /etc/passwd | head -3
run "sed s case-i" sed 's/root/ROOT/gi' /etc/passwd 2>/dev/null | head -3
run "sed s delimiter" sed 's|/bin/bash|/bin/zsh|' /etc/passwd | head -3
run "sed delete line" sed '/^#/d' /etc/hosts
run "sed delete range" sed '3,5d' /etc/passwd | head -5
run "sed print -n" sed -n '1,5p' /etc/passwd
run "sed insert" echo "test" | sed '1i\HEADER LINE'
run "sed append" echo "test" | sed '1a\FOOTER LINE'
run "sed change" echo -e "old line\nkeep" | sed '1c\new line'
run "sed replace line" sed '1s/.*/FIRST LINE/' /etc/passwd | head -3
run "sed multiple -e" sed -e 's/a/A/g' -e 's/e/E/g' "$S/tiny.txt"
run "sed -f script" echo 's/root/ROOT/g' > "$S/sed_script.sed" && sed -f "$S/sed_script.sed" /etc/passwd | head -3
run "sed -i inplace" cp /etc/hosts "$S/hosts_edit" && sed -i 's/localhost/LOCALHOST/g' "$S/hosts_edit"
run "sed -i.bak" cp /etc/hosts "$S/hosts_bak" && sed -i.bak 's/localhost/MODIFIED/' "$S/hosts_bak"
run "sed address" sed -n '/root/p' /etc/passwd
run "sed regex addr" sed -n '/^root/,/^daemon/p' /etc/passwd
run "sed transliterate" echo "hello" | sed 'y/abcde/ABCDE/'
run "sed hold space" echo -e "line1\nline2\nline3" | sed -n 'H;${x;s/\n/ | /g;p}'
run "sed remove blank" echo -e "a\n\nb\n\nc" | sed '/^$/d'
run "sed number lines" sed = /etc/passwd | sed 'N;s/\n/\t/' | head -5
run "sed last line" sed -n '$p' /etc/passwd

# ============================================================================
section "awk — pattern processing"
# ============================================================================
run "awk print" awk '{print}' "$S/tiny.txt"
run "awk print $1" awk '{print $1}' /etc/passwd | head -5
run "awk -F:" awk -F: '{print $1, $3}' /etc/passwd | head -5
run "awk -F: printf" awk -F: '{printf "%-15s UID=%s\n", $1, $3}' /etc/passwd | head -5
run "awk NR" awk 'NR<=5{print NR, $0}' /etc/passwd
run "awk NF" awk '{print NF, $0}' "$S/tiny.txt"
run "awk pattern" awk '/root/{print}' /etc/passwd
run "awk not pattern" awk '!/nologin/{print $0}' /etc/passwd | head -5
run "awk range" awk 'NR>=2 && NR<=5{print}' /etc/passwd
run "awk sum" awk '{sum+=$1} END{print "sum="sum}' "$S/small.txt"
run "awk count" awk 'END{print NR" lines"}' /etc/passwd
run "awk max" awk -F: '{if($3>max)max=$3} END{print "max UID="max}' /etc/passwd
run "awk array" awk -F: '{shells[$7]++} END{for(s in shells) print s, shells[s]}' /etc/passwd
run "awk substr" awk '{print substr($0,1,20)}' /etc/passwd | head -5
run "awk length" awk '{print length($0), $0}' "$S/tiny.txt"
run "awk tolower" awk '{print tolower($0)}' "$S/tiny.txt"
run "awk toupper" awk '{print toupper($0)}' "$S/tiny.txt"
run "awk gsub" awk '{gsub(/root/,"ROOT"); print}' /etc/passwd | head -3
run "awk split" awk -F: '{split($0,a,":"); print a[1]}' /etc/passwd | head -5
run "awk OFS" awk -F: 'BEGIN{OFS=","}{print $1,$3,$7}' /etc/passwd | head -5
run "awk ORS" awk 'BEGIN{ORS=" | "}{print $0}' "$S/tiny.txt"
run "awk BEGIN/END" awk 'BEGIN{print "=START="}{print} END{print "=END="}' "$S/tiny.txt"
run "awk getline" awk '{cmd="hostname"; cmd|getline h; close(cmd); print $0, h}' "$S/tiny.txt"
run "awk system" awk 'BEGIN{system("echo awk subprocess")}'
run "awk multiple files" awk '{print FILENAME, $0}' "$S/tiny.txt" "$S/data.csv" | head -10
run "awk RS" echo "a;b;c;d" | awk 'BEGIN{RS=";"}{print NR, $0}'
run "awk conditional" awk -F: '{print ($3==0)?"ROOT":"user", $1}' /etc/passwd | head -5
run "awk ternary" awk -F: '{print $1, ($7~/bash/?"has bash":"no bash")}' /etc/passwd | head -5

# ============================================================================
section "sort — sort lines"
# ============================================================================
run "sort basic" sort /etc/passwd | head -5
run "sort -r" sort -r /etc/passwd | head -5
run "sort -n" sort -n "$S/small.txt" | tail -5
run "sort -rn" sort -rn "$S/small.txt" | head -5
run "sort -u" sort -u "$S/small.txt" | wc -l
run "sort -f" echo -e "Banana\napple\nCherry" | sort -f
run "sort -t -k" sort -t: -k3 -n /etc/passwd | head -5
run "sort -t -k2" sort -t: -k1,1 /etc/passwd | head -5
run "sort -k2 -k1" sort -t: -k3,3n -k1,1 /etc/passwd | head -5
run "sort -h" echo -e "10K\n1M\n500K\n2G" | sort -h
run "sort -V" echo -e "1.10\n1.2\n1.1\n1.9" | sort -V
run "sort -R" sort -R /etc/passwd | head -3
run "sort -s stable" sort -s -t: -k7 /etc/passwd | head -5
run "sort -o output" sort "$S/small.txt" -o "$S/sorted.txt"
run "sort -z" find "$S" -print0 | sort -z | tr '\0' '\n' | head -5
run "sort -c" sort -c "$S/sorted.txt" 2>/dev/null
run "sort -m merge" sort "$S/small.txt" > "$S/s1.txt" && sort -r "$S/small.txt" > "$S/s2.txt" && sort -m "$S/s1.txt" "$S/s2.txt" | head -5

# ============================================================================
section "uniq, cut, tr, wc, head, tail, tee, xargs"
# ============================================================================
run "uniq" echo -e "a\na\nb\nb\nb\nc" | sort | uniq
run "uniq -c" echo -e "a\na\nb\nb\nb\nc" | sort | uniq -c
run "uniq -d" echo -e "a\na\nb\nb\nb\nc" | sort | uniq -d
run "uniq -u" echo -e "a\na\nb\nb\nb\nc" | sort | uniq -u
run "uniq -i" echo -e "Hello\nhello\nHELLO" | sort -f | uniq -ci

run "cut -d -f" cut -d: -f1 /etc/passwd | head -5
run "cut -d -f1,3" cut -d: -f1,3 /etc/passwd | head -5
run "cut -d -f1-3" cut -d: -f1-3 /etc/passwd | head -5
run "cut -c" cut -c1-10 /etc/passwd | head -5
run "cut -c1" cut -c1 /etc/passwd | head -5
run "cut -b" cut -b1-5 /etc/passwd | head -5
run "cut --complement" cut -d: -f2 --complement /etc/passwd | head -5
run "cut --output-delimiter" cut -d: -f1,7 --output-delimiter=" -> " /etc/passwd | head -5

run "tr lower" echo "HELLO WORLD" | tr '[:upper:]' '[:lower:]'
run "tr upper" echo "hello world" | tr '[:lower:]' '[:upper:]'
run "tr delete digits" echo "hello 123 world 456" | tr -d '[:digit:]'
run "tr delete alpha" echo "hello 123 world 456" | tr -d '[:alpha:]'
run "tr delete spaces" echo "hello   world" | tr -d ' '
run "tr squeeze" echo "hello     world" | tr -s ' '
run "tr replace" echo "hello world" | tr ' ' '_'
run "tr set1 set2" echo "hello" | tr 'aeiou' 'AEIOU'
run "tr -c complement" echo "hello 123" | tr -cd '[:digit:]\n'
run "tr rot13" echo "hello world" | tr 'a-zA-Z' 'n-za-mN-ZA-M'
run "tr delete newlines" echo -e "a\nb\nc" | tr -d '\n'; echo

run "wc" wc /etc/passwd
run "wc -l" wc -l /etc/passwd
run "wc -w" wc -w "$S/tiny.txt"
run "wc -c" wc -c "$S/random_1k.bin"
run "wc -m" wc -m "$S/tiny.txt"
run "wc -L" wc -L /etc/passwd
run "wc multiple" wc "$S/tiny.txt" "$S/small.txt" "$S/medium.txt"

run "head" head /etc/passwd
run "head -n 1" head -n 1 /etc/passwd
run "head -n 5" head -n 5 /etc/passwd
run "head -n 20" head -n 20 /etc/passwd
run "head -c 10" head -c 10 "$S/random_1k.bin" | xxd
run "head -c 100" head -c 100 "$S/tiny.txt"
run "head -q" head -q -n 1 /etc/passwd /etc/hosts

run "tail" tail /etc/passwd
run "tail -n 1" tail -n 1 /etc/passwd
run "tail -n 5" tail -n 5 /etc/passwd
run "tail -n +3" tail -n +3 /etc/passwd | head -5
run "tail -c 20" tail -c 20 "$S/tiny.txt"
run "tail -q" tail -q -n 1 /etc/passwd /etc/hosts
run_t 2 "tail -f" tail -f /dev/null

run "tee" echo "tee test" | tee "$S/tee1.txt"
run "tee -a" echo "appended" | tee -a "$S/tee1.txt"
run "tee multiple" echo "multi" | tee "$S/tee2.txt" "$S/tee3.txt"
run "tee pipe" echo "hello" | tee "$S/tee4.txt" | wc -c

run "xargs basic" echo "a b c" | xargs echo
run "xargs -n 1" echo "a b c d e" | xargs -n 1 echo
run "xargs -n 2" echo "a b c d e f" | xargs -n 2 echo
run "xargs -I{}" echo -e "f1\nf2\nf3" | xargs -I{} touch "$S/{}.xarg" && rm -f "$S"/*.xarg
run "xargs -d" echo "a:b:c" | xargs -d: echo
run "xargs -0" find "$S" -name "*.txt" -print0 | xargs -0 wc -l 2>/dev/null | tail -5
run "xargs -P parallel" echo -e "1\n2\n3\n4" | xargs -P 4 -I{} bash -c 'echo "parallel {}"'
run "xargs -L" echo -e "a\nb\nc\nd" | xargs -L 2 echo
run "xargs grep" echo -e "/etc/passwd\n/etc/hosts" | xargs grep "root" 2>/dev/null | head -3
run "xargs rm" touch "$S/xrm1.tmp" "$S/xrm2.tmp" && echo "$S/xrm1.tmp $S/xrm2.tmp" | xargs rm -f

# ============================================================================
section "touch, mkdir, rmdir, ln, stat, file"
# ============================================================================
run "touch new" touch "$S/new_touch.txt"
run "touch -t" touch -t 202301011200 "$S/new_touch.txt"
run "touch -d" touch -d "2023-06-15 10:30" "$S/new_touch.txt"
run "touch -r" touch -r /etc/passwd "$S/new_touch.txt"
run "touch -a" touch -a "$S/new_touch.txt"
run "touch -m" touch -m "$S/new_touch.txt"
run "touch multiple" touch "$S/t1" "$S/t2" "$S/t3" && rm -f "$S/t1" "$S/t2" "$S/t3"

run "mkdir basic" mkdir "$S/newdir"
run "mkdir -p" mkdir -p "$S/deep/a/b/c/d/e"
run "mkdir -m" mkdir -m 700 "$S/securedir"
run "mkdir -v" mkdir -v "$S/verbosedir"
run "rmdir" rmdir "$S/newdir" "$S/verbosedir" "$S/securedir"
run "rmdir -p" rmdir -p "$S/deep/a/b/c/d/e" 2>/dev/null

run "ln hard" ln "$S/tiny.txt" "$S/hardlink.txt" 2>/dev/null
run "ln -s" ln -s "$S/tiny.txt" "$S/symlink.txt"
run "ln -sf" ln -sf "$S/small.txt" "$S/symlink.txt"
run "ln -s relative" ln -sr "$S/tiny.txt" "$S/relsym.txt"
run "readlink" readlink "$S/symlink.txt"
run "readlink -f" readlink -f "$S/symlink.txt"
run "readlink -e" readlink -e "$S/symlink.txt"
rm -f "$S/hardlink.txt" "$S/symlink.txt" "$S/relsym.txt"

run "stat file" stat "$S/tiny.txt"
run "stat -c format" stat -c "%n %s %U %G %a" "$S/tiny.txt"
run "stat -c inode" stat -c "%i" "$S/tiny.txt"
run "stat -c times" stat -c "access=%x modify=%y change=%z" "$S/tiny.txt"
run "stat -f filesystem" stat -f /
run "stat /etc/passwd" stat /etc/passwd

run "file text" file "$S/tiny.txt"
run "file binary" file "$S/random_1k.bin"
run "file executable" file /usr/bin/ls
run "file symlink" ln -sf "$S/tiny.txt" "$S/flink.txt" && file "$S/flink.txt" && rm -f "$S/flink.txt"
run "file -i mime" file -i "$S/tiny.txt"
run "file -b brief" file -b "$S/tiny.txt"
run "file -L deref" file -L /usr/bin/ls 2>/dev/null
run "file json" file "$S/data.json"
run "file xml" file "$S/data.xml"
run "file csv" file "$S/data.csv"

# ============================================================================
section "dd — data duplicator"
# ============================================================================
run "dd zeros" dd if=/dev/zero of="$S/dd_zeros.bin" bs=1k count=100 2>/dev/null
run "dd urandom" dd if=/dev/urandom of="$S/dd_rand.bin" bs=1k count=50 2>/dev/null
run "dd copy" dd if="$S/tiny.txt" of="$S/dd_copy.txt" 2>/dev/null
run "dd bs=1" dd if="$S/tiny.txt" of="$S/dd_bs1.txt" bs=1 2>/dev/null
run "dd bs=512" dd if=/dev/zero of="$S/dd_512.bin" bs=512 count=10 2>/dev/null
run "dd bs=4k" dd if=/dev/zero of="$S/dd_4k.bin" bs=4k count=25 2>/dev/null
run "dd bs=1M" dd if=/dev/zero of="$S/dd_1m.bin" bs=1M count=1 2>/dev/null
run "dd skip" dd if="$S/medium.txt" of="$S/dd_skip.txt" bs=100 skip=5 count=3 2>/dev/null
run "dd seek" dd if=/dev/zero of="$S/dd_seek.bin" bs=1k seek=5 count=1 2>/dev/null
run "dd conv=ucase" dd if="$S/tiny.txt" of="$S/dd_upper.txt" conv=ucase 2>/dev/null
run "dd conv=lcase" dd if="$S/dd_upper.txt" of="$S/dd_lower.txt" conv=lcase 2>/dev/null
run "dd conv=notrunc" dd if=/dev/zero of="$S/dd_zeros.bin" bs=1 count=10 conv=notrunc 2>/dev/null
run "dd status=progress" dd if=/dev/zero of="$S/dd_prog.bin" bs=1M count=5 status=progress 2>/dev/null
rm -f "$S"/dd_*.bin "$S"/dd_*.txt

# ============================================================================
section "diff/patch"
# ============================================================================
echo -e "line 1\nline 2\nline 3" > "$S/diff_a.txt"
echo -e "line 1\nline 2 modified\nline 3\nline 4 new" > "$S/diff_b.txt"
run "diff" diff "$S/diff_a.txt" "$S/diff_b.txt" || true
run "diff -u" diff -u "$S/diff_a.txt" "$S/diff_b.txt" || true
run "diff -c" diff -c "$S/diff_a.txt" "$S/diff_b.txt" || true
run "diff -y" diff -y "$S/diff_a.txt" "$S/diff_b.txt" || true
run "diff --color" diff --color "$S/diff_a.txt" "$S/diff_b.txt" 2>/dev/null || true
run "diff -q" diff -q "$S/diff_a.txt" "$S/diff_b.txt" || true
run "diff -r" diff -r "$S/tree" "$S/tree" || true
run "diff -rq" diff -rq "$S/tree" "$S/" 2>/dev/null | head -5 || true
run "patch apply" diff -u "$S/diff_a.txt" "$S/diff_b.txt" > "$S/my.patch" || true && patch "$S/diff_a.txt" "$S/my.patch" 2>/dev/null || true
run "patch reverse" patch -R "$S/diff_a.txt" "$S/my.patch" 2>/dev/null || true

# ============================================================================
section "Archive and compression — tar, gzip, bzip2, xz, zip, 7z, zstd, lz4"
# ============================================================================
# tar
run "tar czf" tar czf "$S/arch.tar.gz" -C "$S" tiny.txt small.txt multiline.txt
run "tar cjf" tar cjf "$S/arch.tar.bz2" -C "$S" tiny.txt small.txt
run "tar cJf" tar cJf "$S/arch.tar.xz" -C "$S" tiny.txt small.txt
run "tar cf" tar cf "$S/arch.tar" -C "$S" tiny.txt small.txt medium.txt
run "tar czf dir" tar czf "$S/tree.tar.gz" -C "$S" tree
run "tar tzf" tar tzf "$S/arch.tar.gz"
run "tar tjf" tar tjf "$S/arch.tar.bz2"
run "tar tJf" tar tJf "$S/arch.tar.xz"
run "tar xzf" mkdir -p "$S/untar" && tar xzf "$S/arch.tar.gz" -C "$S/untar/"
run "tar xjf" tar xjf "$S/arch.tar.bz2" -C "$S/untar/"
run "tar xJf" tar xJf "$S/arch.tar.xz" -C "$S/untar/"
run "tar --strip" tar xzf "$S/tree.tar.gz" -C "$S/untar/" --strip-components=1
run "tar -rvf append" tar rvf "$S/arch.tar" -C "$S" data.json 2>/dev/null
run "tar --exclude" tar czf "$S/exclude.tar.gz" -C "$S" --exclude="*.bin" tree
run "tar -tvf" tar tvf "$S/arch.tar"
rm -rf "$S/untar"

# gzip/gunzip
cp "$S/medium.txt" "$S/gz_test.txt"
run "gzip" gzip "$S/gz_test.txt"
run "gunzip" gunzip "$S/gz_test.txt.gz"
run "gzip -k" gzip -k "$S/gz_test.txt"
run "gzip -1" gzip -1 -k -f "$S/gz_test.txt"
run "gzip -9" gzip -9 -k -f "$S/gz_test.txt"
run "gzip -l" gzip -l "$S/gz_test.txt.gz"
run "gzip -t" gzip -t "$S/gz_test.txt.gz"
run "zcat" zcat "$S/gz_test.txt.gz" | wc -l
run "zgrep" zgrep "100" "$S/gz_test.txt.gz"
rm -f "$S/gz_test.txt" "$S/gz_test.txt.gz"

# bzip2/bunzip2
cp "$S/medium.txt" "$S/bz_test.txt"
run "bzip2" bzip2 "$S/bz_test.txt"
run "bunzip2" bunzip2 "$S/bz_test.txt.bz2"
run "bzip2 -k" bzip2 -k "$S/bz_test.txt"
run "bzip2 -1" bzip2 -1 -k -f "$S/bz_test.txt"
run "bzip2 -9" bzip2 -9 -k -f "$S/bz_test.txt"
run "bzcat" bzcat "$S/bz_test.txt.bz2" | wc -l
rm -f "$S/bz_test.txt" "$S/bz_test.txt.bz2"

# xz
cp "$S/medium.txt" "$S/xz_test.txt"
run "xz" xz "$S/xz_test.txt"
run "unxz" unxz "$S/xz_test.txt.xz"
run "xz -k" xz -k "$S/xz_test.txt"
run "xz -0" xz -0 -k -f "$S/xz_test.txt"
run "xz -9" xz -9 -k -f "$S/xz_test.txt"
run "xz -l" xz -l "$S/xz_test.txt.xz"
run "xz -t" xz -t "$S/xz_test.txt.xz"
run "xzcat" xzcat "$S/xz_test.txt.xz" | wc -l
rm -f "$S/xz_test.txt" "$S/xz_test.txt.xz"

# zip/unzip
run "zip" zip -j "$S/test.zip" "$S/tiny.txt" "$S/small.txt"
run "zip -r" zip -r "$S/tree.zip" "$S/tree"
run "zip -e" zip -j -P testpass "$S/encrypted.zip" "$S/tiny.txt" 2>/dev/null
run "zip -9" zip -j -9 "$S/best.zip" "$S/medium.txt"
run "unzip -l" unzip -l "$S/test.zip"
run "unzip" unzip -o "$S/test.zip" -d "$S/unzipped/"
run "unzip -t" unzip -t "$S/test.zip"
rm -rf "$S/unzipped" "$S"/*.zip

# 7z
run "7z a" 7z a "$S/test.7z" "$S/tiny.txt" "$S/small.txt" 2>/dev/null
run "7z l" 7z l "$S/test.7z" 2>/dev/null
run "7z t" 7z t "$S/test.7z" 2>/dev/null
rm -f "$S/test.7z"

# zstd
run "zstd" zstd -q "$S/medium.txt" -o "$S/medium.txt.zst" 2>/dev/null
run "zstd -d" zstd -q -d "$S/medium.txt.zst" -o "$S/medium_zst.txt" 2>/dev/null
run "zstd -1" zstd -q -1 "$S/medium.txt" -o "$S/zst1.zst" -f 2>/dev/null
run "zstd -19" zstd -q -19 "$S/medium.txt" -o "$S/zst19.zst" -f 2>/dev/null
rm -f "$S"/*.zst "$S/medium_zst.txt"

# lz4
run "lz4" lz4 -q "$S/medium.txt" "$S/medium.txt.lz4" 2>/dev/null
run "lz4 -d" lz4 -q -d "$S/medium.txt.lz4" "$S/medium_lz4.txt" 2>/dev/null
rm -f "$S"/*.lz4 "$S/medium_lz4.txt"

# brotli
run "brotli" brotli -k "$S/medium.txt" 2>/dev/null
run "brotli -d" brotli -d "$S/medium.txt.br" -o "$S/medium_br.txt" 2>/dev/null
rm -f "$S"/*.br "$S/medium_br.txt"

# ============================================================================
section "Encoding and hashing"
# ============================================================================
run "base64 encode" base64 "$S/tiny.txt"
run "base64 -w 0" base64 -w 0 "$S/tiny.txt"
run "base64 decode" echo "SGVsbG8gV29ybGQ=" | base64 -d
run "base64 roundtrip" base64 "$S/random_1k.bin" | base64 -d | cmp - "$S/random_1k.bin"
run "xxd" xxd "$S/random_1k.bin" | head -10
run "xxd -p" xxd -p "$S/tiny.txt"
run "xxd -r" xxd "$S/random_1k.bin" | xxd -r > "$S/xxd_restore.bin"
run "xxd -i" xxd -i "$S/tiny.txt" | head -5
run "od -A x" od -A x -t x1 "$S/random_1k.bin" | head -5
run "od -c" od -c "$S/tiny.txt" | head -5
run "od -d" od -d "$S/random_1k.bin" | head -5
run "hexdump -C" hexdump -C "$S/random_1k.bin" | head -5
run "hexdump -v" hexdump -v -e '16/1 "%02x " "\n"' "$S/random_1k.bin" | head -5
run "strings" strings "$S/random_1k.bin"
run "strings -n 8" strings -n 8 /usr/bin/ls | head -10
run "strings -a" strings -a /usr/bin/ls | wc -l

run "md5sum" md5sum "$S/tiny.txt"
run "md5sum multiple" md5sum "$S/tiny.txt" "$S/small.txt" "$S/medium.txt"
run "md5sum -c" md5sum "$S/tiny.txt" > "$S/md5.txt" && md5sum -c "$S/md5.txt"
run "sha1sum" sha1sum "$S/tiny.txt"
run "sha224sum" sha224sum "$S/tiny.txt"
run "sha256sum" sha256sum "$S/tiny.txt"
run "sha384sum" sha384sum "$S/tiny.txt"
run "sha512sum" sha512sum "$S/tiny.txt"
run "sha256sum -c" sha256sum "$S/tiny.txt" > "$S/sha256.txt" && sha256sum -c "$S/sha256.txt"
run "b2sum" b2sum "$S/tiny.txt" 2>/dev/null
run "cksum" cksum "$S/tiny.txt"
run "sum" sum "$S/tiny.txt"
run "openssl md5" openssl dgst -md5 "$S/tiny.txt"
run "openssl sha256" openssl dgst -sha256 "$S/tiny.txt"
run "openssl sha3" openssl dgst -sha3-256 "$S/tiny.txt" 2>/dev/null

rm -f "$S/md5.txt" "$S/sha256.txt" "$S/xxd_restore.bin"

# Cleanup
rm -f "$S/new_touch.txt" "$S/sorted.txt" "$S/s1.txt" "$S/s2.txt"
rm -f "$S/tee"*.txt "$S/sed_script.sed" "$S/hosts_edit" "$S/hosts_bak" "$S/hosts_bak.bak"
rm -f "$S/patterns.txt" "$S/diff_a.txt" "$S/diff_b.txt" "$S/my.patch"
rm -f "$S/concatenated.txt" "$S/arch"*.tar* "$S/tree.tar.gz" "$S/exclude.tar.gz"

domain_end
