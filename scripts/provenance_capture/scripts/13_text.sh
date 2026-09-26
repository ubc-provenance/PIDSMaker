#!/bin/bash
# ============================================================================
# DOMAIN 13: TEXT PROCESSING & DATA FORMATS
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "13 — TEXT PROCESSING"
S=$(generate_sample_files)
echo '{"users":[{"name":"alice","age":30,"role":"admin"},{"name":"bob","age":25,"role":"user"},{"name":"charlie","age":35}],"total":3}' > "$S/data.json"
echo '<?xml version="1.0"?><root><users><user name="alice" age="30"/><user name="bob" age="25"/></users></root>' > "$S/data.xml"

section "jq"
run "jq ." jq '.' "$S/data.json"
run "jq .users" jq '.users' "$S/data.json"
run "jq .users[0]" jq '.users[0]' "$S/data.json"
run "jq .users[].name" jq '.users[].name' "$S/data.json"
run "jq select" jq '.users[]|select(.role=="admin")' "$S/data.json"
run "jq map" jq '[.users[]|.name]' "$S/data.json"
run "jq length" jq '.users|length' "$S/data.json"
run "jq keys" jq '.users[0]|keys' "$S/data.json"
run "jq to_entries" jq '.users[0]|to_entries' "$S/data.json"
run "jq sort_by" jq '.users|sort_by(.age)' "$S/data.json"
run "jq group_by" jq '[.users[]|.role]|group_by(.)|map({key:.[0],count:length})' "$S/data.json"
run "jq add" jq '[.users[]|.age]|add' "$S/data.json"
run "jq min max" jq '[.users[]|.age]|[min,max]' "$S/data.json"
run "jq -r raw" jq -r '.users[].name' "$S/data.json"
run "jq -c compact" jq -c '.users[]' "$S/data.json"
run "jq -S sort keys" jq -S '.' "$S/data.json"
run "jq -n create" jq -n '{"created":true,"list":[1,2,3]}'
run "jq env" jq -n --arg h "$(hostname)" '{"host":$h}'
run "jq slurp" echo -e '{"a":1}\n{"b":2}' | jq -s '.'
run "jq @base64" jq -n '"hello"|@base64'
run "jq @uri" jq -n '"hello world"|@uri'
run "jq if-then" jq '.users[]|if .age>30 then "senior" else "junior" end' "$S/data.json"
run "jq try-catch" jq 'try .nonexistent catch "default"' "$S/data.json"

section "xmlstarlet"
run "xmlstarlet sel" xmlstarlet sel -t -v "//user/@name" "$S/data.xml" 2>/dev/null
run "xmlstarlet sel -t -c" xmlstarlet sel -t -c "//user" "$S/data.xml" 2>/dev/null
run "xmlstarlet fo" xmlstarlet fo "$S/data.xml" 2>/dev/null
run "xmlstarlet val" xmlstarlet val "$S/data.xml" 2>/dev/null
run "xmlstarlet el" xmlstarlet el "$S/data.xml" 2>/dev/null
run "xmlstarlet ed" xmlstarlet ed -s /root -type elem -n new -v "test" "$S/data.xml" 2>/dev/null

section "Text manipulation"
run "rev" echo "Hello World" | rev
run "tac" echo -e "1\n2\n3" | tac
run "paste" paste -d, <(echo -e "a\nb\nc") <(echo -e "1\n2\n3")
run "join" echo -e "1 alice\n2 bob" > /tmp/j1.txt; echo -e "1 admin\n2 user" > /tmp/j2.txt; join /tmp/j1.txt /tmp/j2.txt; rm /tmp/j1.txt /tmp/j2.txt
run "fold" seq 1 50 | tr '\n' ' ' | fold -w 30
run "fmt" echo "This is a very long line that should be reformatted by fmt to a reasonable width for reading" | fmt -w 40
run "column" echo -e "a:1:x\nb:2:y\nc:3:z" | column -t -s:
run "nl" nl /etc/passwd | head -5
run "expand" echo -e "tab\there" | expand -t 8
run "unexpand" echo "    spaces" | unexpand --first-only
run "pr" pr -t -2 /etc/passwd | head -10
run "comm" echo -e "a\nb\nc" > /tmp/c1; echo -e "b\nc\nd" > /tmp/c2; comm /tmp/c1 /tmp/c2; rm /tmp/c1 /tmp/c2
run "iconv" echo "Hello UTF-8" | iconv -f UTF-8 -t ASCII
run "dos2unix" echo -e "line1\r\nline2\r\n" > /tmp/dos.txt; dos2unix /tmp/dos.txt 2>/dev/null; rm /tmp/dos.txt
run "pandoc md->html" echo "# Hello" | pandoc -f markdown -t html 2>/dev/null
run "pandoc md->txt" echo "**bold** _italic_" | pandoc -f markdown -t plain 2>/dev/null
run "bc calculator" echo "scale=10; 4*a(1)" | bc -l
run "dc calculator" echo "2 3 + p" | dc
run "expr" expr 2 + 3
run "seq" seq 1 2 20
run "shuf" shuf -i 1-10 -n 5
run "yes head" yes "repeated" | head -5

domain_end
