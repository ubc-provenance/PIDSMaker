#!/bin/bash
# ============================================================================
# DOMAIN 03: USER & PERMISSION MANAGEMENT
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "03 — USER & PERMISSIONS"

S=$(generate_sample_files)

# ============================================================================
section "Identity queries"
# ============================================================================
run "id" id
run "id root" id root
run "id -u" id -u
run "id -g" id -g
run "id -G" id -G
run "id -un" id -un
run "id -gn" id -gn
run "id -Gn" id -Gn
run "whoami" whoami
run "who" who 2>/dev/null || true
run "who -a" who -a 2>/dev/null || true
run "who -b" who -b 2>/dev/null || true
run "who -r" who -r 2>/dev/null || true
run "w" w 2>/dev/null || true
run "last" last -10 2>/dev/null || true
run "last -a" last -a -10 2>/dev/null || true
run "last -i" last -i -10 2>/dev/null || true
run "last -x" last -x -10 2>/dev/null || true
run "lastlog" lastlog 2>/dev/null || true
run "lastb" lastb -5 2>/dev/null || true
run "finger" finger 2>/dev/null || true
run "finger root" finger root 2>/dev/null || true
run "users" users 2>/dev/null || true
run "logname" logname 2>/dev/null || true
run "groups" groups
run "groups root" groups root

# ============================================================================
section "Reading account databases"
# ============================================================================
run "getent passwd" getent passwd | head -15
run "getent passwd root" getent passwd root
run "getent group" getent group | head -15
run "getent group root" getent group root
run "getent shadow" getent shadow 2>/dev/null | head -5
run "getent hosts" getent hosts localhost
run "getent services" getent services http
run "getent protocols" getent protocols tcp
run "getent networks" getent networks 2>/dev/null | head -5
run "cat /etc/passwd" cat /etc/passwd
run "cat /etc/group" cat /etc/group
run "cat /etc/shadow" cat /etc/shadow 2>/dev/null || true
run "cat /etc/gshadow" cat /etc/gshadow 2>/dev/null || true
run "cat /etc/shells" cat /etc/shells 2>/dev/null || true
run "cat /etc/login.defs" cat /etc/login.defs 2>/dev/null | head -20
run "cat /etc/security/limits.conf" cat /etc/security/limits.conf 2>/dev/null | head -10
run "cat /etc/nsswitch.conf" cat /etc/nsswitch.conf 2>/dev/null

# ============================================================================
section "User management (create, modify, delete)"
# ============================================================================
# Create users with various options
run "useradd basic" useradd testuser_a 2>/dev/null
run "useradd -m homedir" useradd -m testuser_b 2>/dev/null
run "useradd -m -s bash" useradd -m -s /bin/bash testuser_c 2>/dev/null
run "useradd -m -s sh" useradd -m -s /bin/sh testuser_d 2>/dev/null
run "useradd -m -s nologin" useradd -m -s /usr/sbin/nologin testuser_e 2>/dev/null
run "useradd -r system" useradd -r -s /usr/sbin/nologin svc_test_a 2>/dev/null
run "useradd -r -d" useradd -r -d /opt/svctest -s /usr/sbin/nologin svc_test_b 2>/dev/null
run "useradd -u uid" useradd -u 5001 testuser_f 2>/dev/null
run "useradd -g group" useradd -g root testuser_g 2>/dev/null
run "useradd -G groups" useradd -G root,adm testuser_h 2>/dev/null
run "useradd -c comment" useradd -c "Test User" -m testuser_i 2>/dev/null
run "useradd -e expire" useradd -e 2030-01-01 -m testuser_j 2>/dev/null
run "useradd -K" useradd -K PASS_MAX_DAYS=90 -m testuser_k 2>/dev/null || true

# Modify users
run "usermod -s" usermod -s /bin/sh testuser_c 2>/dev/null || true
run "usermod -s back" usermod -s /bin/bash testuser_c 2>/dev/null || true
run "usermod -aG" usermod -aG root testuser_b 2>/dev/null || true
run "usermod -c" usermod -c "Modified Comment" testuser_b 2>/dev/null || true
run "usermod -d" usermod -d /tmp/newhome testuser_b 2>/dev/null || true
run "usermod -l rename" usermod -l testuser_b_renamed testuser_b 2>/dev/null || true
run "usermod -L lock" usermod -L testuser_c 2>/dev/null || true
run "usermod -U unlock" usermod -U testuser_c 2>/dev/null || true
run "usermod -e expire" usermod -e 2031-01-01 testuser_c 2>/dev/null || true

# Password operations
run "passwd --status" passwd --status testuser_c 2>/dev/null || true
run "passwd --status root" passwd --status root 2>/dev/null || true
run "chage -l" chage -l testuser_c 2>/dev/null || true
run "chage -M 90" chage -M 90 testuser_c 2>/dev/null || true
run "chage -m 7" chage -m 7 testuser_c 2>/dev/null || true
run "chage -W 14" chage -W 14 testuser_c 2>/dev/null || true
run "chage -E" chage -E 2030-06-01 testuser_c 2>/dev/null || true
run "chage -I" chage -I 30 testuser_c 2>/dev/null || true
run "chage -l root" chage -l root 2>/dev/null || true

# Group management
run "groupadd" groupadd testgroup_a 2>/dev/null
run "groupadd -g" groupadd -g 5001 testgroup_b 2>/dev/null
run "groupadd -r system" groupadd -r svc_group 2>/dev/null
run "groupmod rename" groupmod -n testgroup_a_renamed testgroup_a 2>/dev/null || true
run "groupmod -g" groupmod -g 5005 testgroup_b 2>/dev/null || true
run "gpasswd -a" gpasswd -a testuser_c testgroup_b 2>/dev/null || true
run "gpasswd -d" gpasswd -d testuser_c testgroup_b 2>/dev/null || true
run "gpasswd -A admin" gpasswd -A testuser_c testgroup_b 2>/dev/null || true
run "gpasswd -M members" gpasswd -M testuser_c,testuser_d testgroup_b 2>/dev/null || true

# su / sudo
run "su -c" su -c "echo running as root" root 2>/dev/null || true
run "su -c id" su -c "id" root 2>/dev/null || true
run "su -s" su -s /bin/sh -c "echo sh shell" root 2>/dev/null || true
run "su testuser" su -c "id" testuser_c 2>/dev/null || true
run "sudo -l" sudo -l 2>/dev/null || true
run "sudo whoami" sudo whoami 2>/dev/null || true
run "sudo -u" sudo -u testuser_c whoami 2>/dev/null || true
run "sudo -i" sudo -i whoami 2>/dev/null || true
run "sudo env" sudo env 2>/dev/null | head -5 || true
run "sudo -V" sudo -V 2>/dev/null | head -5 || true
run "sudo cat shadow" sudo cat /etc/shadow 2>/dev/null | head -3 || true

# Read sudo/auth configs
run "cat /etc/sudoers" cat /etc/sudoers 2>/dev/null || true
run "ls /etc/sudoers.d" ls -la /etc/sudoers.d/ 2>/dev/null || true
run "cat /etc/pam.d/sshd" cat /etc/pam.d/sshd 2>/dev/null || true
run "cat /etc/pam.d/sudo" cat /etc/pam.d/sudo 2>/dev/null || true
run "cat /etc/pam.d/login" cat /etc/pam.d/login 2>/dev/null || true
run "cat /etc/pam.d/common-auth" cat /etc/pam.d/common-auth 2>/dev/null || true
run "cat /etc/pam.d/common-password" cat /etc/pam.d/common-password 2>/dev/null || true
run "ls /etc/pam.d" ls /etc/pam.d/ 2>/dev/null

# ============================================================================
section "ACLs and extended attributes"
# ============================================================================
run "setfacl -m user" setfacl -m u:testuser_c:rw "$S/tiny.txt" 2>/dev/null || true
run "setfacl -m group" setfacl -m g:testgroup_b:r "$S/tiny.txt" 2>/dev/null || true
run "setfacl -m other" setfacl -m o::--- "$S/tiny.txt" 2>/dev/null || true
run "setfacl -m default" setfacl -m d:u:testuser_c:rw "$S/tree" 2>/dev/null || true
run "getfacl" getfacl "$S/tiny.txt" 2>/dev/null || true
run "getfacl tree" getfacl "$S/tree" 2>/dev/null || true
run "setfacl -b remove" setfacl -b "$S/tiny.txt" 2>/dev/null || true
run "setfacl -R" setfacl -R -m u:testuser_c:r "$S/tree" 2>/dev/null || true
run "setfacl -R remove" setfacl -R -b "$S/tree" 2>/dev/null || true

run "setattr +a" setfattr -n user.prov -v test "$S/tiny.txt" 2>/dev/null || true
run "getattr" getfattr -d "$S/tiny.txt" 2>/dev/null || true
run "getfattr -n" getfattr -n user.prov "$S/tiny.txt" 2>/dev/null || true
run "setfattr remove" setfattr -x user.prov "$S/tiny.txt" 2>/dev/null || true

# ============================================================================
section "Capabilities"
# ============================================================================
run "getcap /usr/bin" getcap -r /usr/bin/ 2>/dev/null | head -10
run "getcap /usr/sbin" getcap -r /usr/sbin/ 2>/dev/null | head -10
run "getcap -v" getcap -v /usr/bin/ping 2>/dev/null || true
run "setcap" cp /usr/bin/cat "$S/cap_cat" && setcap cap_net_raw+ep "$S/cap_cat" 2>/dev/null || true
run "getcap set" getcap "$S/cap_cat" 2>/dev/null || true
run "setcap remove" setcap -r "$S/cap_cat" 2>/dev/null || true
rm -f "$S/cap_cat"

# ============================================================================
section "Cleanup test users and groups"
# ============================================================================
for u in testuser_a testuser_b_renamed testuser_c testuser_d testuser_e testuser_f testuser_g testuser_h testuser_i testuser_j testuser_k; do
    run "userdel $u" userdel -r "$u" 2>/dev/null || true
done
for u in svc_test_a svc_test_b; do
    run "userdel $u" userdel "$u" 2>/dev/null || true
done
for g in testgroup_a_renamed testgroup_b svc_group; do
    run "groupdel $g" groupdel "$g" 2>/dev/null || true
done

domain_end
