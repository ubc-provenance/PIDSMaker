#!/bin/bash
# ============================================================================
# DOMAIN 10: CRYPTOGRAPHY & SECURITY TOOLS
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "10 — CRYPTOGRAPHY"
K="$PROV_WORKDIR/keys"; mkdir -p "$K"
echo "Secret provenance data for encryption tests" > "$K/plaintext.txt"

section "OpenSSL key generation"
run "openssl genrsa 1024" openssl genrsa -out "$K/rsa1024.pem" 1024 2>/dev/null
run "openssl genrsa 2048" openssl genrsa -out "$K/rsa2048.pem" 2048 2>/dev/null
run "openssl genrsa 4096" openssl genrsa -out "$K/rsa4096.pem" 4096 2>/dev/null
run "openssl ec prime256v1" openssl ecparam -genkey -name prime256v1 -out "$K/ec256.pem" 2>/dev/null
run "openssl ec secp384r1" openssl ecparam -genkey -name secp384r1 -out "$K/ec384.pem" 2>/dev/null
run "openssl ec secp521r1" openssl ecparam -genkey -name secp521r1 -out "$K/ec521.pem" 2>/dev/null
run "openssl ed25519" openssl genpkey -algorithm Ed25519 -out "$K/ed25519.pem" 2>/dev/null
run "openssl ed448" openssl genpkey -algorithm Ed448 -out "$K/ed448.pem" 2>/dev/null || true
run "openssl x25519" openssl genpkey -algorithm X25519 -out "$K/x25519.pem" 2>/dev/null

section "OpenSSL certificates"
run "openssl CSR" openssl req -new -key "$K/rsa2048.pem" -out "$K/csr.pem" -subj "/CN=prov.test/O=Test/C=US" 2>/dev/null
run "openssl self-signed" openssl req -x509 -newkey rsa:2048 -keyout "$K/ss_key.pem" -out "$K/ss_cert.pem" -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
run "openssl self-signed EC" openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -keyout "$K/ss_ec_key.pem" -out "$K/ss_ec_cert.pem" -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
run "openssl x509 text" openssl x509 -in "$K/ss_cert.pem" -text -noout 2>/dev/null
run "openssl x509 dates" openssl x509 -in "$K/ss_cert.pem" -dates -noout 2>/dev/null
run "openssl x509 subject" openssl x509 -in "$K/ss_cert.pem" -subject -noout 2>/dev/null
run "openssl x509 issuer" openssl x509 -in "$K/ss_cert.pem" -issuer -noout 2>/dev/null
run "openssl x509 serial" openssl x509 -in "$K/ss_cert.pem" -serial -noout 2>/dev/null
run "openssl x509 fingerprint" openssl x509 -in "$K/ss_cert.pem" -fingerprint -noout 2>/dev/null
run "openssl x509 pubkey" openssl x509 -in "$K/ss_cert.pem" -pubkey -noout 2>/dev/null
run "openssl verify" openssl verify -CAfile "$K/ss_cert.pem" "$K/ss_cert.pem" 2>/dev/null
run "openssl rsa pubout" openssl rsa -in "$K/rsa2048.pem" -pubout -out "$K/rsa2048_pub.pem" 2>/dev/null
run "openssl rsa check" openssl rsa -in "$K/rsa2048.pem" -check 2>/dev/null
run "openssl ec pubout" openssl ec -in "$K/ec256.pem" -pubout -out "$K/ec256_pub.pem" 2>/dev/null

section "OpenSSL encryption/decryption"
for cipher in aes-128-cbc aes-192-cbc aes-256-cbc aes-256-gcm chacha20 des-ede3-cbc; do
    run "openssl enc $cipher" openssl enc -$cipher -salt -in "$K/plaintext.txt" -out "$K/enc_${cipher}.bin" -pass pass:test123 -pbkdf2 2>/dev/null
    run "openssl dec $cipher" openssl enc -d -$cipher -in "$K/enc_${cipher}.bin" -out "$K/dec_${cipher}.txt" -pass pass:test123 -pbkdf2 2>/dev/null
done

section "OpenSSL hashing"
for dgst in md5 sha1 sha224 sha256 sha384 sha512 sha3-256 sha3-512 blake2b512 blake2s256; do
    run "openssl dgst $dgst" openssl dgst -$dgst "$K/plaintext.txt" 2>/dev/null
done

section "OpenSSL signing"
run "openssl sign RSA" openssl dgst -sha256 -sign "$K/rsa2048.pem" -out "$K/sig_rsa.bin" "$K/plaintext.txt" 2>/dev/null
run "openssl verify RSA" openssl dgst -sha256 -verify "$K/rsa2048_pub.pem" -signature "$K/sig_rsa.bin" "$K/plaintext.txt" 2>/dev/null
run "openssl sign EC" openssl dgst -sha256 -sign "$K/ec256.pem" -out "$K/sig_ec.bin" "$K/plaintext.txt" 2>/dev/null

section "OpenSSL s_client"
for host in google.com:443 github.com:443 example.com:443; do
    run_t 10 "openssl s_client $host" bash -c "echo Q | openssl s_client -connect $host -brief 2>/dev/null | head -5"
done
run_t 10 "openssl s_client showcerts" bash -c 'echo Q | openssl s_client -connect google.com:443 -showcerts 2>/dev/null | head -30'
run_t 10 "openssl s_client -tls1_2" bash -c 'echo Q | openssl s_client -connect google.com:443 -tls1_2 2>/dev/null | head -5'
run_t 10 "openssl s_client -tls1_3" bash -c 'echo Q | openssl s_client -connect google.com:443 -tls1_3 2>/dev/null | head -5'

section "OpenSSL misc"
run "openssl rand hex" openssl rand -hex 16
run "openssl rand hex 32" openssl rand -hex 32
run "openssl rand hex 64" openssl rand -hex 64
run "openssl rand base64" openssl rand -base64 32
run "openssl rand binary" openssl rand -out "$K/random.bin" 256 2>/dev/null
run "openssl version" openssl version
run "openssl version -a" openssl version -a
run "openssl ciphers" openssl ciphers -v | head -20
run "openssl list" openssl list -cipher-algorithms 2>/dev/null | head -20
run "openssl list digest" openssl list -digest-algorithms 2>/dev/null | head -20
run "openssl passwd" openssl passwd -6 "testpassword" 2>/dev/null
run "openssl passwd -5" openssl passwd -5 "testpassword" 2>/dev/null
run "openssl passwd -1" openssl passwd -1 "testpassword" 2>/dev/null

section "GPG"
run "gpg gen-key" gpg --batch --gen-key --passphrase "" 2>/dev/null << 'GPGEOF'
Key-Type: RSA
Key-Length: 2048
Name-Real: Provenance Test
Name-Email: prov@test.local
Expire-Date: 0
%no-protection
GPGEOF
run "gpg list-keys" gpg --list-keys 2>/dev/null
run "gpg list-secret" gpg --list-secret-keys 2>/dev/null
run "gpg fingerprint" gpg --fingerprint 2>/dev/null
run "gpg export" gpg --export --armor "prov@test.local" > "$K/gpg_pub.asc" 2>/dev/null
run "gpg export secret" gpg --export-secret-keys --armor "prov@test.local" > "$K/gpg_sec.asc" 2>/dev/null
run "gpg encrypt" echo "secret" | gpg --armor --encrypt --recipient "prov@test.local" -o "$K/gpg_enc.asc" 2>/dev/null
run "gpg decrypt" gpg --decrypt "$K/gpg_enc.asc" 2>/dev/null
run "gpg sign" echo "signed" | gpg --armor --sign -o "$K/gpg_sig.asc" 2>/dev/null
run "gpg clearsign" echo "clearsigned" | gpg --armor --clearsign -o "$K/gpg_clear.asc" 2>/dev/null
run "gpg detach-sign" gpg --armor --detach-sign -o "$K/gpg_det.asc" "$K/plaintext.txt" 2>/dev/null
run "gpg verify" gpg --verify "$K/gpg_sig.asc" 2>/dev/null
run "gpg verify detached" gpg --verify "$K/gpg_det.asc" "$K/plaintext.txt" 2>/dev/null

domain_end
