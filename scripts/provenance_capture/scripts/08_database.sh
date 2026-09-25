#!/bin/bash
# ============================================================================
# DOMAIN 08: DATABASE ACTIVITY
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "08 — DATABASE"

DB="$PROV_WORKDIR/db"
mkdir -p "$DB"

section "SQLite3 — comprehensive"
run "sqlite3 create table" sqlite3 "$DB/test.db" "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, email TEXT, age INTEGER, role TEXT);"
run "sqlite3 insert" sqlite3 "$DB/test.db" "INSERT INTO users VALUES(1,'alice','alice@ex.com',30,'admin'),(2,'bob','bob@ex.com',25,'user'),(3,'charlie','charlie@ex.com',35,'admin'),(4,'dave','dave@ex.com',28,'user'),(5,'eve','eve@ex.com',22,'user');"
run "sqlite3 select *" sqlite3 "$DB/test.db" "SELECT * FROM users;"
run "sqlite3 select where" sqlite3 "$DB/test.db" "SELECT * FROM users WHERE role='admin';"
run "sqlite3 select order" sqlite3 "$DB/test.db" "SELECT * FROM users ORDER BY age DESC;"
run "sqlite3 select limit" sqlite3 "$DB/test.db" "SELECT * FROM users LIMIT 3;"
run "sqlite3 select count" sqlite3 "$DB/test.db" "SELECT role, COUNT(*) FROM users GROUP BY role;"
run "sqlite3 select avg" sqlite3 "$DB/test.db" "SELECT role, AVG(age) FROM users GROUP BY role;"
run "sqlite3 select like" sqlite3 "$DB/test.db" "SELECT * FROM users WHERE name LIKE 'a%';"
run "sqlite3 update" sqlite3 "$DB/test.db" "UPDATE users SET email='alice@new.com' WHERE name='alice';"
run "sqlite3 delete" sqlite3 "$DB/test.db" "DELETE FROM users WHERE id=5;"
run "sqlite3 alter table" sqlite3 "$DB/test.db" "ALTER TABLE users ADD COLUMN created TEXT;"
run "sqlite3 create index" sqlite3 "$DB/test.db" "CREATE INDEX idx_role ON users(role);"
run "sqlite3 create index name" sqlite3 "$DB/test.db" "CREATE INDEX idx_name ON users(name);"
run "sqlite3 explain" sqlite3 "$DB/test.db" "EXPLAIN QUERY PLAN SELECT * FROM users WHERE role='admin';"
run "sqlite3 .schema" sqlite3 "$DB/test.db" ".schema"
run "sqlite3 .tables" sqlite3 "$DB/test.db" ".tables"
run "sqlite3 .indices" sqlite3 "$DB/test.db" ".indices"
run "sqlite3 .dump" sqlite3 "$DB/test.db" ".dump" > "$DB/dump.sql"
run "sqlite3 import dump" sqlite3 "$DB/test2.db" < "$DB/dump.sql"
run "sqlite3 vacuum" sqlite3 "$DB/test.db" "VACUUM;"
run "sqlite3 analyze" sqlite3 "$DB/test.db" "ANALYZE;"
run "sqlite3 integrity" sqlite3 "$DB/test.db" "PRAGMA integrity_check;"
run "sqlite3 journal mode" sqlite3 "$DB/test.db" "PRAGMA journal_mode;"
run "sqlite3 wal mode" sqlite3 "$DB/test.db" "PRAGMA journal_mode=WAL;"
run "sqlite3 foreign keys" sqlite3 "$DB/test.db" "PRAGMA foreign_keys=ON;"
run "sqlite3 table info" sqlite3 "$DB/test.db" "PRAGMA table_info(users);"

# Create large table
run "sqlite3 large insert" python3 -c "
import sqlite3; c=sqlite3.connect('$DB/large.db')
cur=c.cursor(); cur.execute('CREATE TABLE data(id INTEGER,key TEXT,value REAL,category TEXT)')
for i in range(50000): cur.execute('INSERT INTO data VALUES(?,?,?,?)',(i,f'key_{i}',i*0.01,['A','B','C','D'][i%4]))
c.commit(); print(f'Rows: {cur.execute(\"SELECT COUNT(*) FROM data\").fetchone()[0]}')
c.close()
"
run "sqlite3 large query" sqlite3 "$DB/large.db" "SELECT category, COUNT(*), AVG(value) FROM data GROUP BY category;"
run "sqlite3 large index" sqlite3 "$DB/large.db" "CREATE INDEX idx_cat ON data(category);"
run "sqlite3 large range" sqlite3 "$DB/large.db" "SELECT * FROM data WHERE value BETWEEN 100 AND 101 LIMIT 5;"

# Join
run "sqlite3 second table" sqlite3 "$DB/test.db" "CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL);"
run "sqlite3 insert orders" sqlite3 "$DB/test.db" "INSERT INTO orders VALUES(1,1,'Widget',9.99),(2,1,'Gadget',19.99),(3,2,'Widget',9.99),(4,3,'Doohickey',29.99);"
run "sqlite3 join" sqlite3 "$DB/test.db" "SELECT u.name, o.product, o.amount FROM users u JOIN orders o ON u.id=o.user_id;"
run "sqlite3 left join" sqlite3 "$DB/test.db" "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id=o.user_id GROUP BY u.name;"
run "sqlite3 subquery" sqlite3 "$DB/test.db" "SELECT * FROM users WHERE id IN (SELECT DISTINCT user_id FROM orders);"

section "Redis (if available)"
run "redis-cli ping" redis-cli ping 2>/dev/null
run "redis-cli set" redis-cli set prov:key1 hello 2>/dev/null
run "redis-cli get" redis-cli get prov:key1 2>/dev/null
run "redis-cli mset" redis-cli mset prov:a 1 prov:b 2 prov:c 3 2>/dev/null
run "redis-cli mget" redis-cli mget prov:a prov:b prov:c 2>/dev/null
run "redis-cli incr" redis-cli incr prov:counter 2>/dev/null
run "redis-cli lpush" redis-cli lpush prov:list a b c 2>/dev/null
run "redis-cli lrange" redis-cli lrange prov:list 0 -1 2>/dev/null
run "redis-cli sadd" redis-cli sadd prov:set x y z 2>/dev/null
run "redis-cli smembers" redis-cli smembers prov:set 2>/dev/null
run "redis-cli hset" redis-cli hset prov:hash name alice age 30 2>/dev/null
run "redis-cli hgetall" redis-cli hgetall prov:hash 2>/dev/null
run "redis-cli keys" redis-cli keys "prov:*" 2>/dev/null
run "redis-cli info" redis-cli info server 2>/dev/null | head -10
run "redis-cli dbsize" redis-cli dbsize 2>/dev/null
run "redis-cli del" redis-cli del prov:key1 prov:a prov:b prov:c prov:counter prov:list prov:set prov:hash 2>/dev/null

section "PostgreSQL client"
run "psql --version" psql --version 2>/dev/null
run "pg_isready" pg_isready 2>/dev/null || true
run "pg_dump --help" pg_dump --help 2>/dev/null | head -5

section "MySQL client"
run "mysql --version" mysql --version 2>/dev/null
run "mysqldump --help" mysqldump --help 2>/dev/null | head -5

rm -rf "$DB"
domain_end
