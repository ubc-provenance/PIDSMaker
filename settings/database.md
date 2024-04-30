# Setting Databases

[1. CADETS E3](#cadets-e3)

[2. THEIA E3](#theia-e3)

[3. ClearScope E3](#clearscope-e3)

[7. ClearScope E3](#trace-e3)

[4. CADETS E5](#cadets-e5)

[5. THEIA E5](#theia-e5)

[6. ClearScope E5](#clearscope-e5)



## CADETS E3
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database cadets_e3;

# switch to the created database
postgres=# \connect cadets_e3;

# create the event table and grant the privileges to postgres
cadets_e3=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
cadets_e3=# alter table event_table owner to postgres;
cadets_e3=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
cadets_e3=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
cadets_e3=# alter table file_node_table owner to postgres;

# create the netflow table
cadets_e3=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
cadets_e3=# alter table netflow_node_table owner to postgres;

# create the subject table
cadets_e3=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
cadets_e3=# alter table subject_node_table owner to postgres;

# create the node2id table
cadets_e3=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
cadets_e3=# alter table node2id owner to postgres;
cadets_e3=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```


## THEIA E3
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database theia_e3;

# switch to the created database
postgres=# \connect theia_e3;

# create the event table and grant the privileges to postgres
theia_e3=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
theia_e3=# alter table event_table owner to postgres;
theia_e3=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
theia_e3=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
theia_e3=# alter table file_node_table owner to postgres;

# create the netflow table
theia_e3=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
theia_e3=# alter table netflow_node_table owner to postgres;

# create the subject table
theia_e3=# create table subject_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    "cmdLine" varchar not null,
    tgid      varchar not null,
    path      varchar not null,
    constraint subject_node_table_pk
        primary key (node_uuid, hash_id)
);
theia_e3=# alter table subject_node_table owner to postgres;

# create the node2id table
theia_e3=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
theia_e3=# alter table node2id owner to postgres;
theia_e3=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```


## ClearScope E3
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database clearscope_e3;

# switch to the created database
postgres=# \connect clearscope_e3;

# create the event table and grant the privileges to postgres
clearscope_e3=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
clearscope_e3=# alter table event_table owner to postgres;
clearscope_e3=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
clearscope_e3=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
clearscope_e3=# alter table file_node_table owner to postgres;

# create the netflow table
clearscope_e3=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
clearscope_e3=# alter table netflow_node_table owner to postgres;

# create the subject table
clearscope_e3=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
clearscope_e3=# alter table subject_node_table owner to postgres;

# create the node2id table
clearscope_e3=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
clearscope_e3=# alter table node2id owner to postgres;
clearscope_e3=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```


## Trace E3
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database trace_e3;

# switch to the created database
postgres=# \connect trace_e3;

# create the event table and grant the privileges to postgres
trace_e3=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
trace_e3=# alter table event_table owner to postgres;
trace_e3=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
trace_e3=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
trace_e3=# alter table file_node_table owner to postgres;

# create the netflow table
trace_e3=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
trace_e3=# alter table netflow_node_table owner to postgres;

# create the subject table
trace_e3=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
trace_e3=# alter table subject_node_table owner to postgres;

# create the node2id table
trace_e3=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
trace_e3=# alter table node2id owner to postgres;
trace_e3=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```

## FiveDirections E3
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database fivedirections_e3;

# switch to the created database
postgres=# \connect fivedirections_e3;

# create the event table and grant the privileges to postgres
fivedirections_e3=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
fivedirections_e3=# alter table event_table owner to postgres;
fivedirections_e3=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
fivedirections_e3=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
fivedirections_e3=# alter table file_node_table owner to postgres;

# create the netflow table
fivedirections_e3=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
fivedirections_e3=# alter table netflow_node_table owner to postgres;

# create the subject table
fivedirections_e3=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    cmdline   varchar,
    exec      varchar
);
fivedirections_e3=# alter table subject_node_table owner to postgres;

# create the node2id table
fivedirections_e3=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
fivedirections_e3=# alter table node2id owner to postgres;
fivedirections_e3=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```

## CADETS E5
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database cadets_e5;

# switch to the created database
postgres=# \connect cadets_e5;

# create the event table and grant the privileges to postgres
cadets_e5=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
cadets_e5=# alter table event_table owner to postgres;
cadets_e5=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
cadets_e5=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
cadets_e5=# alter table file_node_table owner to postgres;

# create the netflow table
cadets_e5=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
cadets_e5=# alter table netflow_node_table owner to postgres;

# create the subject table
cadets_e5=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
cadets_e5=# alter table subject_node_table owner to postgres;

# create the node2id table
cadets_e5=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
cadets_e5=# alter table node2id owner to postgres;
cadets_e5=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```


## THEIA E5
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database theia_1_e5;

# switch to the created database
postgres=# \connect theia_1_e5;

# create the event table and grant the privileges to postgres
theia_1_e5=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
theia_1_e5=# alter table event_table owner to postgres;
theia_1_e5=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
theia_1_e5=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
theia_1_e5=# alter table file_node_table owner to postgres;

# create the netflow table
theia_1_e5=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
theia_1_e5=# alter table netflow_node_table owner to postgres;

# create the subject table
theia_1_e5=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
theia_1_e5=# alter table subject_node_table owner to postgres;

# create the node2id table
theia_1_e5=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
theia_1_e5=# alter table node2id owner to postgres;
theia_1_e5=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```


## ClearScope E5
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database tc_e5_clearscope_dataset_db;

# switch to the created database
postgres=# \connect tc_e5_clearscope_dataset_db;

# create the event table and grant the privileges to postgres
tc_e5_clearscope_dataset_db=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
tc_e5_clearscope_dataset_db=# alter table event_table owner to postgres;
tc_e5_clearscope_dataset_db=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
tc_e5_clearscope_dataset_db=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_e5_clearscope_dataset_db=# alter table file_node_table owner to postgres;

# create the netflow table
tc_e5_clearscope_dataset_db=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_e5_clearscope_dataset_db=# alter table netflow_node_table owner to postgres;

# create the subject table
tc_e5_clearscope_dataset_db=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
tc_e5_clearscope_dataset_db=# alter table subject_node_table owner to postgres;

# create the node2id table
tc_e5_clearscope_dataset_db=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
tc_e5_clearscope_dataset_db=# alter table node2id owner to postgres;
tc_e5_clearscope_dataset_db=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```

## TRACE E5
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database trace_2_e5;

# switch to the created database
postgres=# \connect trace_2_e5;

# create the event table and grant the privileges to postgres
trace_2_e5=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
trace_2_e5=# alter table event_table owner to postgres;
trace_2_e5=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
trace_2_e5=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
trace_2_e5=# alter table file_node_table owner to postgres;

# create the netflow table
trace_2_e5=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
trace_2_e5=# alter table netflow_node_table owner to postgres;

# create the subject table
trace_2_e5=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
trace_2_e5=# alter table subject_node_table owner to postgres;

# create the node2id table
trace_2_e5=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
trace_2_e5=# alter table node2id owner to postgres;
trace_2_e5=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```

## FiveDirections E5
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database fivedirections_2_e5;

# switch to the created database
postgres=# \connect fivedirections_2_e5;

# create the event table and grant the privileges to postgres
fivedirections_2_e5=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    event_uuid    varchar not null,
    timestamp_rec bigint,
    _id           serial
);
fivedirections_2_e5=# alter table event_table owner to postgres;
fivedirections_2_e5=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
fivedirections_2_e5=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
fivedirections_2_e5=# alter table file_node_table owner to postgres;

# create the netflow table
fivedirections_2_e5=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
fivedirections_2_e5=# alter table netflow_node_table owner to postgres;

# create the subject table
fivedirections_2_e5=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    cmdline   varchar,
    exec      varchar
);
fivedirections_2_e5=# alter table subject_node_table owner to postgres;

# create the node2id table
fivedirections_2_e5=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
fivedirections_2_e5=# alter table node2id owner to postgres;
fivedirections_2_e5=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```



## OpTC
```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database optc_db;

# switch to the created database
postgres=# \connect optc_db;

# create the event table and grant the privileges to postgres
optc_db=# create table event_table
(
    src_id     varchar,
    src_type   varchar,
    edge_type  varchar,
    dst_id     varchar,
    dst_type   varchar,
    hostname   varchar,
    timestamp  bigint,
    data_label varchar
);
optc_db=# alter table event_table owner to postgres;

# create the node2id table
optc_db=# create table nodeid2msg
(
    node_id varchar,
    msg     varchar
);
optc_db=# alter table nodeid2msg owner to postgres;
```

