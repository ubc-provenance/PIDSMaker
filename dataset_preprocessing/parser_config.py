# This is the config for dataset parsing and database creation

DATA_DIR = '/path/to/raw/data/'

# Database connection configuration
DATABASE_DEFAULT_CONFIG = {
     "host": 'nerds08.cs.ubc.ca',  
     "user": 'postgres',  
     "password": 'psql@systopia',  
     "port": '5432',  
}

DATASET_DEFAULT_CONFIG = {
    'h051': {
        'database': 'optc_051',
    },
    'h201': {
        'database': 'optc_201',
    },
    'h501': {
        'database': 'optc_501',
    },
}

OPTC_hostname_map = {
    'h051': 'SysClient0051',
    'h201': 'SysClient0201',
    'h501': 'SysClient0501',
}

OPTC_reversed_type = {"READ"}

OPTC_node_type_used=[
    'FILE',
    'FLOW',
    'PROCESS',
]

OPTC_rel2id = {
    1: 'OPEN',
    'OPEN': 1,
    2: 'READ',
    'READ': 2,
    3: 'CREATE',
    'CREATE': 3,
    4: 'MESSAGE',
    'MESSAGE': 4,
    5: 'MODIFY',
    'MODIFY': 5,
    6: 'START',
    'START': 6,
    7: 'RENAME',
    'RENAME': 7,
    8: 'DELETE',
    'DELETE': 8,
    9: 'TERMINATE',
    'TERMINATE': 9,
    10: 'WRITE',
    'WRITE': 10}