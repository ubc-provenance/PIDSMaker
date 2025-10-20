import os
import hashlib
import psycopg2
from datetime import datetime
import time
import pytz
from time import mktime
from parser_config import (
    DATABASE_DEFAULT_CONFIG,
    DATASET_DEFAULT_CONFIG
)

def get_all_filelist(filepath):
    '''get all file paths under the given filepath recursively'''
    file_paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            full_path = os.path.join(root, file)
            abs_path = os.path.abspath(full_path)
            file_paths.append(abs_path)
    return file_paths

def stringtomd5(originstr):
    '''convert a string to its md5 hash value'''
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def init_database_connection(dataset):
    '''initialize the database connection and return the cursor and connection object'''
    database_config = DATASET_DEFAULT_CONFIG[dataset]
    database = database_config['database']

    connect = psycopg2.connect(database=database,
                               host=DATABASE_DEFAULT_CONFIG['host'],
                               user=DATABASE_DEFAULT_CONFIG['user'],
                               password=DATABASE_DEFAULT_CONFIG['password'],
                               port=DATABASE_DEFAULT_CONFIG['port']
                               )
    cur = connect.cursor()
    return cur, connect

def OPTC_datetime_to_timestamp_US(date):
    '''convert OPTC datetime string to timestamp in nanoseconds'''
    date=date.replace('-04:00','')
    if '.' in date:
        date,ms=date.split('.')
    else:
        ms=0
    tz = pytz.timezone('Etc/GMT+4')
    timeArray = time.strptime(date, "%Y-%m-%dT%H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp=timestamp.timestamp()
    timeStamp = timestamp*1000+int(ms)
    return int(timeStamp) * 1000000




