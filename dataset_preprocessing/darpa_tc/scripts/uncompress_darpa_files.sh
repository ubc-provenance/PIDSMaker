#!/bin/bash

RAW_DIR=$1
cd $RAW_DIR

echo Uncompressing...

for file in *.tar.gz; do tar -xzf "$file"; done
rm *tar.gz

for file in *.gz; do gzip -d "$file"; done
rm *.gz

echo Uncompressing done...

cp ./TCCDMDatum.avsc ta3-java-consumer/ta3-serialization-schema/avro/

cd ta3-java-consumer
cd ta3-serialization-schema

echo Running Maven...

mvn clean exec:java
mvn install

cd ../tc-bbn-avro
mvn clean install

cd ../tc-bbn-kafka
mvn assembly:assembly

echo Converting bin to json...

# This converts from bin to json (takes few minutes)
for file in $RAW_DIR/*bin*; do ./json_consumer.sh "$file"; done

# Move output json files to raw dir
mv *.json* $RAW_DIR
