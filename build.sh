#!/bin/sh
set -eu
cd "$(dirname "$0")"

./mvnw -B clean package "$@"

JAR=$(ls target/*-jar-with-dependencies.jar | head -1)
echo
echo "executable jar: $JAR"
echo "run it with:    java -jar $JAR [config.properties] [sosd-data-dir]"
