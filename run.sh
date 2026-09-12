#!/bin/sh
set -eu
cd "$(dirname "$0")"

JAR=$(ls target/*-jar-with-dependencies.jar 2>/dev/null | head -1 || true)
if [ -z "$JAR" ]; then
    echo "no jar found, building it first..." >&2
    ./mvnw -B -q package -DskipTests
    JAR=$(ls target/*-jar-with-dependencies.jar | head -1)
fi

exec java -jar "$JAR" "$@"
