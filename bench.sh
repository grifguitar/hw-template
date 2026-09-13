#!/usr/bin/env bash
#
# Benchmark of learned (neural-network) indexes against binary search over a sorted long[].
#
# Pipeline: build the jar and run all unit tests -> prepare keys, queries and trained models for every
# dataset and size -> time every index in a fresh JVM per configuration and fork -> aggregate and plot.
# Measuring JVMs run Epsilon GC with a fixed, pre-touched heap sized to the data; each JVM verifies every
# answer, checks a checksum after every timed pass and records the bytes allocated while timing.
#
# Output in $OUT:
#   results.csv       one row per measuring JVM
#   summary.md        tables (median over forks, speedup over binary search), model table, warnings
#   summary.csv       the same aggregates as CSV
#   charts/*.pdf      lookup time, speedup and p99 per scenario; model time and search window per dataset
#   environment.txt   hardware, kernel, JVM and benchmark settings
#   runs.log          output of every measuring JVM; failures.txt lists the runs that failed
#
# Usage: [VARIABLE=value ...] ./bench.sh          ./bench.sh --help
#
# Settings are environment variables (defaults in brackets):
#   PROFILE          default | smoke (tiny configuration that only checks the pipeline)       [default]
#   DATASETS         keysets; SOSD keysets need DATA_DIR       [_gauss_int64 _log_norm_int64 and, if DATA_DIR
#                    is set, every SOSD file found there]
#   SIZES            _1e4 _1e5 _1e6 _1e7 _max (_max = up to 2e8 keys, 1.6 GB per keyset)    [_1e5 _1e6 _1e7]
#   WORKLOADS        _uniform _zipf _x_y_90 _x_y_99 _x_y_9999                               [_uniform _zipf]
#   ORDERS           random sorted                                                          [random sorted]
#   BASELINES        binary branchless                                                      [binary branchless]
#   NETS             <activation>:<hidden sizes joined by x, or none>, e.g. _relu:4x4
#                    [_relu:none _relu:4x4 _relu:8 _relu:16x16 _leaky_relu:4x4 _softsign:4x4 _tanh:4x4 _sigmoid:4x4]
#   QUERIES          lookups per queries file                                               [1000000]
#   FORKS            fresh JVMs per configuration                                           [3]
#   RUNS             timed passes over all queries per JVM                                  [10]
#   WARMUP_MIN       minimum warm-up passes                                                 [5]
#   WARMUP_MAX       maximum warm-up passes                                                 [30]
#   WARMUP_CV        warm-up ends when the last 5 passes vary less than this                [0.02]
#   BLOCK            queries per timed block (tail latency)                                 [1024]
#   EPOCHS           training epochs                                                        [100]
#   TRAIN_SAMPLE     keys in the training sample (spread evenly by rank)                    [100000]
#   BATCH            mini-batch size                                                        [32]
#   LOSS             _squared _absolute _huber _log_cosh _power                             [_squared]
#   LR               learning rate; empty = default of the loss                             []
#   SEED             seed of keys, queries and training                                     [42]
#   DATA_DIR         directory with SOSD files                                              []
#   WORK             prepared data                                                          [work]
#   OUT              results directory, must not hold a results.csv yet                     [results/<timestamp>]
#   PREP_HEAP        heap of the preparing JVM                                              [8g]
#   HEAP_MARGIN_MB   heap of a measuring JVM on top of keys + queries                       [512]
#   CPUS             pin measuring JVMs with taskset -c $CPUS                               []
#   NUMA_NODE        bind measuring JVMs with numactl to this node                          []
#   EXTRA_JVM_FLAGS  additional flags for measuring JVMs                                    []
#   JAVA             java executable of a JDK 26+                                           [java]
#   SKIP_BUILD=1     reuse target/*-jar-with-dependencies.jar
#   SKIP_TESTS=1     build without running the unit tests
#   SKIP_PREPARE=1   reuse $WORK of an earlier run (the nets are taken from its models.txt)

set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    exit 0
fi
if [ $# -gt 0 ]; then
    echo "bench.sh takes no arguments; configure it with environment variables (see ./bench.sh --help)" >&2
    exit 2
fi

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die() { log "error: $*"; exit 1; }
csv() { local IFS=,; echo "$*"; }
count() { echo $#; }

PROFILE=${PROFILE:-default}
case "$PROFILE" in
    default) ;;
    smoke)
        : "${DATASETS:=_gauss_int64}"
        : "${SIZES:=_1e4}"
        : "${WORKLOADS:=_uniform}"
        : "${ORDERS:=random}"
        : "${NETS=_relu:none _relu:4x4}"
        : "${QUERIES:=20000}"
        : "${FORKS:=1}"
        : "${RUNS:=3}"
        : "${WARMUP_MIN:=5}"
        : "${WARMUP_MAX:=10}"
        : "${EPOCHS:=2}"
        : "${TRAIN_SAMPLE:=2000}"
        : "${PREP_HEAP:=1g}"
        : "${HEAP_MARGIN_MB:=256}"
        ;;
    *) die "unknown PROFILE '$PROFILE' (expected default or smoke)" ;;
esac

if [ -z "${DATASETS:-}" ] && [ -n "${DATA_DIR:-}" ]; then
    DATASETS="_gauss_int64 _log_norm_int64"
    for f in wiki_ts_200M_uint64 books_200M_uint32 books_800M_uint64 osm_cellids_800M_uint64 fb_200M_uint64; do
        if [ -f "$DATA_DIR/$f" ]; then
            DATASETS="$DATASETS _$f"
        fi
    done
fi
: "${DATASETS:=_gauss_int64 _log_norm_int64}"
: "${SIZES:=_1e5 _1e6 _1e7}"
: "${WORKLOADS:=_uniform _zipf}"
: "${ORDERS:=random sorted}"
: "${BASELINES=binary branchless}"
: "${NETS=_relu:none _relu:4x4 _relu:8 _relu:16x16 _leaky_relu:4x4 _softsign:4x4 _tanh:4x4 _sigmoid:4x4}"
: "${QUERIES:=1000000}"
: "${FORKS:=3}"
: "${RUNS:=10}"
: "${WARMUP_MIN:=5}"
: "${WARMUP_MAX:=30}"
: "${WARMUP_CV:=0.02}"
: "${BLOCK:=1024}"
: "${EPOCHS:=100}"
: "${TRAIN_SAMPLE:=100000}"
: "${BATCH:=32}"
: "${LOSS:=_squared}"
: "${LR=}"
: "${SEED:=42}"
: "${DATA_DIR=}"
: "${WORK:=work}"
: "${OUT:=results/$(date '+%Y%m%d-%H%M%S')}"
: "${PREP_HEAP:=8g}"
: "${HEAP_MARGIN_MB:=512}"
: "${CPUS=}"
: "${NUMA_NODE=}"
: "${EXTRA_JVM_FLAGS=}"
: "${JAVA:=java}"

MODELS_LIST=models.txt

for v in QUERIES FORKS RUNS WARMUP_MIN WARMUP_MAX BLOCK EPOCHS TRAIN_SAMPLE BATCH SEED HEAP_MARGIN_MB; do
    case "${!v}" in
        '' | *[!0-9]*) die "$v must be a non-negative integer, got '${!v}'" ;;
    esac
done
[ "$FORKS" -ge 1 ] || die "FORKS must be at least 1"
for v in DATASETS SIZES WORKLOADS ORDERS; do
    [ "$(count ${!v})" -gt 0 ] || die "$v must not be empty"
done
[ "$(count $BASELINES $NETS)" -gt 0 ] || die "BASELINES and NETS are both empty: nothing to measure"

command -v "$JAVA" >/dev/null 2>&1 || die "java not found; set JAVA=/path/to/jdk-26/bin/java"
java_version=$("$JAVA" -XshowSettings:properties -version 2>&1 | awk -F'= ' '/java.specification.version/ { print $2; exit }')
case "$java_version" in
    '' | *[!0-9]*) die "a JDK 26+ is required, $JAVA reports '$java_version'" ;;
esac
[ "$java_version" -ge 26 ] || die "a JDK 26+ is required, $JAVA is $java_version"

MEASURE_FLAGS=(-XX:+UnlockExperimentalVMOptions -XX:+UseEpsilonGC -XX:+AlwaysPreTouch -XX:-UsePerfData)
THP=/sys/kernel/mm/transparent_hugepage/enabled
if [ "$(uname -s)" = Linux ] && [ -r "$THP" ] && ! grep -q '\[never\]' "$THP"; then
    MEASURE_FLAGS+=(-XX:+UseTransparentHugePages)
fi
if [ -n "$EXTRA_JVM_FLAGS" ]; then
    read -r -a extra_flags <<< "$EXTRA_JVM_FLAGS"
    MEASURE_FLAGS+=("${extra_flags[@]}")
fi
"$JAVA" "${MEASURE_FLAGS[@]}" -Xms64m -Xmx64m -version >/dev/null 2>&1 \
    || die "$JAVA rejects the measuring JVM flags: ${MEASURE_FLAGS[*]}"

PIN=()
if [ -n "$NUMA_NODE" ]; then
    command -v numactl >/dev/null 2>&1 || die "NUMA_NODE is set but numactl is not installed"
    PIN+=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
fi
if [ -n "$CPUS" ]; then
    command -v taskset >/dev/null 2>&1 || die "CPUS is set but taskset is not installed"
    PIN+=(taskset -c "$CPUS")
fi

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    if [ "${SKIP_TESTS:-0}" = 1 ]; then
        log "building the jar without tests"
        ./mvnw -B -q clean package -DskipTests
    else
        log "building the jar and running the unit tests"
        ./mvnw -B -q clean package
    fi
fi
JAR=$(ls target/*-jar-with-dependencies.jar 2>/dev/null | head -n 1 || true)
[ -n "$JAR" ] || die "target/*-jar-with-dependencies.jar is missing; run without SKIP_BUILD=1"

mkdir -p "$OUT" "$WORK"
RESULTS="$OUT/results.csv"
RUNS_LOG="$OUT/runs.log"
FAILURES="$OUT/failures.txt"
[ ! -e "$RESULTS" ] || die "$RESULTS already exists; point OUT to a new directory"

{
    echo "date: $(date)"
    echo "host: $(uname -a)"
    if git rev-parse HEAD >/dev/null 2>&1; then
        echo "git: $(git rev-parse HEAD)$([ -z "$(git status -s 2>/dev/null)" ] || echo ' (uncommitted changes)')"
    fi
    echo "java: $JAVA"
    "$JAVA" -version 2>&1
    echo "measuring JVM flags: ${MEASURE_FLAGS[*]}"
    echo "pinning: ${PIN[*]:-none}"
    for f in "$THP" /sys/kernel/mm/transparent_hugepage/defrag \
        /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
        /sys/devices/system/cpu/intel_pstate/no_turbo /sys/devices/system/cpu/cpufreq/boost; do
        if [ -r "$f" ]; then
            echo "$f: $(cat "$f")"
        fi
    done
    if command -v lscpu >/dev/null 2>&1; then lscpu; fi
    if command -v free >/dev/null 2>&1; then free -m; fi
    echo
    echo "settings:"
    for v in PROFILE DATASETS SIZES WORKLOADS ORDERS BASELINES NETS QUERIES FORKS RUNS WARMUP_MIN WARMUP_MAX \
        WARMUP_CV BLOCK EPOCHS TRAIN_SAMPLE BATCH LOSS LR SEED DATA_DIR WORK OUT PREP_HEAP HEAP_MARGIN_MB CPUS \
        NUMA_NODE EXTRA_JVM_FLAGS SKIP_BUILD SKIP_TESTS SKIP_PREPARE; do
        echo "  $v=${!v:-}"
    done
} > "$OUT/environment.txt" 2>&1

if [ "${SKIP_PREPARE:-0}" != 1 ]; then
    for ds in $DATASETS; do
        for size in $SIZES; do
            log "preparing $ds $size"
            prep_args=(--work "$WORK" --dataset "$ds" --size "$size" --seed "$SEED" --queries "$QUERIES"
                --workloads "$(csv $WORKLOADS)" --orders "$(csv $ORDERS)" --nets "$(csv $NETS)"
                --epochs "$EPOCHS" --train-sample "$TRAIN_SAMPLE" --batch "$BATCH" --loss "$LOSS")
            if [ -n "$DATA_DIR" ]; then
                prep_args+=(--data-dir "$DATA_DIR")
            fi
            if [ -n "$LR" ]; then
                prep_args+=(--lr "$LR")
            fi
            "$JAVA" -Xmx"$PREP_HEAP" -cp "$JAR" me.index.bench.BenchMain prepare "${prep_args[@]}" </dev/null \
                | tee -a "$OUT/prepare.log"
        done
    done
fi

indexes_of() {
    local labels="$BASELINES"
    if [ -f "$1/$MODELS_LIST" ]; then
        labels="$labels $(tr '\n' ' ' < "$1/$MODELS_LIST")"
    fi
    echo $labels
}

heap_mb() {
    local bytes=$(( $(wc -c < "$1") + $(wc -c < "$2") ))
    echo $(( bytes / 1048576 + bytes / 8 / 1048576 + HEAP_MARGIN_MB ))
}

total=0
for ds in $DATASETS; do
    for size in $SIZES; do
        dir="$WORK/$ds/$size"
        [ -f "$dir/keys.bin" ] || die "$dir/keys.bin is missing; run without SKIP_PREPARE=1"
        for wl in $WORKLOADS; do
            for ord in $ORDERS; do
                [ -f "$dir/queries${wl}_${ord}.bin" ] \
                    || die "$dir/queries${wl}_${ord}.bin is missing; run without SKIP_PREPARE=1"
            done
        done
        for idx in $(indexes_of "$dir"); do
            case "$idx" in
                nn_*) [ -f "$dir/$idx.model" ] || die "$dir/$idx.model is missing; run without SKIP_PREPARE=1" ;;
            esac
        done
        total=$(( total + $(count $(indexes_of "$dir")) * $(count $WORKLOADS) * $(count $ORDERS) ))
    done
done
total=$(( total * FORKS ))
log "measuring $total configuration(s), one JVM each; results go to $RESULTS"

done_runs=0
failed=0
started=$(date +%s)
for fork in $(seq 1 "$FORKS"); do
    for ds in $DATASETS; do
        for size in $SIZES; do
            dir="$WORK/$ds/$size"
            for wl in $WORKLOADS; do
                for ord in $ORDERS; do
                    queries="$dir/queries${wl}_${ord}.bin"
                    heap=$(heap_mb "$dir/keys.bin" "$queries")
                    for idx in $(indexes_of "$dir"); do
                        done_runs=$(( done_runs + 1 ))
                        log "[$done_runs/$total] fork $fork: $ds $size $wl $ord $idx (heap ${heap}m)"
                        run_args=(--keys "$dir/keys.bin" --queries "$queries" --index "$idx"
                            --dataset "$ds" --size "$size" --workload "$wl" --order "$ord" --fork "$fork"
                            --runs "$RUNS" --warmup-min "$WARMUP_MIN" --warmup-max "$WARMUP_MAX"
                            --warmup-cv "$WARMUP_CV" --block "$BLOCK" --out "$RESULTS")
                        case "$idx" in
                            nn_*) run_args+=(--model "$dir/$idx.model") ;;
                        esac
                        if ! ${PIN[@]+"${PIN[@]}"} "$JAVA" "${MEASURE_FLAGS[@]}" -Xms"${heap}m" -Xmx"${heap}m" \
                            -cp "$JAR" me.index.bench.BenchMain run "${run_args[@]}" </dev/null 2>&1 \
                            | tee -a "$RUNS_LOG"; then
                            echo "fork=$fork dataset=$ds size=$size workload=$wl order=$ord index=$idx" >> "$FAILURES"
                            failed=$(( failed + 1 ))
                            log "measurement failed, continuing (details in $RUNS_LOG)"
                        fi
                    done
                done
            done
        done
    done
done
log "measured in $(( $(date +%s) - started )) s"

if [ -s "$RESULTS" ]; then
    "$JAVA" -cp "$JAR" me.index.bench.BenchMain report --results "$RESULTS" --out "$OUT" </dev/null
else
    log "no measurement succeeded, nothing to report"
fi
log "results: $RESULTS"
log "summary: $OUT/summary.md"
log "charts:  $OUT/charts/"
if [ "$failed" -gt 0 ]; then
    die "$failed measurement(s) failed, see $FAILURES"
fi
