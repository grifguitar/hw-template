package me.index.bench.report;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * One measuring JVM: per-JVM statistics over its timed passes. All times are nanoseconds per lookup.
 */
public record ResultRow(
        String dataset, String size, int n, String workload, String order, int queries,
        String index, String kind, String activation, String layers, int params,
        int fork, int runs, int warmupPasses,
        double nsMedian, double nsMin, double nsMax, double nsMean, double nsStdev,
        double blockP50, double blockP90, double blockP99, double blockMax,
        double modelNs, int errLo, int errHi, long window, double meanAbsErr,
        long indexBytes, long allocBytes, long jitMs, long checksum, String runNs) {
    public static final String BASELINE = "baseline";
    public static final String NET = "net";

    public ResultRow {
        for (String text : new String[]{dataset, size, workload, order, index, kind, activation, layers, runNs}) {
            Objects.requireNonNull(text);
            if (text.isEmpty() || text.indexOf(',') >= 0 || text.indexOf('\n') >= 0 || text.indexOf('\r') >= 0) {
                throw new IllegalArgumentException("CSV field must be non-empty without commas or newlines: '"
                        + text + "'");
            }
        }
    }

    public static String joinRuns(double[] runs) {
        return Arrays.stream(runs).mapToObj(v -> String.format(Locale.ROOT, "%.3f", v))
                .collect(Collectors.joining(";"));
    }
}
