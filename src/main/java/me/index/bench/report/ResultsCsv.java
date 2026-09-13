package me.index.bench.report;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ResultsCsv {
    public static final List<String> COLUMNS = List.of(
            "dataset", "size", "n", "workload", "order", "queries",
            "index", "kind", "activation", "layers", "params",
            "fork", "runs", "warmup_passes",
            "ns_median", "ns_min", "ns_max", "ns_mean", "ns_stdev",
            "block_p50", "block_p90", "block_p99", "block_max",
            "model_ns", "err_lo", "err_hi", "window", "mean_abs_err",
            "index_bytes", "alloc_bytes", "jit_ms", "checksum", "run_ns");

    private ResultsCsv() {
    }

    public static void append(Path file, ResultRow row) throws IOException {
        Path parent = file.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        boolean header = !Files.exists(file) || Files.size(file) == 0;
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (header) {
                w.write(String.join(",", COLUMNS));
                w.write('\n');
            }
            w.write(format(row));
            w.write('\n');
        }
    }

    static String format(ResultRow r) {
        return String.join(",",
                r.dataset(), r.size(), Integer.toString(r.n()), r.workload(), r.order(), Integer.toString(r.queries()),
                r.index(), r.kind(), r.activation(), r.layers(), Integer.toString(r.params()),
                Integer.toString(r.fork()), Integer.toString(r.runs()), Integer.toString(r.warmupPasses()),
                Double.toString(r.nsMedian()), Double.toString(r.nsMin()), Double.toString(r.nsMax()),
                Double.toString(r.nsMean()), Double.toString(r.nsStdev()),
                Double.toString(r.blockP50()), Double.toString(r.blockP90()), Double.toString(r.blockP99()),
                Double.toString(r.blockMax()),
                Double.toString(r.modelNs()), Integer.toString(r.errLo()), Integer.toString(r.errHi()),
                Long.toString(r.window()), Double.toString(r.meanAbsErr()),
                Long.toString(r.indexBytes()), Long.toString(r.allocBytes()), Long.toString(r.jitMs()),
                Long.toString(r.checksum()), r.runNs());
    }

    public static List<ResultRow> read(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("results file " + file.toAbsolutePath() + " is empty");
        }
        String[] header = lines.get(0).split(",", -1);
        Map<String, Integer> at = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            at.put(header[i].trim(), i);
        }
        for (String column : COLUMNS) {
            if (!at.containsKey(column)) {
                throw new IOException("results file " + file.toAbsolutePath() + " has no column '" + column + "'");
            }
        }
        List<ResultRow> rows = new ArrayList<>();
        for (int line = 1; line < lines.size(); line++) {
            if (lines.get(line).isBlank()) {
                continue;
            }
            String[] f = lines.get(line).split(",", -1);
            if (f.length != header.length) {
                throw new IOException(file.toAbsolutePath() + ":" + (line + 1) + ": expected " + header.length
                        + " fields, got " + f.length);
            }
            try {
                rows.add(parse(new Fields(f, at)));
            } catch (IllegalArgumentException e) {
                throw new IOException(file.toAbsolutePath() + ":" + (line + 1) + ": " + e.getMessage(), e);
            }
        }
        return rows;
    }

    private record Fields(String[] values, Map<String, Integer> at) {
        String s(String column) {
            return values[at.get(column)];
        }

        int i(String column) {
            return Integer.parseInt(s(column));
        }

        long l(String column) {
            return Long.parseLong(s(column));
        }

        double d(String column) {
            return Double.parseDouble(s(column));
        }
    }

    private static ResultRow parse(Fields f) {
        return new ResultRow(
                f.s("dataset"), f.s("size"), f.i("n"), f.s("workload"), f.s("order"), f.i("queries"),
                f.s("index"), f.s("kind"), f.s("activation"), f.s("layers"), f.i("params"),
                f.i("fork"), f.i("runs"), f.i("warmup_passes"),
                f.d("ns_median"), f.d("ns_min"), f.d("ns_max"), f.d("ns_mean"), f.d("ns_stdev"),
                f.d("block_p50"), f.d("block_p90"), f.d("block_p99"), f.d("block_max"),
                f.d("model_ns"), f.i("err_lo"), f.i("err_hi"), f.l("window"), f.d("mean_abs_err"),
                f.l("index_bytes"), f.l("alloc_bytes"), f.l("jit_ms"), f.l("checksum"), f.s("run_ns"));
    }
}
