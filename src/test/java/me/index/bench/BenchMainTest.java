package me.index.bench;

import me.index.bench.report.Report;
import me.index.bench.report.ResultRow;
import me.index.bench.report.ResultsCsv;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchMainTest {
    private record Out(int code, String out, String err) {
    }

    private static Out exec(String... args) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        int code = BenchMain.execute(args, new PrintStream(out, true, StandardCharsets.UTF_8),
                new PrintStream(err, true, StandardCharsets.UTF_8));
        return new Out(code, out.toString(StandardCharsets.UTF_8), err.toString(StandardCharsets.UTF_8));
    }

    private static Out prepare(Path work) {
        return exec("prepare", "--work", work.toString(), "--dataset", "_gauss_int64", "--size", "_1e4",
                "--queries", "4000", "--workloads", "_uniform,_x_y_99", "--orders", "random,sorted",
                "--nets", "_relu:4x4,_tanh:8", "--epochs", "2", "--train-sample", "1000", "--seed", "7");
    }

    private static List<String> runArgs(Path dir, String workload, String order, String index, Path results) {
        List<String> args = new ArrayList<>(List.of("run",
                "--keys", dir.resolve("keys.bin").toString(),
                "--queries", dir.resolve("queries" + workload + "_" + order + ".bin").toString(),
                "--index", index, "--out", results.toString(),
                "--dataset", "_gauss_int64", "--size", "_1e4", "--workload", workload, "--order", order,
                "--runs", "2", "--warmup-min", "1", "--warmup-max", "2", "--block", "500"));
        if (index.startsWith("nn_")) {
            args.add("--model");
            args.add(dir.resolve(index + ".model").toString());
        }
        return args;
    }

    @Test
    void prepareRunAndReportWorkEndToEnd(@TempDir Path tmp) throws IOException {
        Path work = tmp.resolve("work");
        Path out = tmp.resolve("out");
        Path results = out.resolve("results.csv");

        Out prep = prepare(work);
        assertEquals(0, prep.code(), prep.err());
        assertTrue(prep.out().contains("nn_tanh_8"), prep.out());
        Path dir = work.resolve("_gauss_int64").resolve("_1e4");

        List<String> indexes = new ArrayList<>(BenchMain.BASELINES);
        indexes.addAll(Files.readAllLines(dir.resolve("models.txt")));
        assertEquals(List.of("binary", "branchless", "nn_relu_4x4", "nn_tanh_8"), indexes);
        for (String workload : List.of("_uniform", "_x_y_99")) {
            for (String order : List.of("random", "sorted")) {
                for (String index : indexes) {
                    Out r = exec(runArgs(dir, workload, order, index, results).toArray(String[]::new));
                    assertEquals(0, r.code(), r.err());
                    assertTrue(r.out().contains(index), r.out());
                }
            }
        }

        List<ResultRow> rows = ResultsCsv.read(results);
        assertEquals(16, rows.size());
        for (ResultRow row : rows) {
            assertTrue(row.nsMedian() >= 0 && Double.isFinite(row.nsMedian()), row.toString());
            assertTrue(row.allocBytes() <= 0, row.index() + " allocated " + row.allocBytes() + " bytes");
            assertEquals(4000, row.queries());
            assertEquals(2, row.runs());
            if (row.kind().equals(ResultRow.NET)) {
                assertTrue(row.window() >= 1 && row.window() <= row.n(), row.toString());
                assertTrue(Double.isFinite(row.modelNs()), row.toString());
                assertTrue(row.params() > 0);
            } else {
                assertEquals("-", row.activation());
                assertEquals(row.n(), row.window());
            }
        }
        assertEquals(rows.getFirst().checksum(), rows.get(1).checksum(), "same queries must give the same checksum");

        Out report = exec("report", "--results", results.toString(), "--out", out.toString());
        assertEquals(0, report.code(), report.err());
        assertTrue(Files.readString(out.resolve(Report.SUMMARY_MD)).contains("nn_tanh_8"));
        assertTrue(Files.exists(out.resolve(Report.SUMMARY_CSV)));
        for (String chart : List.of("lookup_gauss_int64_uniform_random.pdf", "speedup_gauss_int64_x_y_99_sorted.pdf",
                "p99_gauss_int64_uniform_sorted.pdf", "model_gauss_int64.pdf", "window_gauss_int64.pdf")) {
            Path file = out.resolve(Report.CHARTS).resolve(chart);
            assertTrue(Files.exists(file), chart);
            assertTrue(Files.readString(file, StandardCharsets.ISO_8859_1).startsWith("%PDF-1.4"), chart);
        }
    }

    @Test
    void usageErrorsExitWithTwoAndPrintTheUsage() {
        List<String[]> bad = List.of(
                new String[0],
                new String[]{"frobnicate"},
                new String[]{"report", "--results"},
                new String[]{"report", "--results", "a", "--oops", "b"},
                new String[]{"run", "--keys", "k"},
                new String[]{"prepare", "--work", "w", "--dataset", "_nope", "--size", "_1e4"},
                new String[]{"prepare", "--work", "w", "--dataset", "_gauss_int64", "--size", "_1e4", "--nets", "relu"},
                new String[]{"prepare", "--work", "w", "--dataset", "_gauss_int64", "--size", "_1e4", "--queries", "0"});
        for (String[] args : bad) {
            Out r = exec(args);
            assertEquals(2, r.code(), String.join(" ", args) + ": " + r.err());
            assertTrue(r.err().contains("usage:"), r.err());
        }
    }

    @Test
    void runValidatesTheIndexBeforeTouchingFiles(@TempDir Path tmp) {
        Path results = tmp.resolve("r.csv");
        Path dir = tmp.resolve("missing");
        List<String> baselineWithModel = runArgs(dir, "_uniform", "random", "binary", results);
        baselineWithModel.addAll(List.of("--model", "m"));
        assertEquals(2, exec(baselineWithModel.toArray(String[]::new)).code());

        List<String> netWithoutModel = runArgs(dir, "_uniform", "random", "binary", results);
        netWithoutModel.set(netWithoutModel.indexOf("binary"), "nn_relu_4x4");
        assertEquals(2, exec(netWithoutModel.toArray(String[]::new)).code());

        List<String> unknown = runArgs(dir, "_uniform", "random", "binary", results);
        unknown.set(unknown.indexOf("binary"), "btree");
        assertEquals(2, exec(unknown.toArray(String[]::new)).code());

        Out missing = exec(runArgs(dir, "_uniform", "random", "binary", results).toArray(String[]::new));
        assertEquals(1, missing.code());
        assertTrue(missing.err().startsWith("error:"), missing.err());
        assertTrue(!Files.exists(results), "a failed run must not write a result");
    }

    @Test
    void mismatchedInputsFailWithOne(@TempDir Path tmp) throws IOException {
        Path work = tmp.resolve("work");
        assertEquals(0, prepare(work).code());
        Path dir = work.resolve("_gauss_int64").resolve("_1e4");
        Path results = tmp.resolve("r.csv");

        List<String> wrongModel = runArgs(dir, "_uniform", "random", "nn_relu_4x4", results);
        wrongModel.set(wrongModel.size() - 1, dir.resolve("nn_tanh_8.model").toString());
        Out r = exec(wrongModel.toArray(String[]::new));
        assertEquals(1, r.code());
        assertTrue(r.err().contains("nn_tanh_8"), r.err());

        Path otherWork = tmp.resolve("other");
        assertEquals(0, exec("prepare", "--work", otherWork.toString(), "--dataset", "_log_norm_int64", "--size",
                "_1e4", "--queries", "100").code());
        List<String> foreignQueries = runArgs(dir, "_uniform", "random", "binary", results);
        foreignQueries.set(foreignQueries.indexOf("--queries") + 1,
                otherWork.resolve("_log_norm_int64/_1e4/queries_uniform_random.bin").toString());
        Out f = exec(foreignQueries.toArray(String[]::new));
        assertEquals(1, f.code());
        assertTrue(f.err().contains("does not belong"), f.err());

        assertEquals(1, exec("prepare", "--work", work.toString(), "--dataset", "_gauss_int32", "--size", "_max").code());
        assertEquals(1, exec("report", "--results", tmp.resolve("absent.csv").toString(), "--out", tmp.toString()).code());
        assertTrue(!Files.exists(results));
    }

    @Test
    void helpPrintsTheUsage() {
        Out r = exec("--help");
        assertEquals(0, r.code());
        assertTrue(r.out().contains("BenchMain prepare"));
    }
}
