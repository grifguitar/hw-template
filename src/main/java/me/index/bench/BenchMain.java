package me.index.bench;

import me.index.bench.data.BinFiles;
import me.index.bench.data.Queries;
import me.index.bench.data.QueryOrder;
import me.index.bench.index.BinarySearchIndex;
import me.index.bench.index.BranchlessIndex;
import me.index.bench.index.Index;
import me.index.bench.index.Kernel;
import me.index.bench.index.NetIndex;
import me.index.bench.index.NetModel;
import me.index.bench.index.NetSpec;
import me.index.bench.index.Trainer;
import me.index.bench.report.Report;
import me.index.bench.report.ResultRow;
import me.index.bench.report.ResultsCsv;
import me.index.config.parameters.enums.DataSize;
import me.index.config.parameters.enums.Keyset;
import me.index.config.parameters.enums.LossFunction;
import me.index.config.parameters.enums.Workload;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class BenchMain {
    static final String USAGE = """
            usage:
              BenchMain prepare --work DIR --dataset KEYSET --size DATA_SIZE [--data-dir DIR] [--seed N]
                                [--queries N] [--workloads LIST] [--orders LIST] [--nets LIST]
                                [--epochs N] [--train-sample N] [--batch N] [--loss LOSS] [--lr X]
              BenchMain run --keys FILE --queries FILE --index NAME [--model FILE] --out CSV
                            --dataset NAME --size NAME --workload NAME --order NAME [--fork N]
                            [--runs N] [--warmup-min N] [--warmup-max N] [--warmup-cv X] [--block N]
              BenchMain report --results CSV --out DIR

              LIST is comma-separated; a net is <activation>:<hidden layers joined by x | none>, e.g. _relu:4x4
              baseline indexes: binary, branchless; nets are named nn_<activation>_<layers> or nn_linear
            """;

    static final List<String> BASELINES = List.of(BinarySearchIndex.NAME, BranchlessIndex.NAME);

    static final Set<String> PREPARE_OPTIONS = Set.of("--work", "--dataset", "--size", "--data-dir", "--seed",
            "--queries", "--workloads", "--orders", "--nets", "--epochs", "--train-sample", "--batch", "--loss", "--lr");
    static final Set<String> RUN_OPTIONS = Set.of("--keys", "--queries", "--index", "--model", "--out",
            "--dataset", "--size", "--workload", "--order", "--fork",
            "--runs", "--warmup-min", "--warmup-max", "--warmup-cv", "--block");
    static final Set<String> REPORT_OPTIONS = Set.of("--results", "--out");

    private BenchMain() {
    }

    static void main(String[] args) {
        int code = execute(args, System.out, System.err);
        if (code != 0) {
            System.exit(code);
        }
    }

    static Set<String> allOptions() {
        Set<String> all = new HashSet<>(PREPARE_OPTIONS);
        all.addAll(RUN_OPTIONS);
        all.addAll(REPORT_OPTIONS);
        return all;
    }

    static int execute(String[] args, PrintStream out, PrintStream err) {
        if (args.length == 0) {
            err.print(USAGE);
            return 2;
        }
        try {
            switch (args[0]) {
                case "prepare" -> prepare(Args.parse(args, 1, PREPARE_OPTIONS), out);
                case "run" -> run(Args.parse(args, 1, RUN_OPTIONS), out);
                case "report" -> report(Args.parse(args, 1, REPORT_OPTIONS), out);
                case "help", "-h", "--help" -> out.print(USAGE);
                default -> throw new Args.UsageException("unknown command '" + args[0] + "'");
            }
            return 0;
        } catch (Args.UsageException e) {
            err.println("error: " + e.getMessage());
            err.print(USAGE);
            return 2;
        } catch (IOException | IllegalArgumentException | IllegalStateException | ArithmeticException e) {
            err.println("error: " + e.getMessage());
            return 1;
        }
    }

    private static void prepare(Args a, PrintStream out) throws IOException {
        Keyset keyset = Args.enumValue("--dataset", a.required("--dataset"), Keyset.values());
        DataSize size = Args.enumValue("--size", a.required("--size"), DataSize.values());
        Path work = a.path("--work");
        List<Workload> workloads = a.list("--workloads", Workload._uniform.name()).stream()
                .map(t -> Args.enumValue("--workloads", t, Workload.values())).toList();
        List<QueryOrder> orders = a.list("--orders", QueryOrder.random.name()).stream()
                .map(t -> Args.enumValue("--orders", t, QueryOrder.values())).toList();
        List<NetSpec> nets;
        try {
            nets = NetSpec.parseList(a.string("--nets", ""));
        } catch (IllegalArgumentException e) {
            throw new Args.UsageException("--nets: " + e.getMessage());
        }
        LossFunction loss = Args.enumValue("--loss", a.string("--loss", LossFunction._squared.name()),
                LossFunction.values());
        long seed = a.longValue("--seed", 42);
        Trainer.Settings training = new Trainer.Settings(
                a.intValue("--epochs", 100, 1),
                a.intValue("--train-sample", 100_000, 2),
                a.intValue("--batch", 32, 1),
                loss,
                a.doubleValue("--lr", Double.NaN),
                seed);
        Prepare.run(new Prepare.Settings(work, keyset, size, Path.of(a.string("--data-dir", "")), seed,
                a.intValue("--queries", 1_000_000, 1), workloads, orders, nets, training), out);
    }

    private static void run(Args a, PrintStream out) throws IOException {
        Path keysFile = a.path("--keys");
        Path queriesFile = a.path("--queries");
        String indexName = a.required("--index");
        Path results = a.path("--out");
        String dataset = a.required("--dataset");
        String size = a.required("--size");
        String workload = a.required("--workload");
        String order = a.required("--order");
        int fork = a.intValue("--fork", 1, 1);
        Measure.Settings settings = new Measure.Settings(
                a.intValue("--runs", 10, 1),
                a.intValue("--warmup-min", 5, 0),
                a.intValue("--warmup-max", 30, 0),
                a.doubleValue("--warmup-cv", 0.02),
                a.intValue("--block", 1024, 1));
        boolean net = NetSpec.isNetLabel(indexName);
        Path modelFile = net ? a.path("--model") : null;
        if (!net && a.has("--model")) {
            throw new Args.UsageException("--model is only valid for nn_* indexes, not for " + indexName);
        }
        if (!net && !BASELINES.contains(indexName)) {
            throw new Args.UsageException("--index: unknown index '" + indexName + "'; allowed values: "
                    + BASELINES + " or nn_*");
        }

        long[] keys = BinFiles.readKeys(keysFile);
        Queries queries = BinFiles.readQueries(queriesFile);
        queries.requireConsistentWith(keys);

        NetModel model = null;
        NetIndex netIndex = null;
        Index index;
        if (net) {
            model = NetModel.read(modelFile);
            if (!model.spec().label().equals(indexName)) {
                throw new IllegalArgumentException("model " + modelFile + " holds " + model.spec().label()
                        + ", not " + indexName);
            }
            netIndex = model.index(keys);
            index = netIndex;
        } else {
            index = baseline(indexName, keys);
        }

        Measure.Result r = Measure.lookups(index, queries, settings);
        double modelNs = netIndex == null ? Double.NaN : Measure.modelNsPerQuery(netIndex, queries, settings);

        double[] runs = r.runNsPerQuery();
        double[] blocks = r.blockNsPerQuery();
        ResultRow row = new ResultRow(dataset, size, keys.length, workload, order, queries.size(),
                index.name(), model == null ? ResultRow.BASELINE : ResultRow.NET,
                model == null ? "-" : model.spec().activation().name(),
                model == null ? "-" : model.spec().layersText(),
                model == null ? 0 : Kernel.parameterCount(model.spec().layerSizes()),
                fork, settings.runs(), r.warmupPasses(),
                Stats.median(runs), Stats.min(runs), Stats.max(runs), Stats.mean(runs), Stats.stdev(runs),
                Stats.percentile(blocks, 50), Stats.percentile(blocks, 90), Stats.percentile(blocks, 99),
                Stats.max(blocks), modelNs,
                model == null ? 0 : model.errLo(), model == null ? 0 : model.errHi(),
                model == null ? keys.length : model.window(), model == null ? Double.NaN : model.meanAbsErr(),
                index.sizeBytes(), r.allocatedBytes(), r.jitMillis(), r.checksum(), ResultRow.joinRuns(runs));
        ResultsCsv.append(results, row);
        out.printf(Locale.ROOT, "%s %s %s %s %s fork %d: %.1f ns/lookup (runs %.1f..%.1f, p99 block %.1f)%s,"
                        + " warm-up %d, alloc %d B, jit %d ms%n",
                dataset, size, workload, order, index.name(), fork, row.nsMedian(), row.nsMin(), row.nsMax(),
                row.blockP99(), model == null ? "" : String.format(Locale.ROOT, ", model %.1f ns, window %d",
                        modelNs, model.window()),
                r.warmupPasses(), r.allocatedBytes(), r.jitMillis());
    }

    static Index baseline(String name, long[] keys) {
        return switch (name) {
            case BinarySearchIndex.NAME -> new BinarySearchIndex(keys);
            case BranchlessIndex.NAME -> new BranchlessIndex(keys);
            default -> throw new IllegalArgumentException("unknown baseline '" + name + "'; allowed values: "
                    + BASELINES);
        };
    }

    private static void report(Args a, PrintStream out) throws IOException {
        Path results = a.path("--results");
        Path dir = a.path("--out");
        List<ResultRow> rows = ResultsCsv.read(results);
        List<Path> charts = Report.write(rows, dir, results.getFileName().toString());
        out.printf(Locale.ROOT, "report: %d measurement(s) -> %s, %s and %d chart(s) in %s%n", rows.size(),
                dir.resolve(Report.SUMMARY_MD), dir.resolve(Report.SUMMARY_CSV), charts.size(),
                dir.resolve(Report.CHARTS));
    }
}
