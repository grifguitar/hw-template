package me.index.bench;

import me.index.Context;
import me.index.bench.data.BinFiles;
import me.index.bench.data.QueryGenerator;
import me.index.bench.data.QueryOrder;
import me.index.bench.index.NetModel;
import me.index.bench.index.NetSpec;
import me.index.bench.index.Trainer;
import me.index.config.Config;
import me.index.config.parameters.enums.DataSize;
import me.index.config.parameters.enums.Keyset;
import me.index.config.parameters.enums.Workload;
import me.index.config.parameters.records.Seed;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * Builds everything a measuring JVM loads for one keyset: {@code keys.bin}, one queries file per workload and
 * order, and one trained model per net, all under {@code <work>/<keyset>/<size>/}.
 */
public final class Prepare {
    public static final String KEYS_FILE = "keys.bin";
    public static final String MODELS_LIST = "models.txt";
    public static final String MODELS_CSV = "models.csv";

    private Prepare() {
    }

    public record Settings(Path work, Keyset keyset, DataSize size, Path dataDir, long seed, int queries,
                           List<Workload> workloads, List<QueryOrder> orders, List<NetSpec> nets,
                           Trainer.Settings training) {
        public Settings {
            Objects.requireNonNull(work, "work");
            Objects.requireNonNull(keyset, "keyset");
            Objects.requireNonNull(size, "size");
            Objects.requireNonNull(dataDir, "dataDir");
            Objects.requireNonNull(training, "training");
            workloads = List.copyOf(workloads);
            orders = List.copyOf(orders);
            nets = List.copyOf(nets);
            if (queries <= 0) {
                throw new IllegalArgumentException("query count must be positive, got " + queries);
            }
            if (workloads.isEmpty() || orders.isEmpty()) {
                throw new IllegalArgumentException("at least one workload and one query order are required");
            }
        }
    }

    public static Path directory(Path work, Keyset keyset, DataSize size) {
        return work.resolve(keyset.name()).resolve(size.name());
    }

    public static String queriesFileName(Workload workload, QueryOrder order) {
        return "queries" + workload.name() + "_" + order.name() + ".bin";
    }

    public static String modelFileName(NetSpec spec) {
        return spec.label() + ".model";
    }

    public static Path run(Settings s, PrintStream log) throws IOException {
        Path dir = directory(s.work(), s.keyset(), s.size());
        Files.createDirectories(dir);

        long[] keys = loadKeys(s);
        BinFiles.writeKeys(dir.resolve(KEYS_FILE), keys);
        log.printf(Locale.ROOT, "%s %s: %d distinct keys -> %s%n", s.keyset(), s.size(), keys.length, dir);

        for (Workload workload : s.workloads()) {
            for (QueryOrder order : s.orders()) {
                BinFiles.writeQueries(dir.resolve(queriesFileName(workload, order)),
                        QueryGenerator.generate(keys, workload, order, s.queries(), querySeed(s.seed(), workload)));
            }
        }
        log.printf(Locale.ROOT, "  %d queries for each of %s x %s%n", s.queries(), s.workloads(), s.orders());

        List<String> labels = new ArrayList<>();
        List<String> csv = new ArrayList<>();
        csv.add("index,activation,layers,params,loss,epochs,train_sample,train_mse,train_seconds,"
                + "err_lo,err_hi,window,mean_abs_err");
        for (NetSpec spec : s.nets()) {
            NetModel model = Trainer.train(keys, spec, s.training());
            model.write(dir.resolve(modelFileName(spec)));
            labels.add(spec.label());
            csv.add(String.join(",", spec.label(), spec.activation().name(), spec.layersText(),
                    Integer.toString(model.kernel().parameters().length), model.loss(),
                    Integer.toString(model.epochs()), Integer.toString(model.trainSample()),
                    Double.toString(model.trainMse()), Double.toString(model.trainSeconds()),
                    Integer.toString(model.errLo()), Integer.toString(model.errHi()),
                    Long.toString(model.window()), Double.toString(model.meanAbsErr())));
            log.printf(Locale.ROOT, "  %-22s trained in %.1f s, MSE %.3e, window %d (errLo %d, errHi %d, mean |err| %.1f)%s%n",
                    spec.label(), model.trainSeconds(), model.trainMse(), model.window(), model.errLo(), model.errHi(),
                    model.meanAbsErr(), Double.isFinite(model.trainMse()) ? "" : "  [diverged]");
        }
        Files.write(dir.resolve(MODELS_LIST), labels);
        Files.write(dir.resolve(MODELS_CSV), csv);
        return dir;
    }

    static long[] loadKeys(Settings s) throws IOException {
        Properties p = new Properties();
        p.setProperty(Keyset.KEY, s.keyset().name());
        p.setProperty(DataSize.KEY, s.size().name());
        p.setProperty(Seed.KEY, Long.toString(s.seed()));
        return new Context(Config.read(p), s.dataDir()).keys;
    }

    static long querySeed(long seed, Workload workload) {
        return seed * 0x9E3779B97F4A7C15L + workload.ordinal() + 1;
    }
}
