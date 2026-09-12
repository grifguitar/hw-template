package me.index.config;

import me.index.config.parameters.enums.*;
import me.index.config.parameters.records.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

public record Config(
        Keyset keyset,
        DataSize dataSize,
        Workload workload,
        WorkloadPerm workloadPerm,
        WorkloadFactor workloadFactor,
        MaxErr maxErr,
        Activation activation,
        LossFunction loss,
        LossParameter lossParameter,
        LearningRate learningRate,
        BatchSize batchSize,
        Epochs epochs,
        HiddenLayers hiddenLayers,
        Seed seed,
        PlotPath plotPath
) {
    private static final Set<String> KEYS = ConcurrentHashMap.newKeySet();

    public static Config read(Properties p) {
        LossFunction loss = LossFunction.parse(p).orElse(LossFunction.DEFAULT);
        LossParameter lossParameter = LossParameter.parse(p).orElseGet(() -> LossParameter.defaultFor(loss));
        lossParameter.checkUsableWith(loss);

        Config config = new Config(
                Keyset.parse(p).orElseThrow(Keyset::except),
                DataSize.parse(p).orElseThrow(DataSize::except),
                Workload.parse(p).orElse(Workload.DEFAULT),
                WorkloadPerm.parse(p).orElse(WorkloadPerm.DEFAULT),
                WorkloadFactor.parse(p).orElse(WorkloadFactor.DEFAULT),
                MaxErr.parse(p).orElse(MaxErr.DEFAULT),
                Activation.parse(p).orElse(Activation.DEFAULT),
                loss,
                lossParameter,
                LearningRate.parse(p).orElseGet(() -> LearningRate.defaultFor(loss)),
                BatchSize.parse(p).orElse(BatchSize.DEFAULT),
                Epochs.parse(p).orElse(Epochs.DEFAULT),
                HiddenLayers.parse(p).orElse(HiddenLayers.DEFAULT),
                Seed.parse(p).orElse(Seed.DEFAULT),
                PlotPath.parse(p).orElse(PlotPath.DEFAULT)
        );

        rejectUnknownKeys(p);
        return config;
    }

    public static Config load(Path file) throws IOException {
        Properties props = new Properties();
        try (InputStream in = Files.newInputStream(file)) {
            props.load(in);
        } catch (IOException e) {
            throw new IOException("cannot read config file " + file.toAbsolutePath(), e);
        }
        try {
            return read(props);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("invalid config " + file.toAbsolutePath() + ": " + e.getMessage(), e);
        }
    }

    public static <E extends Enum<E>> Optional<E> parseEnum(Properties p, String key, E[] allowed) {
        KEYS.add(key);
        String value = p.getProperty(key);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (E candidate : allowed) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + key + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(allowed));
    }

    public static <T> Optional<T> parseRecord(Properties p, String key, String expected, Function<String, T> parse) {
        KEYS.add(key);
        String value = p.getProperty(key);
        if (value == null) {
            return Optional.empty();
        }
        String text = value.trim();
        try {
            return Optional.of(parse.apply(text));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "property '" + key + "': expected " + expected + ", got '" + text + "'", e);
        }
    }

    private static void rejectUnknownKeys(Properties p) {
        List<String> unknown = new ArrayList<>();
        for (String key : p.stringPropertyNames()) {
            if (!KEYS.contains(key)) {
                unknown.add(key);
            }
        }
        if (!unknown.isEmpty()) {
            List<String> known = new ArrayList<>(KEYS);
            unknown.sort(null);
            known.sort(null);
            throw new IllegalArgumentException("unknown propert" + (unknown.size() == 1 ? "y " : "ies ")
                    + unknown + "; known properties: " + known);
        }
    }
}
