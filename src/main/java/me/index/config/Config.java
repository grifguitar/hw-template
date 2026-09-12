package me.index.config;

import me.index.config.parameters.enums.*;
import me.index.config.parameters.records.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Set;

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
    public Config {
        requireSet(keyset, Keyset.KEY);
        requireSet(dataSize, DataSize.KEY);
        requireSet(workload, Workload.KEY);
        requireSet(workloadPerm, WorkloadPerm.KEY);
        requireSet(workloadFactor, WorkloadFactor.KEY);
        requireSet(maxErr, MaxErr.KEY);
        requireSet(activation, Activation.KEY);
        requireSet(loss, LossFunction.KEY);
        requireSet(lossParameter, LossParameter.KEY);
        requireSet(learningRate, LearningRate.KEY);
        requireSet(batchSize, BatchSize.KEY);
        requireSet(epochs, Epochs.KEY);
        requireSet(hiddenLayers, HiddenLayers.KEY);
        requireSet(seed, Seed.KEY);
        requireSet(plotPath, PlotPath.KEY);
        lossParameter.checkUsableWith(loss);
    }

    public static final Set<String> KEYS = Set.of(
            Keyset.KEY, DataSize.KEY, Workload.KEY, WorkloadPerm.KEY, WorkloadFactor.KEY, MaxErr.KEY,
            Activation.KEY, LossFunction.KEY, LossParameter.KEY, LearningRate.KEY, BatchSize.KEY,
            Epochs.KEY, HiddenLayers.KEY, Seed.KEY, PlotPath.KEY);

    public static Config read(Properties p) {
        rejectUnknownKeys(p);
        LossFunction loss = LossFunction.read(p).orElse(LossFunction.DEFAULT);
        return new Config(
                Keyset.read(p).orElseThrow(Keyset::missing),
                DataSize.read(p).orElseThrow(DataSize::missing),
                Workload.read(p).orElse(Workload.DEFAULT),
                WorkloadPerm.read(p).orElse(WorkloadPerm.DEFAULT),
                WorkloadFactor.read(p).orElse(WorkloadFactor.DEFAULT),
                MaxErr.read(p).orElse(MaxErr.DEFAULT),
                Activation.read(p).orElse(Activation.DEFAULT),
                loss,
                LossParameter.read(p).orElseGet(() -> LossParameter.defaultFor(loss)),
                LearningRate.read(p).orElseGet(() -> LearningRate.defaultFor(loss)),
                BatchSize.read(p).orElse(BatchSize.DEFAULT),
                Epochs.read(p).orElse(Epochs.DEFAULT),
                HiddenLayers.read(p).orElse(HiddenLayers.DEFAULT),
                Seed.read(p).orElse(Seed.DEFAULT),
                PlotPath.read(p).orElse(PlotPath.DEFAULT)
        );
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

    private static void requireSet(Object value, String key) {
        if (value == null) {
            throw new IllegalArgumentException(key + " must not be null");
        }
    }
}
