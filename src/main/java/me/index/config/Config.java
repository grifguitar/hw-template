package me.index.config;

import java.util.Properties;

public record Config(
        Keyset keyset,
        Workload workload,
        WorkloadPerm workloadPerm,
        WorkloadFactor workloadFactor,
        DataSize dataSize,
        MaxErr maxErr
) {
    public static Config read(Properties p) {
        try {
            String k = p.getProperty("keyset");
            String w = p.getProperty("workload.distribution");
            String wp = p.getProperty("workload.permutation");
            String wf = p.getProperty("workload.factor");
            String ds = p.getProperty("data.size");
            String err = p.getProperty("max.err");
            return new Config(
                    (k != null) ? Keyset.valueOf(k) : null,
                    (w != null) ? Workload.valueOf(w) : null,
                    (wp != null) ? WorkloadPerm.valueOf(wp) : null,
                    (wf != null) ? WorkloadFactor.valueOf(wf) : null,
                    (ds != null) ? DataSize.valueOf(ds) : null,
                    (err != null) ? MaxErr.valueOf(err) : null
            );
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
