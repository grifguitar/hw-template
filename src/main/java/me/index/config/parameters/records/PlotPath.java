package me.index.config.parameters.records;

import java.nio.file.Path;
import java.util.Optional;
import java.util.Properties;

public record PlotPath(String value) {
    public static final String KEY = "plot.path";
    public static final PlotPath DEFAULT = new PlotPath("plot.pdf");

    public PlotPath {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("property '" + KEY + "': expected a non-blank path");
        }
    }

    public static Optional<PlotPath> read(Properties p) {
        String value = p.getProperty(KEY);
        return value == null ? Optional.empty() : Optional.of(new PlotPath(value.trim()));
    }

    public Path path() {
        return Path.of(value);
    }
}
