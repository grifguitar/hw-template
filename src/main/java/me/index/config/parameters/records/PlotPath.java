package me.index.config.parameters.records;

import me.index.config.Config;

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

    public static Optional<PlotPath> parse(Properties p) {
        return Config.parseRecord(p, KEY, "a path", PlotPath::new);
    }

    public Path path() {
        return Path.of(value);
    }
}
