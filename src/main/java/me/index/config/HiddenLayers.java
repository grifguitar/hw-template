package me.index.config;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Properties;

public record HiddenLayers(List<Integer> sizes) {
    public static final String KEY = "net.hidden.layers";
    public static final HiddenLayers DEFAULT = new HiddenLayers(List.of(4, 4));

    public HiddenLayers {
        sizes = List.copyOf(sizes);
        for (int size : sizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("property '" + KEY + "': sizes must be positive, got " + size);
            }
        }
    }

    public static Optional<HiddenLayers> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String list = value.trim();
        if (list.isEmpty()) {
            return Optional.of(new HiddenLayers(List.of()));
        }
        List<Integer> sizes = new ArrayList<>();
        for (String part : list.split(",", -1)) {
            try {
                sizes.add(Integer.parseInt(part.trim()));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "property '" + KEY + "': expected comma-separated ints, got '" + list + "'", e);
            }
        }
        return Optional.of(new HiddenLayers(sizes));
    }

    public int[] layerSizes() {
        int[] layers = new int[sizes.size() + 2];
        layers[0] = 1;
        for (int i = 0; i < sizes.size(); i++) {
            layers[i + 1] = sizes.get(i);
        }
        layers[layers.length - 1] = 1;
        return layers;
    }
}
