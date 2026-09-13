package me.index.bench;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class Args {
    static final class UsageException extends IllegalArgumentException {
        @java.io.Serial
        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private final Map<String, String> values;

    private Args(Map<String, String> values) {
        this.values = values;
    }

    static Args parse(String[] args, int from, Set<String> allowed) {
        Map<String, String> values = new LinkedHashMap<>();
        for (int i = from; i < args.length; i += 2) {
            String key = args[i];
            if (!allowed.contains(key)) {
                throw new UsageException("unknown option '" + key + "'");
            }
            if (i + 1 >= args.length) {
                throw new UsageException("option " + key + " needs a value");
            }
            if (values.putIfAbsent(key, args[i + 1]) != null) {
                throw new UsageException("option " + key + " is given twice");
            }
        }
        return new Args(values);
    }

    boolean has(String key) {
        return values.containsKey(key);
    }

    String required(String key) {
        String value = values.get(key);
        if (value == null) {
            throw new UsageException("missing required option " + key);
        }
        return value;
    }

    String string(String key, String defaultValue) {
        return values.getOrDefault(key, defaultValue);
    }

    Path path(String key) {
        return Path.of(required(key));
    }

    int intValue(String key, int defaultValue, int min) {
        String text = values.get(key);
        if (text == null) {
            return defaultValue;
        }
        int value;
        try {
            value = Integer.parseInt(text.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(key + ": expected an integer, got '" + text + "'");
        }
        if (value < min) {
            throw new UsageException(key + ": expected at least " + min + ", got " + value);
        }
        return value;
    }

    long longValue(String key, long defaultValue) {
        String text = values.get(key);
        if (text == null) {
            return defaultValue;
        }
        try {
            return Long.parseLong(text.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(key + ": expected an integer, got '" + text + "'");
        }
    }

    double doubleValue(String key, double defaultValue) {
        String text = values.get(key);
        if (text == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(text.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(key + ": expected a number, got '" + text + "'");
        }
    }

    List<String> list(String key, String defaultValue) {
        List<String> items = new ArrayList<>();
        for (String part : string(key, defaultValue).split(",")) {
            if (!part.isBlank()) {
                items.add(part.trim());
            }
        }
        return items;
    }

    static <E extends Enum<E>> E enumValue(String option, String text, E[] allowed) {
        String name = text.trim();
        for (E candidate : allowed) {
            if (candidate.name().equals(name)) {
                return candidate;
            }
        }
        throw new UsageException(option + ": unknown value '" + name + "'; allowed values: " + Arrays.toString(allowed));
    }
}
