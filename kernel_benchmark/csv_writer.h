#pragma once

#include <string>

namespace bench_common {

inline std::string EscapeCSV(const std::string& field) {
    bool needs_quote = false;
    for (size_t i = 0; i < field.size(); ++i) {
        const char c = field[i];
        if (c == ',' || c == '"' || c == '\n') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        return field;
    }
    std::string out = "\"";
    for (size_t i = 0; i < field.size(); ++i) {
        if (field[i] == '"') {
            out += "\"\"";
        } else {
            out.push_back(field[i]);
        }
    }
    out += "\"";
    return out;
}

}  // namespace bench_common

