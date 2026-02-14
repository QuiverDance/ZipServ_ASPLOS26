#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace bench_common {

struct ManifestTensorInfo {
    int layer = -1;
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    std::string path;
    size_t nbytes = 0;
};

inline std::string Trim(const std::string& input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

inline std::vector<std::string> SplitByComma(const std::string& input) {
    std::vector<std::string> out;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = Trim(token);
        if (!token.empty()) {
            out.push_back(token);
        }
    }
    return out;
}

inline bool EndsWith(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

inline std::string ShapeToString(const std::vector<int>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

inline bool ParsePositiveIntList(const std::string& spec, std::vector<int>* out, std::string* error) {
    out->clear();
    std::vector<std::string> tokens = SplitByComma(spec);
    if (tokens.empty()) {
        *error = "empty list";
        return false;
    }
    std::set<int> uniq;
    for (size_t i = 0; i < tokens.size(); ++i) {
        int value = 0;
        try {
            value = std::stoi(tokens[i]);
        } catch (...) {
            *error = "invalid integer token: " + tokens[i];
            return false;
        }
        if (value <= 0) {
            *error = "token must be > 0: " + tokens[i];
            return false;
        }
        uniq.insert(value);
    }
    out->assign(uniq.begin(), uniq.end());
    return true;
}

inline bool ParseLayerSpec(const std::string& spec, std::set<int>* layers, std::string* error) {
    layers->clear();
    if (spec.empty()) {
        return true;
    }
    std::vector<std::string> tokens = SplitByComma(spec);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        size_t dash = token.find('-');
        if (dash == std::string::npos) {
            int layer = 0;
            try {
                layer = std::stoi(token);
            } catch (...) {
                *error = "Invalid layer token: " + token;
                return false;
            }
            if (layer < 0) {
                *error = "Layer index must be non-negative: " + token;
                return false;
            }
            layers->insert(layer);
            continue;
        }
        std::string start_str = Trim(token.substr(0, dash));
        std::string end_str = Trim(token.substr(dash + 1));
        int start = 0;
        int end = 0;
        try {
            start = std::stoi(start_str);
            end = std::stoi(end_str);
        } catch (...) {
            *error = "Invalid layer range token: " + token;
            return false;
        }
        if (start < 0 || end < 0 || start > end) {
            *error = "Invalid layer range: " + token;
            return false;
        }
        for (int layer = start; layer <= end; ++layer) {
            layers->insert(layer);
        }
    }
    return true;
}

inline bool TensorNameMatchesAnyFilter(const std::string& name, const std::vector<std::string>& filters) {
    if (filters.empty()) {
        return true;
    }
    for (size_t i = 0; i < filters.size(); ++i) {
        if (name.find(filters[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

inline size_t FileSize(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        return 0;
    }
    return static_cast<size_t>(ifs.tellg());
}

inline bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>* data, std::string* error) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs.is_open()) {
        *error = "Cannot open tensor file: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    std::streamoff sz = ifs.tellg();
    if (sz < 0) {
        *error = "Failed to get file size: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    data->resize(static_cast<size_t>(sz));
    if (sz > 0) {
        ifs.read(reinterpret_cast<char*>(data->data()), sz);
    }
    if (!ifs.good() && !ifs.eof()) {
        *error = "Failed to read file: " + path;
        return false;
    }
    return true;
}

namespace detail {

inline bool LocateJSONValue(const std::string& line, const std::string& key, size_t* value_pos) {
    std::string token = "\"" + key + "\"";
    size_t key_pos = line.find(token);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon_pos = line.find(':', key_pos + token.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    size_t pos = colon_pos + 1;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    if (pos >= line.size()) {
        return false;
    }
    *value_pos = pos;
    return true;
}

inline bool ExtractJSONString(const std::string& line, const std::string& key, std::string* value) {
    size_t pos = 0;
    if (!LocateJSONValue(line, key, &pos)) {
        return false;
    }
    if (line[pos] != '"') {
        return false;
    }
    ++pos;
    std::string out;
    while (pos < line.size()) {
        char c = line[pos++];
        if (c == '\\') {
            if (pos >= line.size()) {
                return false;
            }
            out.push_back(line[pos++]);
            continue;
        }
        if (c == '"') {
            *value = out;
            return true;
        }
        out.push_back(c);
    }
    return false;
}

inline bool ExtractJSONInt64(const std::string& line, const std::string& key, int64_t* value) {
    size_t pos = 0;
    if (!LocateJSONValue(line, key, &pos)) {
        return false;
    }
    size_t start = pos;
    if (line[pos] == '-') {
        ++pos;
    }
    while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    if (pos == start || (line[start] == '-' && pos == start + 1)) {
        return false;
    }
    try {
        *value = std::stoll(line.substr(start, pos - start));
    } catch (...) {
        return false;
    }
    return true;
}

inline bool ExtractJSONIntArray(const std::string& line, const std::string& key, std::vector<int>* values) {
    size_t pos = 0;
    if (!LocateJSONValue(line, key, &pos)) {
        return false;
    }
    if (line[pos] != '[') {
        return false;
    }
    ++pos;
    values->clear();

    while (pos < line.size()) {
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        if (pos >= line.size()) {
            return false;
        }
        if (line[pos] == ']') {
            ++pos;
            return true;
        }

        size_t start = pos;
        if (line[pos] == '-') {
            ++pos;
        }
        while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        if (pos == start || (line[start] == '-' && pos == start + 1)) {
            return false;
        }

        int64_t parsed = 0;
        try {
            parsed = std::stoll(line.substr(start, pos - start));
        } catch (...) {
            return false;
        }
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        values->push_back(static_cast<int>(parsed));

        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        if (pos >= line.size()) {
            return false;
        }
        if (line[pos] == ',') {
            ++pos;
            continue;
        }
        if (line[pos] == ']') {
            ++pos;
            return true;
        }
        return false;
    }
    return false;
}

inline bool ParseManifestLine(const std::string& line, ManifestTensorInfo* tensor, std::string* error) {
    int64_t layer_i64 = 0;
    int64_t nbytes_i64 = 0;
    if (!ExtractJSONInt64(line, "layer", &layer_i64)) {
        *error = "missing/invalid layer";
        return false;
    }
    if (layer_i64 < 0 || layer_i64 > std::numeric_limits<int>::max()) {
        *error = "layer out of range";
        return false;
    }
    tensor->layer = static_cast<int>(layer_i64);

    if (!ExtractJSONString(line, "name", &tensor->name)) {
        *error = "missing/invalid name";
        return false;
    }
    if (!ExtractJSONIntArray(line, "shape", &tensor->shape)) {
        *error = "missing/invalid shape";
        return false;
    }
    if (!ExtractJSONString(line, "dtype", &tensor->dtype)) {
        *error = "missing/invalid dtype";
        return false;
    }
    if (!ExtractJSONString(line, "path", &tensor->path)) {
        *error = "missing/invalid path";
        return false;
    }
    if (!ExtractJSONInt64(line, "nbytes", &nbytes_i64)) {
        *error = "missing/invalid nbytes";
        return false;
    }
    if (nbytes_i64 < 0) {
        *error = "nbytes < 0";
        return false;
    }
    tensor->nbytes = static_cast<size_t>(nbytes_i64);
    return true;
}

}  // namespace detail

inline bool LoadManifestJsonl(const std::string& path, std::vector<ManifestTensorInfo>* tensors, std::string* error) {
    tensors->clear();
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open()) {
        *error = "Failed to open manifest: " + path;
        return false;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        ManifestTensorInfo t;
        std::string parse_error;
        if (!detail::ParseManifestLine(line, &t, &parse_error)) {
            std::ostringstream oss;
            oss << "Manifest parse failed at line " << line_no << ": " << parse_error;
            *error = oss.str();
            return false;
        }
        tensors->push_back(t);
    }
    return true;
}

}  // namespace bench_common

