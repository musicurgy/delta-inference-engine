// tools/v41/engine/ds41_unmap.hpp - STRATA_FILE_TIER_UNMAP (default OFF): let the file tier's
// unbuffered (O_DIRECT / FILE_FLAG_NO_BUFFERING) reads run on shard files that nothing maps and nothing holds open
// buffered.
//
// Why: on NTFS a non-cached read of a file that has a data section (a mapped view, or a cached handle) goes through a
// cache-coherency check, and the readers of one file queue on it.  Siege-Ladder 0007 3aa63fac78 (13 Oct): 11 readers on
// one expert file 13.5 GB/s unmapped vs 4.5 mapped; 0009 74d0419d3d: host tensors copied out so the streamed files
// unmap, +13% decode (5070 Ti, Gen5, Win11).  Our own CHANGES.md:99: Engram unbuffered on a mapped shard ~10x slower.
// The ds41 path kept every shard mapped twice for the whole run (Ds41Dense's GgufModel, Ds41DSpark's im.main) and
// opened a buffered handle beside each O_DIRECT one (Ds41FileBlobs::sfd_).  Re-implemented here (Siege's patches are
// llama.cpp code; only the idea is used).
//
// With STRATA_FILE_TIER_UNMAP=1:
//   - Ds41Dense (CUDA backend only: the CPU backend binds its weights IN the mapping) and every Ds41DSpark init copy
//     token_embd.weight out ONCE (ds41_embd_copy, shared) and release their mappings after the upload;
//   - after the tier is built, Ds41FileBlobs::drop_buffered closes the buffered shard handles (read() and the
//     read_part fallback then read through the O_DIRECT twins + a pooled bounce, the same bytes) and touches each shard
//     unbuffered (ds41_unbuffered_touch) - again after every DSpark (re)init, which maps the model for its load.
// Exact: every byte read is the byte the mapping / buffered handle gave; only the I/O path differs.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "ds41_env.hpp"
#include "ds41_sys.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::v41 {

/// STRATA_FILE_TIER_UNMAP=1 turns the unmapped file tier on (default off: the A/B arm until the GPU cards decide).
inline bool ds41_p3_unmap() {
    static const bool on = [] {
        const char* e = ds41_env("STRATA_FILE_TIER_UNMAP", "STRATA_P3_UNMAP");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

/// The model's token_embd.weight bytes, copied out of `m`'s mapping once per file and shared by every caller
/// (Ds41Dense, each Ds41DSpark init): the same bytes the mapping held, so every embedding row is bit-identical.
/// nullptr (and err) when the tensor is missing, unmapped or out of bounds.
inline std::shared_ptr<const std::vector<uint8_t>> ds41_embd_copy(const GgufModel& m, std::string& err) {
    static std::mutex mu;
    static std::map<std::string, std::shared_ptr<const std::vector<uint8_t>>> cache;
    size_t si = 0;
    const TensorInfo* te = m.find("token_embd.weight", &si);
    if (!te) {
        err = "unmap: the model has no token_embd.weight";
        return nullptr;
    }
    const std::string key = m.shard(si).path() + "@" + std::to_string(te->offset);
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const uint8_t* src = m.shard(si).tensor_data(*te);
    const uint64_t bytes = tensor_payload_bytes(*te);
    if (!src || bytes == 0 || !m.in_bounds(*te, si)) {
        err = "unmap: token_embd.weight is not mapped or out of bounds";
        return nullptr;
    }
    auto v = std::make_shared<std::vector<uint8_t>>((size_t) bytes);
    std::memcpy(v->data(), src, (size_t) bytes);
    std::fprintf(stderr, "unmap: token_embd copied out of the mapping (%.2f GB of RAM)\n", (double) bytes / 1e9);
    std::shared_ptr<const std::vector<uint8_t>> c = v;
    cache.emplace(key, c);
    return c;
}

/// Opens and closes an unbuffered handle on each file; returns how many opened.  The FastFAT create path flushes and
/// purges a file's leftover data section when a non-cached open finds no cached handle left (and NTFS is expected to
/// do the same; tools/v41/probe_file_tier.py's 'remap' modes measure whether it does on this PC): after the
/// mappings are gone and the buffered handles closed, this is what lets the section go instead of lingering with the
/// pages it cached.  Harmless where it does nothing.
inline int ds41_unbuffered_touch(const std::vector<std::string>& files) {
    int n = 0;
    for (const auto& f : files) {
        const int fd = ds41_sys::open(f.c_str(), O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            ds41_sys::close(fd);
            ++n;
        }
    }
    return n;
}

}  // namespace strata::v41
