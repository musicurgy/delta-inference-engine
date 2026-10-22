// tools/v41/engine/ds41_posix.hpp - the few POSIX file / memory calls the V4.1 engine makes, portable.
//
// Linux: thin forwards to open / pread / pwrite / close / fstat / posix_fadvise / posix_memalign (behaviour unchanged).
// Windows: files are opened with CreateFileW(FILE_FLAG_OVERLAPPED) and wrapped as CRT fds, so a pread from many
// threads on ONE fd runs concurrently (a synchronous handle serializes every ReadFile on the file object, which would
// cap the Engram / file-tier queue depth at 1).  O_DIRECT -> FILE_FLAG_NO_BUFFERING (same 4 KiB alignment rules the
// callers already keep).  Aligned blocks must be released with ds41_aligned_free (_aligned_free on Windows).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <malloc.h>

typedef SSIZE_T ssize_t;
typedef int64_t ds41_off_t;

#ifndef O_DIRECT
#define O_DIRECT 0x40000000
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#define POSIX_FADV_NORMAL 0
#define POSIX_FADV_RANDOM 1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4

namespace ds41_posix_detail {
inline std::wstring widen(const char* s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(n > 0 ? (size_t) n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    w.resize(w.size() - 1);
    return w;
}
inline HANDLE event() {   // one auto-reset event per thread for its overlapped reads / writes
    thread_local struct E { HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr); ~E() { if (h) CloseHandle(h); } } e;
    return e.h;
}
inline ssize_t xfer(int fd, void* buf, size_t n, int64_t off, bool wr) {
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    size_t done = 0;
    while (done < n) {
        const DWORD k = (DWORD) (n - done < (1u << 30) ? n - done : (1u << 30));
        OVERLAPPED ov{};
        const uint64_t o = (uint64_t) off + done;
        ov.Offset = (DWORD) o;
        ov.OffsetHigh = (DWORD) (o >> 32);
        ov.hEvent = event();
        DWORD got = 0;
        const BOOL ok = wr ? WriteFile(h, (const uint8_t*) buf + done, k, nullptr, &ov)
                           : ReadFile(h, (uint8_t*) buf + done, k, nullptr, &ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return done ? (ssize_t) done : -1;
        }
        if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return done ? (ssize_t) done : -1;
        }
        if (got == 0) break;
        done += got;
        if (got < k) break;
    }
    return (ssize_t) done;
}
}  // namespace ds41_posix_detail

inline int ds41_open(const char* path, int flags, int /*mode*/ = 0644) {
    const bool wr = (flags & (_O_WRONLY | _O_RDWR)) != 0;
    DWORD attrs = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (flags & O_DIRECT) attrs |= FILE_FLAG_NO_BUFFERING;
    const DWORD disp = (flags & _O_CREAT) ? ((flags & _O_TRUNC) ? CREATE_ALWAYS : OPEN_ALWAYS)
                                          : ((flags & _O_TRUNC) ? TRUNCATE_EXISTING : OPEN_EXISTING);
    const HANDLE h = CreateFileW(ds41_posix_detail::widen(path).c_str(), wr ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, disp, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle((intptr_t) h, wr ? 0 : _O_RDONLY);
    if (fd < 0) CloseHandle(h);
    return fd;
}
inline int ds41_close(int fd) { return _close(fd); }
inline ssize_t ds41_pread(int fd, void* buf, size_t n, ds41_off_t off) { return ds41_posix_detail::xfer(fd, buf, n, off, false); }
inline ssize_t ds41_pwrite(int fd, const void* buf, size_t n, ds41_off_t off) {
    return ds41_posix_detail::xfer(fd, const_cast<void*>(buf), n, off, true);
}
inline int ds41_fsync(int fd) { return FlushFileBuffers((HANDLE) _get_osfhandle(fd)) ? 0 : -1; }
inline int ds41_fadvise(int, ds41_off_t, ds41_off_t, int) { return 0; }   // no per-range hint on Windows
inline int64_t ds41_fsize(int fd) {
    LARGE_INTEGER s;
    return GetFileSizeEx((HANDLE) _get_osfhandle(fd), &s) ? (int64_t) s.QuadPart : -1;
}
inline bool ds41_exists(const char* path) { return GetFileAttributesW(ds41_posix_detail::widen(path).c_str()) != INVALID_FILE_ATTRIBUTES; }
inline void* ds41_aligned_alloc(size_t align, size_t bytes) { return _aligned_malloc(bytes ? bytes : align, align); }
inline void ds41_aligned_free(void* p) { _aligned_free(p); }
inline int ds41_setenv(const char* k, const char* v) { return _putenv_s(k, v); }

// DS41_HEAPCHECK=1 (Windows): validate the process heap at checkpoints - names the step after which it is corrupt
inline void ds41_heapcheck(const char * where, int a = -1) {
    static const bool on = std::getenv("DS41_HEAPCHECK") != nullptr;
    if (!on) return;
    HANDLE hs[256];
    const DWORD nh = GetProcessHeaps(256, hs);
    int bad = 0;
    for (DWORD i = 0; i < nh && i < 256; ++i) if (!HeapValidate(hs[i], 0, nullptr)) ++bad;
    std::fprintf(stderr, "heapcheck %s %d: %lu heaps, %d CORRUPT\n", where, a, (unsigned long) nh, bad);
    std::fflush(stderr);
}


#else  // POSIX

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

typedef off_t ds41_off_t;

inline int ds41_open(const char* path, int flags, int mode = 0644) { return ::open(path, flags, mode); }
inline int ds41_close(int fd) { return ::close(fd); }
inline ssize_t ds41_pread(int fd, void* buf, size_t n, ds41_off_t off) { return ::pread(fd, buf, n, off); }
inline ssize_t ds41_pwrite(int fd, const void* buf, size_t n, ds41_off_t off) { return ::pwrite(fd, buf, n, off); }
inline int ds41_fsync(int fd) { return ::fsync(fd); }
inline int ds41_fadvise(int fd, ds41_off_t off, ds41_off_t len, int advice) { return ::posix_fadvise(fd, off, len, advice); }
inline int64_t ds41_fsize(int fd) { struct stat st; return ::fstat(fd, &st) == 0 ? (int64_t) st.st_size : -1; }
inline bool ds41_exists(const char* path) { struct stat st; return ::stat(path, &st) == 0; }
inline void* ds41_aligned_alloc(size_t align, size_t bytes) {
    void* p = nullptr;
    return ::posix_memalign(&p, align, bytes) == 0 ? p : nullptr;
}
inline void ds41_aligned_free(void* p) { std::free(p); }
inline int ds41_setenv(const char* k, const char* v) { return ::setenv(k, v, 1); }
inline void ds41_heapcheck(const char*, int = -1) {}

#endif
