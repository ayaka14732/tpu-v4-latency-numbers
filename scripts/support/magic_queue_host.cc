// Included after the pinned backend shim in one translation unit.
// Current libtpu 0.0.46: MapDmaBuffer -> ReadFromPremappedSharedMemory
// -> ReadFromMemoryHelper -> MagicQueueDescriptor plus reserved outfeed receive.
#include <chrono>
#include <immintrin.h>
#include <memory>

namespace {
struct RawHostSlice {
    void* vptr;
    std::uint64_t size;
    std::uint64_t address0;
    std::uint64_t address1;
    std::int32_t queue = -1;
    std::uint32_t padding = 0;
};
static_assert(sizeof(RawHostSlice) == 40);
struct HostReadSession {
    void* driver = nullptr;
    void* mapped = nullptr;
    unsigned char* host = nullptr;
    std::size_t size = 0;
    std::size_t wrapper = 0;
    std::uint64_t token = 0;
    int receive_queue = 0;
    RawHostSlice slice{};
};
using Clock = std::chrono::steady_clock;
}

extern "C" int MagicHostCreate(std::size_t wrapper, std::uint64_t token, std::size_t size, void** result, void** host_output) {
    const auto* base = resident_libtpu_base;
    if (!base || !process_zero_runtime_active || wrapper >= kMaximumObservedResidentWrappers ||
        !token || resident_owned_hbm_tokens[wrapper] != token || size < 4096 || size > 64 * 1024 * 1024 || size % 4096) return -1;
    // These are function entries in the current image, not offsets from crucible-notes.
    constexpr unsigned char prologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};
    for (const auto offset : {0xc99a560, 0xc99c6b0, 0xc99ca90}) {
        if (std::memcmp(base + offset, prologue, sizeof(prologue))) return -2;
    }
    void* shared = nullptr;
    std::uint64_t owner_core = 0;
    if (!ResolveProcessZeroCmem(wrapper, &shared, &owner_core)) return -3;
    auto* session = new HostReadSession;
    session->driver = *reinterpret_cast<void**>(static_cast<unsigned char*>(shared) + 0x220);
    session->wrapper = wrapper;
    session->token = token;
    session->size = size;
    using Queue = int (*)(void*);
    session->receive_queue = reinterpret_cast<Queue>(const_cast<unsigned char*>(base) + 0xc9a46f0)(session->driver);
    void* storage = nullptr;
    if (posix_memalign(&storage, 4096, size + 8192)) { delete session; return -4; }
    session->host = static_cast<unsigned char*>(storage);
    std::memset(storage, 0xd3, size + 8192);
    // Native call sites pass StatusOr<unique_ptr<DmaBuffer>> through an sret
    // pointer: raw status at 0, owned buffer pointer at 8.
    std::uintptr_t mapped_result[2] = {};
    using Map = void (*)(void*, void*, const void*, std::size_t);
    reinterpret_cast<Map>(const_cast<unsigned char*>(base) + 0xc99a560)(mapped_result, session->driver, storage, size + 8192);
    if (mapped_result[0] != 1 || !mapped_result[1]) { free(storage); delete session; return -5; }
    session->mapped = reinterpret_cast<void*>(mapped_result[1]);
    const auto* words = reinterpret_cast<const std::uint64_t*>(session->mapped);
    if (words[1] != size + 8192) std::abort();
    session->slice = {const_cast<unsigned char*>(base) + 0x1dc661c0, size, words[2] + 4096, words[3] + 4096, -1, 0};
    *result = session;
    *host_output = session->host;
    return 0;
}

extern "C" int MagicHostRunWindow(void* opaque, std::uint64_t first_slot, std::uint64_t slots, std::uint64_t iterations, std::uint64_t window, double* elapsed_us) {
    auto* session = static_cast<HostReadSession*>(opaque);
    if (!session || !slots || !iterations || !window || window > 64 || session->size % window || first_slot >= slots ||
        resident_owned_hbm_tokens[session->wrapper] != session->token) return -1;
    const auto size = session->size / window;
    if (size % 4096 || slots > resident_owned_hbm_sizes[session->wrapper] / size) return -1;
    const auto* base = resident_libtpu_base;
    using Read = void (*)(void*, RawHostSlice*, int, std::int64_t, int, const RawPxcIoOptionalSyncFlag*, RawPxcIoAnyInvocable*);
    const auto read = reinterpret_cast<Read>(const_cast<unsigned char*>(base) + 0xc99c6b0);
    RawPxcIoOptionalSyncFlag no_completion{};
    auto completions = std::make_unique<RawPxcRemoteTransferCompletion[]>(window);
    const auto start = Clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        for (std::uint64_t w = 0; w < window; ++w) {
            auto* pointer = &completions[w];
            *pointer = {};
            RawPxcIoAnyInvocable callback{};
            std::memcpy(callback.state, &pointer, sizeof(pointer));
            callback.manager = reinterpret_cast<void (*)(int, void*, void*)>(const_cast<unsigned char*>(base) + 0xc90b520);
            callback.invoker = &CompleteRawPxcRemoteTransfer;
            auto slice = session->slice;
            slice.size = size;
            slice.address0 += w * size;
            slice.address1 += w * size;
            const auto address = resident_owned_hbm_addresses[session->wrapper] + ((first_slot + i * window + w) % slots) * size;
            read(session->driver, &slice, 0, address, session->receive_queue, &no_completion, &callback);
            // Native AnyInvocable is moved by read(). Destroy its now-empty shell.
            callback.manager(0, &callback, &callback);
        }
        bool ok = true;
        for (std::uint64_t w = 0; w < window; ++w) {
            std::uint64_t spins = 0;
            while (!__atomic_load_n(&completions[w].called, __ATOMIC_ACQUIRE)) {
                _mm_pause();
                if ((++spins & 0xfffff) == 0 && Clock::now() - start > std::chrono::seconds(30)) std::abort();
            }
            if (!completions[w].ok) { std::fprintf(stderr, "%s\n", completions[w].error); ok = false; }
        }
        if (!ok) return -2;
    }
    *elapsed_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    return 0;
}

extern "C" int MagicHostRun(void* opaque, std::uint64_t first_slot, std::uint64_t slots, std::uint64_t iterations, double* elapsed_us) {
    return MagicHostRunWindow(opaque, first_slot, slots, iterations, 1, elapsed_us);
}

extern "C" int MagicHostClose(void* opaque) {
    auto* session = static_cast<HostReadSession*>(opaque);
    if (!session) return -1;
    using Delete = void (*)(void*);
    const auto* vtable = *reinterpret_cast<const std::uintptr_t* const*>(session->mapped);
    reinterpret_cast<Delete>(vtable[1])(session->mapped);
    free(session->host);
    delete session;
    return 0;
}
