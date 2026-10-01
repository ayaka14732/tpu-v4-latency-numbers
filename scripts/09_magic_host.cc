// Included after the fixed 0.0.46 backend shim and the existing Magic Queue helper.
// The two timestamps come from one BC's paired LCC resident. No host clock conversion.
namespace {
bool ReadMagicCounterFlag(HostReadSession* session, std::uint32_t flag, std::uint32_t* value) {
    void* wrapper = __atomic_load_n(&resident_wrappers[session->wrapper], __ATOMIC_ACQUIRE);
    if (!wrapper) return false;
    const auto* vtable = *reinterpret_cast<const std::uintptr_t* const*>(wrapper);
    if (!vtable || !vtable[25]) return false;
    using Read = void (*)(void*, void*, std::uint32_t);
    alignas(16) unsigned char result[32] = {};
    reinterpret_cast<Read>(vtable[25])(result, wrapper, flag);
    if (*reinterpret_cast<const std::uintptr_t*>(result) != 1) return false;
    *value = *reinterpret_cast<const std::uint32_t*>(result + 8);
    return true;
}

bool MagicCounterGate(HostReadSession* session, std::uint32_t* sequence) {
    std::uintptr_t status = 0;
    char diagnostic[4096] = {};
    const int code = TpuEmbeddingShimWriteObservedResidentSflag(session->wrapper, 1, 1, &status, diagnostic, sizeof(diagnostic));
    if (code || status != 1) return false;
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    std::uint32_t observed = 0;
    ++*sequence;
    do {
        if (!ReadMagicCounterFlag(session, 28, &observed)) return false;
        if (observed > *sequence || Clock::now() > deadline) return false;
    } while (observed != *sequence);
    return true;
}
}

extern "C" int MagicHostWindowLcc(void* opaque, std::uint64_t first, std::uint64_t slots, std::uint64_t window,
                                  std::uint32_t* sequence, std::uint64_t* endpoints, int empty) {
    auto* session = static_cast<HostReadSession*>(opaque);
    if (!session || !sequence || !endpoints || !slots || first >= slots || !window || window > 64 || session->size % window) return -1;
    if (resident_owned_hbm_tokens[session->wrapper] != session->token) return -1;
    const auto size = session->size / window;
    if (size % 4096 || slots > resident_owned_hbm_sizes[session->wrapper] / size) return -1;
    const auto* base = resident_libtpu_base;
    using Read = void (*)(void*, RawHostSlice*, int, std::int64_t, int, const RawPxcIoOptionalSyncFlag*, RawPxcIoAnyInvocable*);
    const auto read = reinterpret_cast<Read>(const_cast<unsigned char*>(base) + 0xc99c6b0);
    RawPxcIoOptionalSyncFlag no_completion{};
    auto completions = std::make_unique<RawPxcRemoteTransferCompletion[]>(window);
    if (!MagicCounterGate(session, sequence)) return -2;
    if (!empty) {
        for (std::uint64_t w = 0; w < window; ++w) {
            auto* pointer = &completions[w];
            RawPxcIoAnyInvocable callback{};
            std::memcpy(callback.state, &pointer, sizeof(pointer));
            callback.manager = reinterpret_cast<void (*)(int, void*, void*)>(const_cast<unsigned char*>(base) + 0xc90b520);
            callback.invoker = &CompleteRawPxcRemoteTransfer;
            auto slice = session->slice;
            slice.size = size;
            slice.address0 += w * size;
            slice.address1 += w * size;
            const auto address = resident_owned_hbm_addresses[session->wrapper] + ((first + w) % slots) * size;
            read(session->driver, &slice, 0, address, session->receive_queue, &no_completion, &callback);
            callback.manager(0, &callback, &callback);
        }
        const auto deadline = Clock::now() + std::chrono::seconds(30);
        for (std::uint64_t w = 0; w < window; ++w) {
            std::uint64_t spins = 0;
            while (!__atomic_load_n(&completions[w].called, __ATOMIC_ACQUIRE)) {
                _mm_pause();
                if ((++spins & 0xfffff) == 0 && Clock::now() > deadline) std::abort();
            }
            if (!completions[w].ok) return -3;
        }
    }
    if (!MagicCounterGate(session, sequence)) return -4;
    std::uint32_t halves[4] = {};
    for (std::uint32_t i = 0; i < 4; ++i) {
        if (!ReadMagicCounterFlag(session, 24 + i, &halves[i])) return -5;
    }
    endpoints[0] = (static_cast<std::uint64_t>(halves[1]) << 32) | halves[0];
    endpoints[1] = (static_cast<std::uint64_t>(halves[3]) << 32) | halves[2];
    return 0;
}
