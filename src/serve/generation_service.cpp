#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "serve/translate.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    std::size_t peak   = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

void trim_cache_markers(std::vector<ninfer::PromptCacheMarker>& markers, std::uint32_t maximum) {
    std::vector<ninfer::PromptCacheMarker> unique;
    unique.reserve(markers.size());
    for (const ninfer::PromptCacheMarker& marker : markers) {
        if (std::find(unique.begin(), unique.end(), marker) == unique.end()) {
            unique.push_back(marker);
        }
    }
    if (unique.size() > maximum) {
        unique.erase(unique.begin(), unique.end() - static_cast<std::ptrdiff_t>(maximum));
    }
    markers = std::move(unique);
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninfer::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

} // namespace

// Read the pool geometry once, on the first request that needs it. Not at construction: the core's
// `instance_.program` is created by the worker, so an earlier read dereferences nothing and the
// process dies without an error line (measured 2026-10-01 on sm_89; the frozen binary, which never
// reads it that early, serves the same command line fine). A zero page count still means "the Engine
// publishes no KV capacity", and then the check stays off.
GenerationService::PoolSnapshot GenerationService::pool_snapshot() const {
    std::call_once(pool_once_, [this] {
        const ninfer::MemorySummary memory = engine_->memory_summary();
        pool_snapshot_.pool_pages  = memory.kv_capacity_page_groups;
        pool_snapshot_.pool_tokens = memory.kv_capacity;
        // kv_capacity is page_groups * page_tokens exactly (SequenceCapacityCurve::resolved_tokens).
        pool_snapshot_.page_tokens =
            pool_snapshot_.pool_pages == 0
                ? 0U
                : static_cast<std::uint32_t>(memory.kv_capacity / pool_snapshot_.pool_pages);
    });
    return pool_snapshot_;
}

// State for the over-pool warning, FILE-LOCAL on purpose (2026-10-02).
//
// It used to be members of GenerationService. Adding a member changes sizeof(GenerationService),
// which invalidates every translation unit that sees the class (9 of them, including the one that
// stack-allocates the service by value) and turned a one-file edit into a full ~50-minute rebuild.
// Nothing here needs to be per-instance -- the pool is one Engine per process -- so it belongs in
// the .cpp. Keep it that way: touching this file alone costs one recompile and one link.
namespace {
struct OverPoolWarnState {
    std::mutex mutex;
    std::chrono::steady_clock::time_point last_emit{};
    std::uint64_t suppressed = 0;  // over-pool requests since the last emitted line
};
constexpr auto kOverPoolInterval = std::chrono::seconds(30);
} // namespace

// The over-pool warning. Deliberately says what was MEASURED and what is UNKNOWN, and never says
// "broken": the point is to remove the silence around a long prompt whose middle can go missing, not
// to claim a root cause that is still unlocalized (2026-10-01 handoff, section 10 red line 4: no
// speculative fix for this one).
//
// Re-emitted periodically rather than once (2026-10-02). See the note on the declaration: the
// once-only version left every over-pool request after the first one silent, so "I saw no warning"
// was not evidence of safety -- which is exactly how a user reads a log. The number of requests
// skipped between emissions rides along on the next line, so the cadence loses no information.
//
// The interval is measured from the FIRST over-pool request of a spell, never from process start: a
// cold 70k-token prefill takes over a minute, and counting from process start would let that prefill
// push the first line past the interval and turn the one request that matters into a "suppressed" one.
void GenerationService::warn_on_prompt_over_pool(std::uint32_t prompt_count) const {
    const PoolSnapshot pool = pool_snapshot();
    if (pool.pool_pages == 0 || pool.page_tokens == 0 || prompt_count == 0) { return; }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(prompt_count) - 1ULL) / pool.page_tokens;
    if (pages <= pool.pool_pages) { return; }

    static OverPoolWarnState state;
    std::uint64_t suppressed = 0;
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard lock(state.mutex);
        if (state.last_emit.time_since_epoch().count() != 0 &&
            now - state.last_emit < kOverPoolInterval) {
            ++state.suppressed;
            return;
        }
        suppressed       = state.suppressed;
        state.suppressed = 0;
        state.last_emit  = now;
    }

    std::string message =
        "prompt exceeds the resident Device KV pool: prompt " + std::to_string(prompt_count) +
        " tokens (" + std::to_string(pages) + " pages) > pool " + std::to_string(pool.pool_tokens) +
        " tokens (" + std::to_string(pool.pool_pages) +
        " pages). Part of the prompt is therefore not resident, and the MIDDLE of such a prompt "
        "has been measured to go missing with no error line, HTTP 200 and a plausible wrong "
        "answer; the root cause is not localized. Treat every answer to this request as "
        "unverified against its context, and raise --kv-capacity to at least the prompt's token "
        "count (measured: prompt >= pool is the condition that reproduces it) before reading a "
        "result.";
    if (suppressed != 0) {
        message += " [" + std::to_string(suppressed) +
                   " more over-pool request(s) since the previous line]";
    }
    message +=
        " [This line repeats at most every 30 s while over-pool requests keep arriving; a MISSING"
        " line is never evidence that a request was safe.]";
    write_console_log(ConsoleLogLevel::Warning, message);
}

ninfer::EngineOptions make_engine_options(const ServeOptions& options) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = options.artifact_path;
    engine_options.chat_template_path       = options.chat_template_path;
    engine_options.device                   = options.device;
    engine_options.max_context              = options.max_context;
    engine_options.kv_capacity              = options.kv_capacity;
    engine_options.max_concurrency          = options.max_concurrency;
    engine_options.max_pending_requests     = options.max_pending_requests;
    engine_options.pending_timeout_ms       = options.pending_timeout_ms;
    engine_options.prefill_chunk            = options.prefill_chunk;
    engine_options.fast_prefill_kernel      = options.fast_prefill_kernel;
    engine_options.kv_cache                 = options.kv_cache;
    engine_options.enable_vision            = options.enable_vision;
    engine_options.vision_residency         = options.vision_residency;
    engine_options.vision_max_merged_tokens = options.vision_max_merged_tokens;
    engine_options.use_cuda_graph           = options.use_cuda_graph;
    engine_options.lm_head_q4               = options.lm_head_q4;
    engine_options.lm_head_q6               = options.lm_head_q6;
    engine_options.embedding_q4             = options.embedding_q4;
    engine_options.embedding_q6             = options.embedding_q6;
    engine_options.mtp_experts_q4           = options.mtp_experts_q4;
    engine_options.gdn_state_fp16           = options.gdn_state_fp16;
    engine_options.rope_yarn                = options.rope_yarn;
    engine_options.rope_yarn_factor         = options.rope_yarn_factor;
    engine_options.rope_scaling_factor           = options.rope_scaling_factor;
    engine_options.rope_scaling_original_context = options.rope_scaling_original_context;
    engine_options.structured_output        = options.structured_output;
    engine_options.concurrent_prefill       = options.concurrent_prefill;
    engine_options.recover_invariant_failures = options.recover_invariant_failures;
    engine_options.thinking_budget_message  = options.thinking_budget_message;
    engine_options.cuda_graph_allowance_bytes =
        static_cast<std::size_t>(options.cuda_graph_allowance_mib) << 20;
    engine_options.wddm_evictable_budget    = options.wddm_evictable_budget;
    engine_options.mlp_a8_decode            = options.mlp_a8_decode;
    engine_options.prefill_a8               = options.prefill_a8;
    engine_options.prefill_cublas           = options.prefill_cublas;
    engine_options.prefill_cublas_projections = options.prefill_cublas_projections;
    engine_options.glp_vector               = options.glp_vector;
    engine_options.glp_alpha                = options.glp_alpha;
    engine_options.speculative              = options.speculative;
    engine_options.context_cache            = options.context_cache;
    engine_options.devices                  = options.devices;
    engine_options.stage_layers             = options.stage_layers;
    engine_options.context_cost.preset_path = options.context_cost_presets;
    engine_options.device_profile           = options.device_profile;
    engine_options.device_profile_path      = options.device_profile_path;
    engine_options.media_cache_bytes        = options.media_cache_bytes;
    engine_options.media_live_bytes         = options.media_live_bytes;
    engine_options.media_preprocess_threads = options.media_preprocess_threads;
    return engine_options;
}

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer,
                                     DiagnosticObserver diagnostic_observer)
    : options_(std::move(options)) {
    // Inline ECC on GDDR6X GeForce cards reserves ~6.25% of VRAM for checksums and taxes
    // memory bandwidth on every access. Decode is bandwidth-bound, so an ECC-enabled card
    // silently loses a large share of its published throughput and KV capacity while looking
    // exactly like an engine regression. ECC is off by default on GeForce; warn loudly when
    // someone (or some tool) left it on.
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, options_.device) == cudaSuccess &&
            props.ECCEnabled != 0) {
            write_console_log(ConsoleLogLevel::Warning,
                              std::string("ECC is ENABLED on ") + props.name +
                                  ": GDDR6X stores ECC checksums in VRAM, costing ~6.25% of "
                                  "capacity (23,028 vs 24,564 MiB on a 24 GB card) and ~12% of "
                                  "memory bandwidth (measured 803 vs 902 GB/s on an RTX 3090 Ti). "
                                  "KV capacity and prefill suffer accordingly. ECC is off by "
                                  "default on GeForce; if this is not a deliberate reliability "
                                  "choice, disable it with `nvidia-smi -e 0` and reboot.");
        }
    }
    ninfer::EngineOptions engine_options = make_engine_options(options_);
    engine_options.startup_observer      = std::move(startup_observer);
    engine_options.diagnostic_observer   = std::move(diagnostic_observer);
    engine_           = std::make_unique<ninfer::Engine>(std::move(engine_options));
    // A null engine_ used to be able to survive construction: the next method call on it
    // (load_summary() in apps/serve/main.cpp, right before "engine ready") then walked into a null
    // `this` and produced a bare 0xC0000005 with no message anywhere. Never let that be quiet.
    if (engine_ == nullptr) {
        throw std::runtime_error(
            "engine construction produced a null instance -- see the startup log above for the "
            "step that failed");
    }
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
}

std::size_t GenerationService::admitted_requests() const {
    std::lock_guard lock(request_capacity_->mutex);
    return request_capacity_->active;
}

std::size_t GenerationService::peak_admitted_requests() const {
    std::lock_guard lock(request_capacity_->mutex);
    return request_capacity_->peak;
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
        request_capacity_->peak = std::max(request_capacity_->peak, request_capacity_->active);
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    return prepare_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& incoming,
                                                GenerationConsumerMode consumer_mode,
                                                ninfer::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                ContextCacheHints context_cache,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy) const {
    std::optional<GenerationRequest> unconstrained;
    if (incoming.structured_output.kind != StructuredOutputKind::None &&
        !options_.structured_output) {
        if (!options_.unconstrained_response_format) {
            ApiError error;
            error.message =
                "structured output requires the server to start with --structured-output";
            error.param = "response_format";
            error.code  = "response_format_not_supported";
            throw ApiException(std::move(error));
        }
        unconstrained.emplace(incoming);
        unconstrained->structured_output = {};
    }
    const GenerationRequest& request = unconstrained ? *unconstrained : incoming;
    PreparedRequest prepared;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    request_options.ngram_session = request.ngram_session;
    prepared.thinking_budget      = request_options.execution.thinking.budget;
    prepared.reasoning_effort     = semantics.reasoning_effort;
    prepared.preserve_thinking    = semantics.preserve_thinking;
    prepared.parallel_tool_calls  = request.parallel_tool_calls;
    const bool request_has_media  = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        input.context_cache.allow_engine_prefix_grid =
            input.context_cache.allow_engine_prefix_grid || options_.auto_prefix_grid;
        if (options_.derive_session_keys && !input.context_cache.session_key &&
            cache_participation == CacheParticipation::ReadWrite) {
            input.context_cache.session_key = derived_session_key(request);
        }
        trim_cache_markers(input.context_cache.markers,
                           engine_->options().context_cache.max_cache_markers_per_request.value());
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        // Removes the SILENCE around a long prompt whose middle can go missing (see the note on the
        // declaration). Fires before submit, so it lands in the log even if the answer never comes.
        warn_on_prompt_over_pool(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninfer::OutputConsumerMode::Streaming
                                                  : ninfer::OutputConsumerMode::Aggregate,
                                              observation, prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) { throw_request_error(exception); }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);
    if (result.first_token_logprobs) {
        const auto view = [this](const ninfer::TokenLogprob& entry) {
            return TokenLogprobView{.bytes   = engine_->token_bytes(entry.token),
                                    .logprob = entry.logprob};
        };
        FirstTokenLogprobsView logprobs{.selected = view(result.first_token_logprobs->selected)};
        for (const ninfer::TokenLogprob& entry : result.first_token_logprobs->top) {
            logprobs.top.push_back(view(entry));
        }
        outcome.first_token_logprobs = std::move(logprobs);
    }

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.overlay_windows           = result.timings.overlay_windows;
    outcome.metrics.overlay_exclusive_windows = result.timings.overlay_exclusive_windows;
    outcome.metrics.overlay_window_seconds  = result.timings.overlay_window_seconds;
    outcome.metrics.overlay_evict_seconds   = result.timings.overlay_evict_seconds;
    outcome.metrics.overlay_restore_seconds = result.timings.overlay_restore_seconds;
    outcome.metrics.overlay_evicted_bytes   = result.timings.overlay_evicted_bytes;
    outcome.metrics.overlay_staged_bytes    = result.timings.overlay_staged_bytes;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.engine_timing                = result.engine_timing;
    outcome.metrics.prefix_cache_hit_tokens      = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path            = result.prefix_reuse_path;
    outcome.metrics.materialization              = result.materialization;
    outcome.metrics.speculative_backend          = result.speculative.backend;
    outcome.metrics.speculative_draft_window     = result.speculative.draft_window;
    outcome.metrics.speculative_rounds           = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens     = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens  = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps   = result.speculative.fallback_steps;
    outcome.metrics.ngram_rounds                 = result.speculative.ngram_rounds;
    outcome.metrics.ngram_drafted_tokens         = result.speculative.ngram_drafted_tokens;
    outcome.metrics.ngram_accepted_tokens        = result.speculative.ngram_accepted_tokens;
    outcome.metrics.ngram_archive_rounds         = result.speculative.ngram_archive_rounds;
    outcome.metrics.ngram_archive_drafted_tokens = result.speculative.ngram_archive_drafted_tokens;
    outcome.metrics.ngram_archive_accepted_tokens =
        result.speculative.ngram_archive_accepted_tokens;
    outcome.metrics.ngram_archive = result.ngram_archive;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);
    outcome.metrics.speculative_adaptive           = result.speculative.adaptive;
    outcome.metrics.speculative_window_transitions = result.speculative.window_transitions;
    outcome.metrics.speculative_rounds_per_window = std::move(result.speculative.rounds_per_window);

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    // parallel_tool_calls=false promises the caller at most one tool call per assistant turn.
    // Decoding is not constrained, so enforce it here: keep the first call and drop the rest. The
    // model can ask for the next one on the following turn, which is how a sequential executor
    // consumes them anyway.
    if (!prepared.parallel_tool_calls && outcome.tool_calls.size() > 1) {
        outcome.tool_calls.resize(1);
    }
    return outcome;
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
    run(prepared, nullptr);
}

} // namespace ninfer::serve
