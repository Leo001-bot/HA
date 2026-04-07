#include <portaudio.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_SHERPA_ONNX_C_API
#include <sherpa-onnx/c-api/c-api.h>
#endif

namespace {

constexpr double kSampleRate = 16000.0;
constexpr unsigned long kFramesPerBuffer = 512;
constexpr int kInputChannels = 1;
constexpr int kOutputChannels = 1;
constexpr size_t kSttQueueMaxChunks = 32;

std::atomic<bool> g_running{true};

struct RuntimeConfig {
    std::atomic<bool> bypass_all{false};
    std::atomic<float> volume{0.65f};
    std::atomic<float> hp_r{0.985f};
    std::atomic<float> compressor_threshold{0.18f};
    std::atomic<float> compressor_ratio{3.0f};
};

struct DspState {
    float hp_prev_x = 0.0f;
    float hp_prev_y = 0.0f;
    float env = 0.0f;
};

struct CallbackContext {
    RuntimeConfig* cfg = nullptr;
    DspState state;
    std::mutex stt_mutex;
    std::condition_variable stt_cv;
    std::queue<std::vector<float>> stt_queue;
};

struct ModelBundle {
    std::string tokens;
    std::string encoder;
    std::string decoder;
    std::string joiner;
};

std::optional<ModelBundle> find_model_bundle(const std::filesystem::path& model_root) {
    if (!std::filesystem::exists(model_root)) {
        return std::nullopt;
    }

    auto best_bundle_score = -1;
    std::optional<ModelBundle> best;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(model_root)) {
        if (!entry.is_regular_file() || entry.path().filename() != "tokens.txt") {
            continue;
        }

        const auto parent = entry.path().parent_path();
        std::string encoder_int8;
        std::string encoder_fp;
        std::string decoder_int8;
        std::string decoder_fp;
        std::string joiner_int8;
        std::string joiner_fp;

        for (const auto& f : std::filesystem::directory_iterator(parent)) {
            if (!f.is_regular_file()) {
                continue;
            }
            const auto name = f.path().filename().string();
            const auto full = f.path().string();
            const bool is_int8 = name.find(".int8.onnx") != std::string::npos;

            if (name.rfind("encoder", 0) == 0 && name.find(".onnx") != std::string::npos) {
                if (is_int8) encoder_int8 = full; else encoder_fp = full;
            } else if (name.rfind("decoder", 0) == 0 && name.find(".onnx") != std::string::npos) {
                if (is_int8) decoder_int8 = full; else decoder_fp = full;
            } else if (name.rfind("joiner", 0) == 0 && name.find(".onnx") != std::string::npos) {
                if (is_int8) joiner_int8 = full; else joiner_fp = full;
            }
        }

        const std::string encoder = !encoder_int8.empty() ? encoder_int8 : encoder_fp;
        const std::string decoder = !decoder_fp.empty() ? decoder_fp : decoder_int8;
        const std::string joiner = !joiner_int8.empty() ? joiner_int8 : joiner_fp;

        if (encoder.empty() || decoder.empty() || joiner.empty()) {
            continue;
        }

        int score = 0;
        const auto p = parent.string();
        if (p.find("bilingual-zh-en-2023-02-20") != std::string::npos) score += 1000;
        if (p.find("bilingual") != std::string::npos) score += 200;
        if (p.find("zh-en") != std::string::npos) score += 100;
        if (!encoder_int8.empty()) score += 30;
        if (!joiner_int8.empty()) score += 30;

        if (score > best_bundle_score) {
            best_bundle_score = score;
            best = ModelBundle{entry.path().string(), encoder, decoder, joiner};
        }
    }

    return best;
}

float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

float soft_clip(float x) {
    // Keep output below hard clipping while preserving some dynamics.
    return std::tanh(1.1f * x) / std::tanh(1.1f);
}

float apply_compressor(float x, DspState& st, const RuntimeConfig& cfg) {
    const float abs_x = std::fabs(x);

    const float attack = 0.30f;
    const float release = 0.02f;
    if (abs_x > st.env) {
        st.env = attack * abs_x + (1.0f - attack) * st.env;
    } else {
        st.env = release * abs_x + (1.0f - release) * st.env;
    }

    const float threshold = clampf(cfg.compressor_threshold.load(), 0.02f, 0.95f);
    const float ratio = clampf(cfg.compressor_ratio.load(), 1.0f, 12.0f);

    if (st.env <= threshold) {
        return x;
    }

    const float over = st.env - threshold;
    const float compressed = threshold + over / ratio;
    const float gain = compressed / (st.env + 1e-9f);
    return x * gain;
}

float apply_highpass(float x, DspState& st, const RuntimeConfig& cfg) {
    const float r = clampf(cfg.hp_r.load(), 0.90f, 0.999f);
    const float y = x - st.hp_prev_x + r * st.hp_prev_y;
    st.hp_prev_x = x;
    st.hp_prev_y = y;
    return y;
}

std::optional<PaDeviceIndex> parse_device_index_env(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return std::nullopt;
    }

    try {
        long parsed = std::stol(value);
        if (parsed < 0) {
            return std::nullopt;
        }
        return static_cast<PaDeviceIndex>(parsed);
    } catch (...) {
        return std::nullopt;
    }
}

int audio_callback(const void* input,
                   void* output,
                   unsigned long frame_count,
                   const PaStreamCallbackTimeInfo*,
                   PaStreamCallbackFlags,
                   void* user_data) {
    auto* ctx = static_cast<CallbackContext*>(user_data);
    const auto* in = static_cast<const float*>(input);
    auto* out = static_cast<float*>(output);

    if (!out || !ctx || !ctx->cfg) {
        return paContinue;
    }

    for (unsigned long i = 0; i < frame_count; ++i) {
        float x = 0.0f;
        if (in) {
            x = in[i * kInputChannels];
        }

        float y = x;
        if (!ctx->cfg->bypass_all.load()) {
            y = apply_highpass(y, ctx->state, *ctx->cfg);
            y = apply_compressor(y, ctx->state, *ctx->cfg);
        }

        y *= clampf(ctx->cfg->volume.load(), 0.0f, 6.0f);
        y = soft_clip(y);
        y = clampf(y, -0.95f, 0.95f);

        out[i * kOutputChannels] = y;
    }

    if (in && frame_count > 0) {
        std::vector<float> chunk(frame_count);
        for (unsigned long i = 0; i < frame_count; ++i) {
            chunk[i] = in[i * kInputChannels];
        }
        {
            std::lock_guard<std::mutex> lock(ctx->stt_mutex);
            if (ctx->stt_queue.size() >= kSttQueueMaxChunks) {
                ctx->stt_queue.pop();
            }
            ctx->stt_queue.push(std::move(chunk));
        }
        ctx->stt_cv.notify_one();
    }

    return g_running.load() ? paContinue : paComplete;
}

void signal_handler(int) {
    g_running.store(false);
}

void control_loop(RuntimeConfig& cfg) {
    std::cout << "Commands: b=toggle bypass, +=volume up, -=volume down, q=quit" << std::endl;
    while (g_running.load()) {
        char c = 0;
        if (!(std::cin >> c)) {
            g_running.store(false);
            break;
        }

        if (c == 'b') {
            const bool next = !cfg.bypass_all.load();
            cfg.bypass_all.store(next);
            std::cout << "bypass_all=" << (next ? "ON" : "OFF") << std::endl;
        } else if (c == '+') {
            float v = cfg.volume.load();
            v = clampf(v + 0.05f, 0.0f, 6.0f);
            cfg.volume.store(v);
            std::cout << "volume=" << v << std::endl;
        } else if (c == '-') {
            float v = cfg.volume.load();
            v = clampf(v - 0.05f, 0.0f, 6.0f);
            cfg.volume.store(v);
            std::cout << "volume=" << v << std::endl;
        } else if (c == 'q') {
            g_running.store(false);
            break;
        }
    }
}

#ifdef HAVE_SHERPA_ONNX_C_API
void stt_loop(CallbackContext& ctx, const std::filesystem::path& model_root) {
    const auto bundle_opt = find_model_bundle(model_root);
    if (!bundle_opt.has_value()) {
        std::cerr << "STT disabled: no valid bundle found under " << model_root << std::endl;
        return;
    }
    const auto bundle = bundle_opt.value();

    SherpaOnnxOnlineRecognizerConfig config;
    std::memset(&config, 0, sizeof(config));
    config.model_config.tokens = bundle.tokens.c_str();
    config.model_config.transducer.encoder = bundle.encoder.c_str();
    config.model_config.transducer.decoder = bundle.decoder.c_str();
    config.model_config.transducer.joiner = bundle.joiner.c_str();
    config.model_config.num_threads = 2;
    config.model_config.provider = "cpu";
    config.feat_config.sample_rate = static_cast<int32_t>(kSampleRate);
    config.feat_config.feature_dim = 80;
    config.decoding_method = "greedy_search";
    config.enable_endpoint = 1;
    config.rule1_min_trailing_silence = 1.2f;
    config.rule2_min_trailing_silence = 0.35f;
    config.rule3_min_utterance_length = 8.0f;

    auto* recognizer = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!recognizer) {
        std::cerr << "STT disabled: failed to create sherpa-onnx recognizer" << std::endl;
        return;
    }
    auto* stream = SherpaOnnxCreateOnlineStream(recognizer);
    if (!stream) {
        std::cerr << "STT disabled: failed to create online stream" << std::endl;
        SherpaOnnxDestroyOnlineRecognizer(recognizer);
        return;
    }

    std::cout << "STT started with bundle:" << std::endl;
    std::cout << "  tokens:  " << bundle.tokens << std::endl;
    std::cout << "  encoder: " << bundle.encoder << std::endl;
    std::cout << "  decoder: " << bundle.decoder << std::endl;
    std::cout << "  joiner:  " << bundle.joiner << std::endl;

    std::string last_text;
    while (g_running.load()) {
        std::vector<float> chunk;
        {
            std::unique_lock<std::mutex> lock(ctx.stt_mutex);
            ctx.stt_cv.wait_for(lock, std::chrono::milliseconds(20), [&ctx]() {
                return !ctx.stt_queue.empty() || !g_running.load();
            });
            if (!g_running.load()) {
                break;
            }
            if (ctx.stt_queue.empty()) {
                continue;
            }
            chunk = std::move(ctx.stt_queue.front());
            ctx.stt_queue.pop();
        }

        SherpaOnnxOnlineStreamAcceptWaveform(
            stream,
            static_cast<int32_t>(kSampleRate),
            chunk.data(),
            static_cast<int32_t>(chunk.size())
        );

        while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
            SherpaOnnxDecodeOnlineStream(recognizer, stream);
        }

        auto* result = SherpaOnnxGetOnlineStreamResult(recognizer, stream);
        if (result && result->text) {
            std::string text = result->text;
            if (!text.empty() && text != last_text) {
                std::cout << "[STT] " << text << std::endl;
                last_text = text;
            }
        }
        SherpaOnnxDestroyOnlineRecognizerResult(result);

        if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
            SherpaOnnxOnlineStreamReset(recognizer, stream);
            last_text.clear();
        }
    }

    SherpaOnnxDestroyOnlineStream(stream);
    SherpaOnnxDestroyOnlineRecognizer(recognizer);
}
#else
void stt_loop(CallbackContext&, const std::filesystem::path&) {
    std::cout << "STT C API not linked. Build with sherpa-onnx C API to enable STT." << std::endl;
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);

    std::filesystem::path model_root = "../models";
    if (argc >= 2) {
        model_root = argv[1];
    }

    RuntimeConfig cfg;
    CallbackContext ctx;
    ctx.cfg = &cfg;

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    PaStreamParameters in_params{};
    PaStreamParameters out_params{};

    const auto input_device_override = parse_device_index_env("PA_INPUT_DEVICE");
    const auto output_device_override = parse_device_index_env("PA_OUTPUT_DEVICE");

    in_params.device = input_device_override.has_value()
                           ? *input_device_override
                           : Pa_GetDefaultInputDevice();
    out_params.device = output_device_override.has_value()
                            ? *output_device_override
                            : Pa_GetDefaultOutputDevice();

    if (in_params.device == paNoDevice || out_params.device == paNoDevice) {
        std::cerr << "No default audio input/output device" << std::endl;
        Pa_Terminate();
        return 1;
    }

    const PaDeviceInfo* in_info = Pa_GetDeviceInfo(in_params.device);
    const PaDeviceInfo* out_info = Pa_GetDeviceInfo(out_params.device);
    std::cout << "Using input device:  "
              << (in_info ? in_info->name : "<unknown>")
              << " (default index " << in_params.device << ")" << std::endl;
    std::cout << "Using output device: "
              << (out_info ? out_info->name : "<unknown>")
              << " (default index " << out_params.device << ")" << std::endl;

    in_params.channelCount = kInputChannels;
    in_params.sampleFormat = paFloat32;
    in_params.suggestedLatency = Pa_GetDeviceInfo(in_params.device)->defaultLowInputLatency;
    in_params.hostApiSpecificStreamInfo = nullptr;

    out_params.channelCount = kOutputChannels;
    out_params.sampleFormat = paFloat32;
    out_params.suggestedLatency = Pa_GetDeviceInfo(out_params.device)->defaultLowOutputLatency;
    out_params.hostApiSpecificStreamInfo = nullptr;

    PaStream* stream = nullptr;
    err = Pa_OpenStream(
        &stream,
        &in_params,
        &out_params,
        kSampleRate,
        kFramesPerBuffer,
        paNoFlag,
        audio_callback,
        &ctx
    );

    if (err != paNoError) {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    std::cout << "C++ hearing-aid realtime engine started" << std::endl;
    std::cout << "SampleRate=" << kSampleRate << ", FramesPerBuffer=" << kFramesPerBuffer
              << ", Channels in/out=" << kInputChannels << "/" << kOutputChannels << std::endl;
    std::cout << "Model root=" << model_root << std::endl;

    std::thread controls(control_loop, std::ref(cfg));
    std::thread stt_worker(stt_loop, std::ref(ctx), model_root);

    while (g_running.load() && Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(50);
    }

    g_running.store(false);
    ctx.stt_cv.notify_all();
    if (controls.joinable()) {
        controls.join();
    }
    if (stt_worker.joinable()) {
        stt_worker.join();
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cout << "C++ hearing-aid realtime engine stopped" << std::endl;
    return 0;
}
