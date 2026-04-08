#include <portaudio.h>

#include <atomic>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
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
    std::atomic<float> volume{1.10f};
    std::atomic<float> hp_r{0.985f};
    std::atomic<float> compressor_threshold{0.060f};
    std::atomic<float> compressor_ratio{8.0f};
    std::atomic<float> compressor_makeup{2.20f};
    std::atomic<float> agc_target_rms{0.12f};
    std::atomic<float> agc_max_gain{10.0f};
};

struct DspState {
    float hp_prev_x = 0.0f;
    float hp_prev_y = 0.0f;
    float env = 0.0f;
    float agc_gain = 1.0f;
};

struct CallbackContext {
    RuntimeConfig* cfg = nullptr;
    DspState state;
    std::mutex stt_mutex;
    std::condition_variable stt_cv;
    std::queue<std::vector<float>> stt_queue;
    std::atomic<float> in_rms{0.0f};
    std::atomic<float> out_rms{0.0f};
    std::atomic<float> left_rms{0.0f};
    std::atomic<float> right_rms{0.0f};
    std::atomic<float> attenuation_db{0.0f};
    std::mutex telemetry_mutex;
    std::vector<float> latest_in_chunk;
    std::vector<float> latest_out_chunk;
};

std::vector<float> compute_spectrum_bins(const std::vector<float>& signal, int bins = 48) {
    if (signal.empty() || bins <= 0) {
        return {};
    }

    const int n = static_cast<int>(signal.size());
    const double min_hz = 80.0;
    const double max_hz = 6000.0;
    const double nyquist = kSampleRate * 0.5;
    const double max_used_hz = std::min(max_hz, nyquist - 1.0);

    std::vector<float> spectrum(static_cast<size_t>(bins), 0.0f);
    if (max_used_hz <= min_hz) {
        return spectrum;
    }

    constexpr double kPi = 3.14159265358979323846;

    for (int i = 0; i < bins; ++i) {
        const double t = (bins == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(bins - 1);
        const double target_hz = min_hz + (max_used_hz - min_hz) * t;
        const int k = std::max(1, std::min(n / 2 - 1, static_cast<int>(std::lround((target_hz * n) / kSampleRate))));
        const double omega = (2.0 * kPi * k) / static_cast<double>(n);
        const double coeff = 2.0 * std::cos(omega);

        double q0 = 0.0;
        double q1 = 0.0;
        double q2 = 0.0;
        for (int j = 0; j < n; ++j) {
            q0 = coeff * q1 - q2 + static_cast<double>(signal[static_cast<size_t>(j)]);
            q2 = q1;
            q1 = q0;
        }
        double power = q1 * q1 + q2 * q2 - coeff * q1 * q2;
        if (!std::isfinite(power) || power < 0.0) {
            power = 0.0;
        }
        spectrum[static_cast<size_t>(i)] = static_cast<float>(power);
    }

    float peak = 0.0f;
    for (float v : spectrum) {
        if (v > peak) {
            peak = v;
        }
    }
    if (peak > 1e-12f) {
        for (float& v : spectrum) {
            v = std::sqrt(v / peak);
        }
    }

    return spectrum;
}

std::string join_csv(const std::vector<float>& values) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << values[i];
    }
    return oss.str();
}

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

    const float attack = 0.22f;
    const float release = 0.015f;
    if (abs_x > st.env) {
        st.env = attack * abs_x + (1.0f - attack) * st.env;
    } else {
        st.env = release * abs_x + (1.0f - release) * st.env;
    }

    const float threshold = clampf(cfg.compressor_threshold.load(), 0.02f, 0.95f);
    const float ratio = clampf(cfg.compressor_ratio.load(), 1.0f, 12.0f);
    const float makeup = clampf(cfg.compressor_makeup.load(), 0.2f, 6.0f);
    const float target_rms = clampf(cfg.agc_target_rms.load(), 0.03f, 0.25f);
    const float max_gain = clampf(cfg.agc_max_gain.load(), 1.0f, 20.0f);

    float y = x;
    if (st.env <= threshold) {
        // Leave low-level signals uncompressed before makeup/AGC stage.
        y = x;
    } else {
        const float over = st.env - threshold;
        const float compressed = threshold + over / ratio;
        const float gain = compressed / (st.env + 1e-9f);
        y = x * gain;
    }

    // Restore intelligibility after compression and keep quiet speech audible.
    y *= makeup;

    // Slow AGC stage to emulate hearing-aid style loudness normalization.
    const float desired_agc = clampf(target_rms / (st.env + 1e-6f), 0.35f, max_gain);
    st.agc_gain = 0.16f * desired_agc + 0.84f * st.agc_gain;
    y *= st.agc_gain;

    return y;
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

    double in_energy = 0.0;
    double out_energy = 0.0;

    for (unsigned long i = 0; i < frame_count; ++i) {
        float x = 0.0f;
        if (in) {
            x = in[i * kInputChannels];
        }
        in_energy += static_cast<double>(x) * static_cast<double>(x);

        float y = x;
        if (!ctx->cfg->bypass_all.load()) {
            y = apply_highpass(y, ctx->state, *ctx->cfg);
            y = apply_compressor(y, ctx->state, *ctx->cfg);
        }

        y *= clampf(ctx->cfg->volume.load(), 0.0f, 6.0f);
        y = soft_clip(y);
        y = clampf(y, -0.95f, 0.95f);
        out_energy += static_cast<double>(y) * static_cast<double>(y);

        out[i * kOutputChannels] = y;
    }

    if (frame_count > 0) {
        const float in_rms = static_cast<float>(std::sqrt(in_energy / static_cast<double>(frame_count)));
        const float out_rms = static_cast<float>(std::sqrt(out_energy / static_cast<double>(frame_count)));
        const float att_db = 20.0f * std::log10((out_rms + 1e-9f) / (in_rms + 1e-9f));
        ctx->in_rms.store(in_rms);
        ctx->out_rms.store(out_rms);
        ctx->left_rms.store(in_rms);
        ctx->right_rms.store(in_rms);
        ctx->attenuation_db.store(att_db);

        std::vector<float> in_chunk(frame_count);
        std::vector<float> out_chunk(frame_count);
        for (unsigned long i = 0; i < frame_count; ++i) {
            in_chunk[i] = in ? in[i * kInputChannels] : 0.0f;
            out_chunk[i] = out[i * kOutputChannels];
        }
        {
            std::lock_guard<std::mutex> lock(ctx->telemetry_mutex);
            ctx->latest_in_chunk = std::move(in_chunk);
            ctx->latest_out_chunk = std::move(out_chunk);
        }
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

    auto next_telemetry_emit = std::chrono::steady_clock::now();

    while (g_running.load() && Pa_IsStreamActive(stream) == 1) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_telemetry_emit) {
            const float meter_l = ctx.left_rms.load();
            const float meter_r = ctx.right_rms.load();
            const float in_rms = ctx.in_rms.load();
            const float out_rms = ctx.out_rms.load();
            const float att_db = ctx.attenuation_db.load();

            std::vector<float> in_chunk;
            std::vector<float> out_chunk;
            {
                std::lock_guard<std::mutex> lock(ctx.telemetry_mutex);
                in_chunk = ctx.latest_in_chunk;
                out_chunk = ctx.latest_out_chunk;
            }

            const std::vector<float> spectrum_in = compute_spectrum_bins(in_chunk, 48);
            const std::vector<float> spectrum_out = compute_spectrum_bins(out_chunk, 48);

            double in_pow = 0.0;
            double out_pow = 0.0;
            const size_t n = std::min(spectrum_in.size(), spectrum_out.size());
            for (size_t i = 0; i < n; ++i) {
                in_pow += static_cast<double>(spectrum_in[i]) * static_cast<double>(spectrum_in[i]);
                out_pow += static_cast<double>(spectrum_out[i]) * static_cast<double>(spectrum_out[i]);
            }
            const float reduction_db = static_cast<float>(10.0 * std::log10((in_pow + 1e-9) / (out_pow + 1e-9)));

            std::cout << "[METER] left=" << meter_l << " right=" << meter_r << std::endl;
            std::cout << "[QUALITY] in_rms=" << in_rms
                      << " out_rms=" << out_rms
                      << " attenuation_db=" << att_db
                      << " reduction_db=" << reduction_db
                      << " band_low_hz=120"
                      << " band_high_hz=6000"
                      << " nr_active=0"
                      << " spectrum_in=" << join_csv(spectrum_in)
                      << " spectrum_out=" << join_csv(spectrum_out)
                      << std::endl;

            next_telemetry_emit = now + std::chrono::milliseconds(160);
        }
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
