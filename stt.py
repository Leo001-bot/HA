import queue
import threading
import os
import time
from pathlib import Path

import numpy as np
import sherpa_onnx


class StreamingSTT:
    """Streaming STT using Sherpa-ONNX with model auto-discovery."""

    OFFICIAL_BILINGUAL_MODEL_URL = (
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/"
        "zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-"
        "bilingual-zh-en-2023-02-20-bilingual-chinese-english"
    )

    def __init__(self, model_path="", tokens_path="", sample_rate=16000, model_root="models"):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=32)
        self.text_queue = queue.Queue(maxsize=64)
        self._running = False
        self._thread = None
        self._last_text = ""
        self._silence_seconds = 0.0
        self._segment_seconds = 0.0

        # Tunables for real-time behavior.
        self._silence_threshold = 0.0012
        self._utterance_silence_seconds = 0.30
        self._partial_emit_interval = 0.04
        self._max_segment_seconds = 6.0
        self._last_partial_emit_time = 0.0
        self._diag = {
            "enabled": True,
            "sample_rate": int(sample_rate),
            "model_root": str(model_root),
            "partial_emits": 0,
            "final_emits": 0,
            "errors": 0,
            "queue_drops": 0,
            "last_error": "",
            "last_text": "",
        }

        self.recognizer = self._create_recognizer(
            model_path=model_path,
            tokens_path=tokens_path,
            model_root=model_root,
        )
        self.stream = self.recognizer.create_stream()

    def _create_recognizer(self, model_path="", tokens_path="", model_root="models"):
        root = Path(model_root)
        if not root.exists():
            raise FileNotFoundError(f"Model directory not found: {root}")

        model_path = "" if model_path is None else str(model_path).strip()
        tokens_path = "" if tokens_path is None else str(tokens_path).strip()

        if model_path and tokens_path:
            model_file = Path(model_path)
            tokens_file = Path(tokens_path)
            if model_file.is_file() and tokens_file.is_file():
                return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                    model=str(model_file),
                    tokens=str(tokens_file),
                    num_threads=2,
                    provider="cpu",
                    sample_rate=self.sample_rate,
                    feature_dim=80,
                )

        transducer = self._find_transducer_bundle(root)
        if transducer is not None:
            # Keep STT CPU usage moderate so real-time audio callback has headroom.
            # This bundle matches the official Sherpa bilingual zipformer example.
            num_threads = max(2, min(4, (os.cpu_count() or 4) // 2))
            self._diag["model_type"] = "transducer"
            self._diag["model_files"] = {
                "encoder": str(transducer["encoder"]),
                "decoder": str(transducer["decoder"]),
                "joiner": str(transducer["joiner"]),
                "tokens": str(transducer["tokens"]),
            }
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(transducer["encoder"]),
                decoder=str(transducer["decoder"]),
                joiner=str(transducer["joiner"]),
                tokens=str(transducer["tokens"]),
                num_threads=num_threads,
                provider="cpu",
                sample_rate=self.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=1.2,
                rule2_min_trailing_silence=0.35,
                rule3_min_utterance_length=8.0,
                decoding_method="greedy_search",
                model_type="zipformer",
            )

        zipformer_ctc = self._find_zipformer2_ctc_bundle(root)
        if zipformer_ctc is not None:
            num_threads = max(2, min(4, (os.cpu_count() or 4) // 2))
            self._diag["model_type"] = "zipformer2_ctc"
            self._diag["model_files"] = {
                "model": str(zipformer_ctc["model"]),
                "tokens": str(zipformer_ctc["tokens"]),
            }
            return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=str(zipformer_ctc["model"]),
                tokens=str(zipformer_ctc["tokens"]),
                num_threads=num_threads,
                provider="cpu",
                sample_rate=self.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=1.2,
                rule2_min_trailing_silence=0.35,
                rule3_min_utterance_length=8.0,
                decoding_method="greedy_search",
            )

        raise FileNotFoundError(
            "No compatible STT model files found. Expected one of: "
            "(encoder*.onnx + decoder*.onnx + joiner*.onnx + tokens.txt) or "
            f"(ctc*.onnx + tokens.txt) under: {root}"
        )

    def _choose_preferred_model(self, files, prefer_int8=True):
        if not files:
            return None

        def score(path_obj: Path):
            name = path_obj.name.lower()
            s = 0
            if ".int8" in name and prefer_int8:
                s += 30
            if ".int8" not in name and not prefer_int8:
                s += 30
            if "-left-" in name:
                s += 5
            # Prefer shorter names when scores tie; usually indicates canonical export names.
            s -= len(name) * 0.001
            return s

        return max(files, key=score)

    def _bundle_priority(self, bundle_path: Path):
        name = str(bundle_path).lower()
        score = 0
        # Prefer the bilingual zh-en model the user asked for.
        if "bilingual-zh-en-2023-02-20" in name:
            score += 1000
        if "bilingual" in name:
            score += 200
        if "zh-en" in name:
            score += 100
        # Prefer int8 export bundles for lower latency.
        if "int8" in name:
            score += 50
        # Slightly prefer shorter paths as a tie-breaker.
        score -= len(name) * 0.001
        return score

    def _find_transducer_bundle(self, root: Path):
        tokens_candidates = sorted(root.rglob("tokens.txt"), key=self._bundle_priority, reverse=True)
        for tokens in tokens_candidates:
            parent = tokens.parent
            encoders = sorted(parent.glob("encoder*.onnx"))
            decoders = sorted(parent.glob("decoder*.onnx"))
            joiners = sorted(parent.glob("joiner*.onnx"))
            if encoders and decoders and joiners:
                return {
                    "tokens": tokens,
                    "encoder": self._choose_preferred_model(encoders, prefer_int8=True),
                    "decoder": self._choose_preferred_model(decoders, prefer_int8=False),
                    "joiner": self._choose_preferred_model(joiners, prefer_int8=True),
                }
        return None

    def _find_zipformer2_ctc_bundle(self, root: Path):
        tokens_candidates = sorted(root.rglob("tokens.txt"), key=self._bundle_priority, reverse=True)
        for tokens in tokens_candidates:
            parent = tokens.parent
            ctc_models = sorted(parent.glob("ctc*.onnx"))
            if ctc_models:
                return {
                    "tokens": tokens,
                    "model": self._choose_preferred_model(ctc_models),
                }
        return None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def feed_audio(self, audio_data):
        """Feed a chunk of audio (numpy array, float32, range -1..1)."""
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            self._diag["queue_drops"] += 1
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass

    def get_diagnostics(self):
        d = dict(self._diag)
        d["running"] = bool(self._running)
        d["audio_queue_size"] = int(self.audio_queue.qsize())
        d["text_queue_size"] = int(self.text_queue.qsize())
        return d

    def get_transcript(self, block=False, timeout=None):
        """Get latest transcribed text chunk."""
        try:
            return self.text_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _process_loop(self):
        error_streak = 0

        def emit_final_and_reset_stream():
            # Flush buffered states so the last words of an utterance are emitted.
            try:
                self.stream.input_finished()
                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)
            except Exception:
                pass

            final_text = str(self.recognizer.get_result(self.stream)).strip()
            if final_text:
                self._diag["final_emits"] += 1
                self._diag["last_text"] = final_text
                try:
                    self.text_queue.put_nowait(final_text)
                except queue.Full:
                    pass

            self.stream = self.recognizer.create_stream()
            self._last_text = ""
            self._silence_seconds = 0.0
            self._segment_seconds = 0.0
            self._last_partial_emit_time = 0.0

        while self._running:
            try:
                first = self.audio_queue.get(timeout=0.005)
                chunks = [first]
                # Keep micro-batch short to reduce recognition latency.
                while len(chunks) < 2:
                    try:
                        chunks.append(self.audio_queue.get_nowait())
                    except queue.Empty:
                        break

                samples = np.asarray(np.concatenate(chunks), dtype=np.float32)
                if samples.ndim > 1:
                    samples = samples[:, 0]

                duration_s = float(samples.shape[0]) / float(self.sample_rate)
                rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
                self._segment_seconds += duration_s

                self.stream.accept_waveform(self.sample_rate, samples)

                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)

                text = str(self.recognizer.get_result(self.stream)).strip()

                if rms < self._silence_threshold:
                    self._silence_seconds += duration_s
                else:
                    self._silence_seconds = 0.0

                # Emit partial updates at a controlled rate using direct sherpa result text.
                now = time.monotonic()
                if text:
                    text_changed = text != self._last_text
                    emit_due = (now - self._last_partial_emit_time) >= self._partial_emit_interval
                    if text_changed and emit_due:
                        self._last_text = text
                        self._last_partial_emit_time = now
                        self._diag["partial_emits"] += 1
                        self._diag["last_text"] = text
                        try:
                            self.text_queue.put_nowait(text)
                        except queue.Full:
                            pass

                endpoint_reached = False
                if hasattr(self.recognizer, "is_endpoint"):
                    try:
                        endpoint_reached = bool(self.recognizer.is_endpoint(self.stream))
                    except Exception:
                        endpoint_reached = False

                # End-of-utterance handling for continuous usage.
                if (
                    endpoint_reached
                    or self._silence_seconds >= self._utterance_silence_seconds
                    or self._segment_seconds >= self._max_segment_seconds
                ):
                    emit_final_and_reset_stream()
                error_streak = 0
            except queue.Empty:
                continue
            except Exception as e:
                error_streak += 1
                print(f"STT error: {e}")
                self._diag["errors"] += 1
                self._diag["last_error"] = str(e)
                # Recreate stream after decoder errors to recover from bad internal state.
                try:
                    self.stream = self.recognizer.create_stream()
                    self._last_text = ""
                    self._silence_seconds = 0.0
                    self._segment_seconds = 0.0
                    self._last_partial_emit_time = 0.0
                except Exception:
                    pass
                time.sleep(min(1.0, 0.05 * error_streak))
