import os
import queue
import subprocess
import threading
from pathlib import Path


class CppRealtimeBridge:
    """Launch and monitor the standalone C++ hearing-aid engine from Python."""

    def __init__(self, model_root=None, executable=None):
        self.model_root = Path(model_root or os.environ.get(
            "CPP_MODEL_ROOT",
            "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        ))
        self.executable = self._resolve_executable(executable)
        self.process = None
        self.reader_thread = None
        self.running = False
        self.latest_transcript = ""
        self.latest_status = "idle"
        self.output_queue = queue.Queue(maxsize=32)

    def _resolve_executable(self, executable=None):
        if executable:
            return Path(executable)

        env_bin = os.environ.get("CPP_ENGINE_BIN")
        if env_bin:
            return Path(env_bin)

        candidates = []
        base_dir = Path(__file__).resolve().parent / "cpp" / "build"
        candidates.append(base_dir / "hearing_aid_realtime")
        candidates.append(base_dir / "hearing_aid_realtime.exe")
        candidates.append(Path("cpp") / "build" / "hearing_aid_realtime")
        candidates.append(Path("cpp") / "build" / "hearing_aid_realtime.exe")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def start(self):
        if self.running:
            return

        if not self.executable.exists():
            raise FileNotFoundError(f"C++ engine not found: {self.executable}")

        cmd = [str(self.executable), str(self.model_root)]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self.running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def stop(self):
        self.running = False
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)

    def _emit(self, message):
        try:
            self.output_queue.put_nowait(message)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(message)
            except queue.Empty:
                pass

    def _read_loop(self):
        if self.process is None or self.process.stdout is None:
            self.running = False
            return

        while self.running:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    break
                continue

            text = line.strip()
            if not text:
                continue

            self.latest_status = text
            self._emit(text)
            print(f"[CPP] {text}")

            if text.startswith("[STT]"):
                transcript = text.split("[STT]", 1)[-1].strip()
                if transcript:
                    self.latest_transcript = transcript

    def get_diagnostics(self):
        return {
            "running": bool(self.running),
            "executable": str(self.executable),
            "model_root": str(self.model_root),
            "latest_status": self.latest_status,
            "latest_transcript": self.latest_transcript,
            "output_queue_size": int(self.output_queue.qsize()),
        }
