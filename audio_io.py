import queue

import numpy as np
import sounddevice as sd


class AudioIO:
    """Handles real-time capture/playback with small queues."""

    def __init__(self, samplerate=16000, blocksize=64, channels=2):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        # Keep short queues to avoid long buffering delay that sounds like echo.
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.stream = None
        self._running = False

    def start(self):
        self._running = True
        self.stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype=np.float32,
            callback=self._callback,
            latency='low',
        )
        self.stream.start()

    def stop(self):
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        if self._running:
            try:
                self.input_queue.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest chunk to keep latency bounded.
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.input_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass

        # Play the newest available output frame and drop stale frames.
        latest = None
        while True:
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            outdata.fill(0)
        else:
            outdata[:] = latest

    def get_input(self, block=True, timeout=None):
        return self.input_queue.get(block=block, timeout=timeout)

    def put_output(self, data):
        try:
            self.output_queue.put_nowait(data)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(data)
            except queue.Empty:
                pass
