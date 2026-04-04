# Audio Distortion Fixes: Output Limiter & Feedback Canceller Integration

## Summary
Added a dedicated **OutputLimiter** class to the DSP chain and enhanced **FeedbackCanceller** integration to reduce audio distortion and prevent clipping artifacts.

---

## 1. New OutputLimiter Class (processing.py)

### Purpose
- **Adaptive peak limiting** with intelligent attack/release behavior
- Prevents clipping and distortion by detecting peaks and reducing gain dynamically
- Preserves audio quality better than hard clipping

### Key Features
- **Threshold**: 0.95 (adjustable; clips at 95% of full scale)
- **Attack Time**: 2ms (fast response to peaks)
- **Release Time**: 50ms (smooth recovery to normal gain)
- **Adaptive Gain Control**:
  - Detects overshoot amount intelligently
  - Proportional release based on release phase progress
  - Gain locked between 0.1 and 1.0 to avoid excessive attenuation

### Algorithm
```
For each sample:
  1. If peak > threshold:
     - Calculate overshoot = peak / threshold
     - Set target_gain = 1 / (overshoot + 0.01)
     - Quick attack: gain moves toward target_gain rapidly
  2. If peak < threshold:
     - Increment release counter
     - Gradually return gain to 1.0 over release_time
     - Proportional recovery based on release progress
```

### Parameters
- `threshold`: Limiter threshold (0.5-1.0, default 0.95)
- `attack_ms`: Attack time in milliseconds (default 2.0)
- `release_ms`: Release time in milliseconds (default 50.0)
- `sample_rate`: Audio sample rate (default 16000 Hz)

---

## 2. FeedbackCanceller Integration (main.py)

### Current Implementation
The **FeedbackCanceller** (NLMS adaptive filter) is now fully integrated into the audio processing chain:

#### What it does:
- **Detects feedback/echo** from speaker leakage into the microphone
- **Cancels coherent feedback** using adaptive filtering
- **Handles near-end suppression** for earphone usage (optional)

#### Configuration
```python
feedback_canceller_l = FeedbackCanceller(
    filter_length=1024,      # Adaptive filter tap count
    mu=0.025,                # Step size (learning rate)
    sample_rate=16000        # Audio sample rate
)
feedback_canceller_r = FeedbackCanceller(...)  # Right channel
```

#### Features:
1. **Reference Alignment**: Automatically aligns speaker signal to microphone time-domain
2. **Block Projection**: Removes strong coherent feedback quickly  
3. **Adaptive Step Size**: Adjusts learning rate based on signal stationarity
4. **Residual Suppression**: Detects and suppresses lingering echo
5. **Near-End Mode**: Optional mode that suppresses own voice in earphone feedback scenarios

---

## 3. Output Processing Chain (Updated main.py)

### Flow
```
Mono Input (16kHz, mono or stereo)
  ↓
DC Blocking (remove electrical hum)
  ↓
Feedback Canceller (AEC - optional)
  ↓
Noise Reduction
  ↓
Compression
  ↓
Downward Expansion (suppress quiet noise)
  ↓
[ NEW ] OutputLimiter (adaptive peak limiter)
  ↓
Soft Clipping (tanh saturation)
  ↓
Output (mono duplicated to stereo if needed)
```

### Key Improvements
1. **Adaptive Limiting**: Replaces simple hard clipping with intelligent peak detection
2. **Proportional Scaling**: Stereo channels scaled proportionally to maintain balance
3. **Soft Final Clipper**: Gentle tanh saturation after limiting (not before)
4. **Safe Levels**: Output clipped to [-0.95, 0.95] to prevent digital clipping

---

## 4. Usage & Configuration

### Enable/Disable Limiter
The limiter runs automatically on all output. To adjust behavior, modify threshold:

```python
output_limiter = OutputLimiter(
    threshold=0.95,          # Lower = more aggressive limiting
    attack_ms=2.0,           # Faster attack for transients
    release_ms=50.0,         # Smoother release
)
```

### Feedback Canceller Settings (via config API)
```python
config.set('aec_enabled', True)           # Enable AEC
config.set('aec_strength', 1.0)           # 0.2-3.0 (higher = more aggressive)
config.set('aec_delay_blocks', 4)         # Adjust for output latency
config.set('near_end_suppression_mode', False)  # For earphone mode
config.set('near_end_suppression_strength', 0.6)
```

---

## 5. Testing

### Compile Check
```powershell
python -m py_compile processing.py main.py
```

### Offline STT Test
```python
python << 'EOF'
import wave, numpy as np
from stt import StreamingSTT

# Load test WAV file
with wave.open('models/sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav', 'rb') as f:
    audio_data = np.frombuffer(f.readframes(f.getnframes()), 
                              dtype=np.int16).astype(np.float32) / 32768.0
    sr = f.getframerate()

# Test STT
stt = StreamingSTT(model_root='models', sample_rate=16000)
stt.start()

chunk_size = 800
for i in range(0, audio_data.size, chunk_size):
    stt.feed_audio(audio_data[i:i+chunk_size])

stt.stop()
print("STT test completed")
EOF
```

---

## 6. Expected Improvements

### Distortion Reduction
- ✅ Adaptive gain control prevents harsh clipping
- ✅ Fast attack catches peaks before saturation
- ✅ Smooth release prevents "pumping" artifacts

### Feedback Cancellation
- ✅ NLMS filter removes speaker leakage
- ✅ Adaptive alignment for varying delays
- ✅ Optional near-end mode for earphone scenarios

### Audio Quality
- ✅ Peaks are soft-limited, not hard-clipped
- ✅ Stereo balance maintained during limiting
- ✅ Electrical noise further suppressed by DC blocker + AEC

---

## 7. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Still distorted | Limiter threshold too high | Lower to 0.90 or 0.85 |
| Pumping/breathing sound | Release time too short | Increase `release_ms` to 75-100 |
| Quiet parts cut off | Attack too slow | Decrease `attack_ms` to 1.0 |
| Feedback/echo present | AEC disabled | Set `aec_enabled` to True |
| Earphone sidetone loud | NES mode off | Enable `near_end_suppression_mode` |

---

## Files Modified

1. **processing.py** - Added `OutputLimiter` class (lines ~320-380)
2. **main.py** - Imported `OutputLimiter`, instantiated, and integrated into output chain

---

## References

- **NLMS Filtering**: Adaptive filter for feedback cancellation
- **Peak Limiting**: Industry-standard technique in audio processing
- **Soft Clipping**: Preferred over hard clipping for audio quality
