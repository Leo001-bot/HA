from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import threading
import queue


app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# Shared references (set by main.py)
config = None
audio_meter_queue = None
transcription_queue = None
quality_queue = None
latest_transcription = None
latest_transcription_seq = 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    if config is None:
        return jsonify({}), 503
    return jsonify(config.get_all())


@app.route('/api/config', methods=['POST'])
def set_config():
    if config is None:
        return jsonify({'error': 'config not ready'}), 503
    data = request.get_json(silent=True) or {}
    for key, value in data.items():
        config.set(key, value)
    return jsonify({'status': 'ok'})


@app.route('/api/upload_audiogram', methods=['POST'])
def upload_audiogram():
    return jsonify({'status': 'received'})


def background_meter_updater():
    """Push meter levels to WebSocket clients."""
    while True:
        try:
            meter = audio_meter_queue.get(timeout=0.1)
            socketio.emit('meter', {'left': float(meter[0]), 'right': float(meter[1])})
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Meter updater error: {e}")


def background_transcription_updater():
    """Push transcriptions to WebSocket clients."""
    global latest_transcription, latest_transcription_seq
    while True:
        try:
            text = transcription_queue.get(timeout=0.1)
            if text:
                latest_transcription = str(text)
                latest_transcription_seq += 1
                socketio.emit('transcription', {'text': text})
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Transcription updater error: {e}")


def background_quality_updater():
    """Push processing effectiveness metrics to WebSocket clients."""
    while True:
        try:
            payload = quality_queue.get(timeout=0.1)
            socketio.emit('quality', payload)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Quality updater error: {e}")


@app.route('/api/transcription/latest', methods=['GET'])
def get_latest_transcription():
    return jsonify({
        'text': latest_transcription or '',
        'seq': latest_transcription_seq,
    })


@socketio.on('connect')
def handle_connect():
    if latest_transcription:
        socketio.emit('transcription', {'text': latest_transcription})


def start_server(host='0.0.0.0', port=5000, config_obj=None, meter_q=None, trans_q=None, quality_q=None):
    global config, audio_meter_queue, transcription_queue, quality_queue
    config = config_obj
    audio_meter_queue = meter_q
    transcription_queue = trans_q
    quality_queue = quality_q

    threading.Thread(target=background_meter_updater, daemon=True).start()
    threading.Thread(target=background_transcription_updater, daemon=True).start()
    threading.Thread(target=background_quality_updater, daemon=True).start()

    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
