"""
Vortex ML — Main Flask Application
Routes, API endpoints, and SocketIO event handlers.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session, send_file, make_response
from flask_socketio import SocketIO
import threading

from data_processor import save_uploaded_file, analyze_dataset, prepare_dataset
from training_engine import (
    create_model, train_model, stop_training, MODEL_REGISTRY,
    parse_weight_filename, WEIGHTS_DIR,
)

app = Flask(__name__)
app.secret_key = "vortex-ml-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

# Cache-busting: append version to static URLs so browser loads fresh JS/CSS
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.context_processor
def inject_cache_buster():
    import time
    return {"cache_version": int(time.time())}

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session state (for single-user simplicity)
_state = {
    "dataset_file": None,
    "dataset_path": None,
    "dataset_info": None,
    "feature_cols": [],
    "target_col": None,
    "model_config": None,
    "training_thread": None,
    "project_name": "VortexProject",
    "last_weights_file": None,
}


# ─────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html")


@app.route("/architect")
def architect_page():
    return render_template("architect.html")


@app.route("/training")
def training_page():
    return render_template("training.html")


# ─────────────────────────────────────────────────────────
# Dataset API
# ─────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = file.filename.lower().endswith((".csv", ".xlsx", ".xls"))
    if not allowed:
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    try:
        filename, filepath = save_uploaded_file(file)
        info = analyze_dataset(filepath)
        _state["dataset_file"] = filename
        _state["dataset_path"] = filepath
        _state["dataset_info"] = info
        return jsonify({"filename": filename, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/analyze", methods=["GET"])
def get_dataset_info():
    if _state["dataset_info"] is None:
        return jsonify({"error": "No dataset uploaded yet"}), 404
    return jsonify({
        "filename": _state["dataset_file"],
        "info": _state["dataset_info"],
    })


@app.route("/api/dataset/configure", methods=["POST"])
def configure_dataset():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    feature_cols = data.get("feature_cols", [])
    target_col = data.get("target_col")

    if not feature_cols:
        return jsonify({"error": "Select at least one feature column"}), 400
    if not target_col:
        return jsonify({"error": "Select a target column"}), 400

    _state["feature_cols"] = feature_cols
    _state["target_col"] = target_col

    return jsonify({"status": "ok", "features": feature_cols, "target": target_col})


# ─────────────────────────────────────────────────────────
# Model API
# ─────────────────────────────────────────────────────────
@app.route("/api/architectures", methods=["GET"])
def list_architectures():
    archs = [
        {"key": "mlp", "name": "Multi-Layer Perceptron", "short": "MLP",
         "desc": "Classic feedforward network. Great for general tabular classification and regression.",
         "icon": "🔵"},
        {"key": "dnn", "name": "Deep Neural Network", "short": "DNN",
         "desc": "Deep feedforward with BatchNorm and Dropout for complex patterns.",
         "icon": "🟣"},
        {"key": "cnn1d", "name": "1D Convolutional Network", "short": "CNN-1D",
         "desc": "Detects local patterns and sequences in feature columns.",
         "icon": "🟢"},
        {"key": "rnn", "name": "Recurrent Neural Network", "short": "RNN",
         "desc": "Processes sequential/time-series data with memory.",
         "icon": "🔴"},
        {"key": "lstm", "name": "Long Short-Term Memory", "short": "LSTM",
         "desc": "Captures long-term dependencies in sequential data.",
         "icon": "🟡"},
        {"key": "gru", "name": "Gated Recurrent Unit", "short": "GRU",
         "desc": "Efficient alternative to LSTM with fewer parameters.",
         "icon": "🟠"},
        {"key": "autoencoder", "name": "Autoencoder", "short": "AE",
         "desc": "Learns compressed feature representations. Good for anomaly detection.",
         "icon": "🔷"},
        {"key": "resnet", "name": "Residual Network", "short": "ResNet",
         "desc": "Skip connections allow very deep networks without vanishing gradients.",
         "icon": "⬛"},
        {"key": "transformer", "name": "Transformer", "short": "TF",
         "desc": "Attention-based architecture for learning feature relationships.",
         "icon": "💎"},
        {"key": "wide_deep", "name": "Wide & Deep Network", "short": "W&D",
         "desc": "Combines memorization (wide) with generalization (deep).",
         "icon": "🌐"},
    ]
    return jsonify(archs)


@app.route("/api/model/configure", methods=["POST"])
def configure_model():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    arch_type = data.get("arch_type")
    layer_sizes = data.get("layer_sizes", [64])
    epochs = data.get("epochs", 50)
    lr = data.get("lr", 0.001)
    batch_size = data.get("batch_size", 32)
    optimizer = data.get("optimizer", "adam")
    activation = data.get("activation", "relu")
    project_name = data.get("project_name", "VortexProject")

    # Early stopping
    early_stopping = data.get("early_stopping", {})

    if arch_type not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown architecture: {arch_type}"}), 400

    _state["project_name"] = project_name
    _state["model_config"] = {
        "arch_type": arch_type,
        "layer_sizes": layer_sizes,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "activation": activation,
        "project_name": project_name,
        "early_stopping": early_stopping,
    }

    return jsonify({"status": "ok", "config": _state["model_config"]})


# ─────────────────────────────────────────────────────────
# Training API
# ─────────────────────────────────────────────────────────
@app.route("/api/training/start", methods=["POST"])
def start_training():
    if not _state["dataset_path"]:
        return jsonify({"error": "No dataset uploaded"}), 400
    if not _state["feature_cols"] or not _state["target_col"]:
        return jsonify({"error": "Dataset not configured (features/target)"}), 400
    if not _state["model_config"]:
        return jsonify({"error": "Model not configured"}), 400

    config = _state["model_config"]

    try:
        # Prepare data
        data = prepare_dataset(
            _state["dataset_path"],
            _state["feature_cols"],
            _state["target_col"],
            batch_size=config["batch_size"]
        )

        # Create model
        model = create_model(
            config["arch_type"],
            config["layer_sizes"],
            data["input_dim"],
            data["output_dim"],
            activation=config.get("activation", "relu")
        )

        training_info = {
            "arch_type": config["arch_type"],
            "layer_sizes": config["layer_sizes"],
            "input_dim": data["input_dim"],
            "output_dim": data["output_dim"],
            "task_type": data["task_type"],
            "train_size": data["train_size"],
            "val_size": data["val_size"],
            "test_size": data["test_size"],
            "epochs": config["epochs"],
            "early_stopping": config.get("early_stopping", {}),
        }

        # Start training in background
        from training_engine import _build_weight_filename

        def run():
            train_model(
                model, data["train_loader"], data["val_loader"],
                data["task_type"], config, socketio
            )
            # Store the weight filename for download
            _state["last_weights_file"] = _build_weight_filename(config)

        thread = socketio.start_background_task(run)
        _state["training_thread"] = thread

        return jsonify({"status": "started", "info": training_info})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/stop", methods=["POST"])
def stop_training_route():
    stop_training()
    return jsonify({"status": "stopped"})


@app.route("/api/state", methods=["GET"])
def get_state():
    """Return current app state for page resumption."""
    return jsonify({
        "has_dataset": _state["dataset_path"] is not None,
        "dataset_file": _state["dataset_file"],
        "has_features": len(_state["feature_cols"]) > 0,
        "feature_cols": _state["feature_cols"],
        "target_col": _state["target_col"],
        "has_model": _state["model_config"] is not None,
        "model_config": _state["model_config"],
        "last_weights_file": _state["last_weights_file"],
    })


# ─────────────────────────────────────────────────────────
# Weights API
# ─────────────────────────────────────────────────────────
@app.route("/api/weights/download", methods=["GET"])
def download_weights_redirect():
    """Redirect to the named download URL."""
    filename = _state.get("last_weights_file")
    if not filename:
        return jsonify({"error": "No weights file available"}), 404
    return jsonify({"redirect": f"/api/weights/file/{filename}", "filename": filename})


@app.route("/api/weights/file/<filename>", methods=["GET"])
def download_weights_file(filename):
    """Download a specific weights file by name."""
    # Ensure it ends with .pt for safety
    if not filename.endswith(".pt"):
        return jsonify({"error": "Invalid filename"}), 400
    filepath = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Weights file not found on disk"}), 404

    with open(filepath, "rb") as f:
        data = f.read()
    response = make_response(data)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Content-Length"] = len(data)
    return response


@app.route("/api/weights/upload", methods=["POST"])
def upload_weights():
    """Upload a .pt weights file and parse its filename to extract architecture config."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt weight files are supported"}), 400

    try:
        # Save the file
        filepath = os.path.join(WEIGHTS_DIR, file.filename)
        file.save(filepath)

        # Parse config from filename
        config = parse_weight_filename(file.filename)

        # Validate architecture exists
        if config["arch_type"] not in MODEL_REGISTRY:
            return jsonify({"error": f"Unknown architecture in weights: {config['arch_type']}"}), 400

        # Store in state
        _state["last_weights_file"] = file.filename
        _state["project_name"] = config.get("project_name", "VortexProject")
        _state["model_config"] = config

        return jsonify({
            "status": "ok",
            "filename": file.filename,
            "config": config,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse weights: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────
# SocketIO events
# ─────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")


# ─────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5050, debug=True)
