# ML chord engine setup (CHORD_ENGINE=ml)

Requires **Python 3.11+**. Create a venv with:

```bash
cd backend
./scripts/setup_venv.sh
source .venv/bin/activate
```

## 1. System dependencies (macOS)

```bash
brew install vamp-plugin-sdk
```

## 2. NNLS-Chroma VAMP plugin (required — bundled .so is Linux-only)

Download the **macOS** NNLS-Chroma plugin from the [Vamp Plugin Pack](https://vamp-plugins.org/pack.html) or [NNLS-Chroma project files](https://code.soundsoftware.ac.uk/projects/nnls-chroma/files).

Copy the `.dylib` and `.cat` files into:

```
~/Library/Audio/Plug-Ins/Vamp/
```

Verify with:

```bash
ls ~/Library/Audio/Plug-Ins/Vamp/
# Should include nnls-chroma.dylib (or similar), NOT the Linux .so from autochord
```

## 3. Python packages (project venv)

```bash
cd backend
source .venv/bin/activate   # or venv2 while migrating
pip install -r requirements-ml.txt

# vamp must be built against Homebrew SDK:
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
pip install vamp --no-build-isolation
pip install autochord --no-deps
pip install lazycats gdown
```

**Important:** autochord requires `tensorflow<2.16` (Keras 2). TensorFlow 2.20+ will fail to load the model.

## 4. Test

```bash
CHORD_ENGINE=ml ./venv/bin/python compare_engines.py
CHORD_ENGINE=ml ./venv/bin/python eval_chords.py --require-all
```

## 5. Enable in app

Set in `.env`:

```
CHORD_ENGINE=ml
```

Falls back to classic automatically if ML fails.
