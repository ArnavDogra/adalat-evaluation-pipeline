# Adalat Audio Evaluation Pipeline

This is a standalone Python application that evaluates ASR models by executing a fully modular, Windows-compatible pipeline. It performs audio profiling, hybrid quality scoring, Meta MMS language identification, Silero VAD, and ASR transcription via `adalat-ai/whisper-medium-hi-high-lr`, followed by comprehensive normalized metrics calculation.

## Features
- **Cross-Platform:** Works on Windows 10/11 and Linux without modification.
- **Language Routing:** Uses `facebook/mms-lid-256` to identify language. Non-Hindi audio skips ASR to save time while still recording metrics.
- **Detailed Telemetry:** Measures Real Time Factor (RTF) at every stage (Profiling, LID, VAD, ASR, Metrics).
- **Reports:** Generates static HTML evaluation reports, CSV summaries, and timing breakdowns (Pie and Bar charts).

## Repository Structure
```
project/
├── models/
│   ├── mms_lid.py       # Meta MMS-LID logic
│   ├── adalat_asr.py    # Whisper pipeline logic
│   └── vad.py           # Silero VAD logic
├── pipeline/
│   ├── profiling.py     # Audio duration and acoustic quality scoring
│   ├── metrics.py       # WER/CER, text normalization, recall logic
│   └── reporting.py     # Static HTML/PNG report generation
├── tests/               # Unit tests
├── config.py            # Global constants
├── utils.py             # Logging and environment validation
├── run_pipeline.py      # Main CLI entry point
├── sample_config.yaml   # Config file template
├── requirements.txt     # Dependency list
└── README.md
```

## Installation

### 1. Python Version
Ensure you have Python 3.9, 3.10, or 3.11 installed.

### 2. Virtual Environment
Create and activate a virtual environment:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation
Install the required packages using the pinned versions:
```bash
pip install -r requirements.txt
```
*(Note: If you are using a GPU, you should install the CUDA-specific version of PyTorch according to your system via the official PyTorch website before running this command).*

## Usage

### 1. Input Formats
- **Audio:** A directory containing `.wav` or `.mp3` files.
- **Transcripts:** A `.csv` file containing two columns at minimum: `audio` (filename/clip_id) and `sarvam_transcript` (reference text).

### 2. Running the Pipeline via CLI
You can pass arguments directly via the command line:

```bash
python run_pipeline.py --audio_dir "./audio" --transcript_csv "./sarvam.csv" --output_dir "./outputs"
```

Alternatively, you can use a YAML configuration file:
```bash
python run_pipeline.py --config sample_config.yaml
```

### 3. Model Downloads
On the first run, the pipeline will automatically download:
- `facebook/mms-lid-256`
- `adalat-ai/whisper-medium-hi-high-lr`
- `snakers4/silero-vad` (PyTorch Hub)

These will be cached locally in `~/.cache/huggingface/` and `~/.cache/torch/hub/` for subsequent runs.

## Expected Outputs
The pipeline creates the following files in your chosen `--output_dir`:
1. `pipeline.log`: Full execution log containing timestamps, errors, and environment validation.
2. `pipeline_results.csv`: The master dataframe containing all audio, predictions, metrics, and timings.
3. `timing_breakdown.csv`: Aggregate time spent in each module.
4. `timing_pie_chart.png` & `timing_bar_chart.png`: Visual representations of module runtime.
5. `lid_summary.csv`: Aggregated view of detected languages and confidences.
6. `one_minute_benchmark.csv`: Extrapolated runtimes for varying audio lengths based on RTF.
7. `final_report.html`: A static summary report of all findings.

## Troubleshooting
- **CUDA Out of Memory:** Reduce the `chunk_length_s` in `config.py` from `30` to `15`.
- **Files Not Found:** Ensure your `transcript_csv` contains `clip_id`s that match the audio filenames precisely (without the extension).
- **Symlink Warnings (Windows):** HuggingFace may warn about symlinks on Windows. This is harmless; the models will still download and function correctly.
