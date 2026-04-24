# moca-face-matcher

[![PyPI version](https://img.shields.io/pypi/v/moca-face-matcher.svg)](https://pypi.org/project/moca-face-matcher/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Ultra-fast face matching** library using OpenCV's YuNet for detection, SFace for recognition, and FAISS for high-performance vector search. Designed as both a production-grade SDK and a powerful CLI tool.

---

## Features

- **Blazing Fast**: Integrated with **FAISS** for near-instant search in large databases.
- **State-of-the-Art Models**: Uses YuNet (fastest detection) and SFace (robust recognition).
- **GPU Ready**: Hardware acceleration support via OpenCV DNN/CUDA.
- **Developer First**: Clean SDK interface with strict type hinting (Mypy) and JSON-based secure cache.
- **Secure by Design**: Automatic SHA-256 model verification and no insecure serialization (no `pickle`).
- **Portable Cache**: Uses relative paths in the JSON cache, making it portable across different machines.

## Requirements

- Python 3.9 or superior.
- Internet connection for the **first run** (downloads ~40 MB of ONNX models from [OpenCV Zoo](https://github.com/opencv/opencv_zoo)).

## Installation

```bash
pip install moca-face-matcher
```

*For local development:*
```bash
git clone https://github.com/Moca9801/face-match.git
cd face-match
pip install -e ".[dev]"
```

## Usage

### Command Line Interface (CLI)

Search for a face in a gallery folder:
```bash
face-match path/to/query.jpg --db path/to/gallery/
```

### Library Usage (SDK)

The core function is `find_matches`. It returns a structured dictionary and does not print to the console, making it ideal for integration into other systems.

```python
from pathlib import Path
from face_match import find_matches

results = find_matches(
    query=Path("query.jpg"),
    db=Path("./gallery"),
    top=3,
    threshold=0.4
)

print(f"Scanned {results['total_scanned']} images.")
for match in results['results']:
    print(f"Match found: {match['path']} (Distance: {match['distance']})")
```

### CLI vs SDK Mapping

| CLI Option | SDK Parameter (`find_matches`) | Description |
| :--- | :--- | :--- |
| `query` | `query` (Path) | Path to the query image. |
| `--db` | `db` (Path) | Path to the gallery/database folder. |
| `--top` | `top` (int) | Number of results to return. |
| `--metric` | `distance` (int) | `0` for Cosine (default), `1` for L2. |
| `--threshold`| `threshold` (float)| Similarity threshold. |
| `--device` | `device` (str) | `cpu` (default) or `gpu`. |
| `--rebuild` | `rebuild_cache` (bool) | Force rescan of all images. |

---

## Technical Details

### Metrics and Thresholds
The library uses predefined thresholds based on the SFace model's official recommendations:
- **Cosine Similarity** (`--metric coseno`): Matches range from `-1` to `1`. Higher is better. Recommended threshold: `0.363`.
- **L2 Distance** (`--metric l2`): Matches start from `0`. Lower is better. Recommended threshold: `1.128`.

### Development and Testing
Contributors are welcome! To run quality checks locally:
```bash
# Run Linting
ruff check src tests

# Run Type Checking
mypy src/face_match

# Run Tests
pytest
```

---

## Security and Data Privacy

> [!CAUTION]
> **Biometric Data on Disk**: The `.face_embeddings_cache.json` file generated in your database folder contains mathematical representations (embeddings) and **relative file paths**.
> - **Do not share** this file.
> - Treat it as sensitive personal data.
> - Access to the file should be restricted using system-level permissions.

## Disclaimer

- This software is **not** a certified biometric identification system.
- It is intended for research and low-risk identification tasks.
- The author is not responsible for misuse or legal implications of using biometric data.

---

# Versión en Español (Resumen)

**moca-face-matcher** es una librería de búsqueda de coincidencias faciales ultra rápida.

## Instalación rápida
```bash
pip install moca-face-matcher
```

## Características
- Búsqueda vectorial con FAISS.
- Modelos YuNet + SFace.
- Soporte para GPU.
- Caché segura en JSON con rutas relativas.

## Seguridad
Este software maneja datos biométricos. El archivo de caché `.face_embeddings_cache.json` es sensible y **no debe ser compartido**. 

---

## License
Distributed under the **MIT License**. See `LICENSE` for more information.
