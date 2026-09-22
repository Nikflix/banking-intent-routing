# Running and maintaining the project

Use Python 3.12 on a 64-bit machine. Training runs on a CPU; no API keys or paid
services are required. Run commands from the repository root. The first NLP run
downloads about 90 MB of encoder weights in addition to Python dependencies.

```bash
git clone https://github.com/Nikflix/banking-intent-routing.git
cd banking-intent-routing
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -e .
python -m intentlab.train
python -m pytest -q
python -m streamlit run app.py
```

The demo opens at http://localhost:8501. The checked-in `reports/metrics.json`
contains the reference run. Training replaces reports and creates local artifacts;
the demo reads those same files. Raw data, weights and fitted estimators are
excluded from Git. Data downloads are checked against fixed SHA-256 hashes.

For the API, use a second terminal with the environment activated:

```bash
python -m uvicorn intentlab.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to submit requests. `GET /health` tests the process;
`GET /ready` also checks that the model was loaded. Missing or corrupt model files
produce HTTP 503; invalid request bodies produce HTTP 422. Responses include a
model hash and request ID. Request bodies are not logged by the API.

## Container configuration

Train locally first. The image excludes data and artifacts; mount the exact
locally trained bundle read-only. Example for a POSIX shell:

```bash
docker build -t banking-intent-routing .
docker run --rm -p 127.0.0.1:8000:8000 -v "${PWD}/artifacts:/models:ro" -e MODEL_DIR=/models -v "${PWD}/data/encoder:/encoder:ro" -e ENCODER_DIR=/encoder banking-intent-routing
```

The API and demos were exercised locally. Docker configuration is supplied, but
the reference execution environment did not have a Docker daemon, so the image
build/run was not tested there. The container is a local deployment example;
authentication, TLS termination and infrastructure monitoring are not included.

## Reproduction and checks

`python -m pytest -q` runs offline unit and API tests. The full-model integration
test runs when local artifacts exist, otherwise it is explicitly skipped. GitHub
Actions runs these tests on pushes and pull requests. Manually running the
**Checks** workflow also trains from source, runs the integration test and uploads
the new model and evaluation as a workflow artifact. For notebooks and linting:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
```

The notebook is an analysis entry point. Source modules own model training;
running notebook cells does not silently retune the model. Different hardware,
Python builds or numerical libraries can change timings and artifact hashes.
Report the run environment alongside any new metrics.

## Model changes

Keep the current bundle and report together before retraining. Review data
changes, split integrity, predictive performance and known failure cases, then
restart the service with the approved bundle. Roll back by restoring both the
model and manifest; never edit the manifest to make a modified model load. A
checksum detects changed bytes, not malicious provenance. Only load locally
trained or otherwise trusted joblib artifacts.
