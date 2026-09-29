# Module 1 Report — From Notebook to Production-Ready Service

**Project:** ProdML — NYC Green Taxi Ride Duration Prediction
**Module:** The MLOps Practitioner, Module 1
**Author:** `{{Abdulrahman Gomaa }}`
**Status:** `{{COMPLETED}}`



## 1. Executive summary

This module converts an exploratory NYC TLC Green Taxi ride-duration notebook into an installable Python package and a containerized inference service. The model predicts trip duration in minutes from a pickup–dropoff route (`PU_DO`) and `trip_distance`, using `DictVectorizer` and `LinearRegression`. The implementation separates data preparation, feature engineering, training, prediction, configuration, and API concerns. It also introduces structured JSON logs, request correlation IDs, ONNX export, automated testing, and Docker packaging.


## 2. Baseline and reproducibility

### 2.1 Data and experiment configuration

| Item | Value |
|---|---|
| Data source | NYC TLC Green Taxi Trip Records |
| Dataset month | `{{MONTH_AND_YEAR — e.g., 2023-01}}` |
| Source URL or file | `{{DATASET_URL_OR_LOCAL_FILENAME}}` |
| Raw rows | `{{RAW_ROW_COUNT}}` |
| Rows after cleaning | `{{CLEAN_ROW_COUNT}}` |
| Target | Trip duration, in minutes |
| Target calculation | `(dropoff_datetime - pickup_datetime).total_seconds() / 60` |
| Duration filter | `{{MIN_DURATION}}–{{MAX_DURATION}}` minutes |
| Input features | `PU_DO`, `trip_distance` |
| Categorical encoding | `DictVectorizer` fitted **only** on training rows |
| Model | `LinearRegression` |
| Train / validation split | `{{TRAIN_PERCENT}}%` / `{{VALIDATION_PERCENT}}%` |
| Random seed | `{{RANDOM_STATE}}` |
| Training rows | `{{TRAIN_ROW_COUNT}}` |
| Validation rows | `{{VALIDATION_ROW_COUNT}}` |
| Python / scikit-learn versions | `{{PYTHON_VERSION}}` / `{{SKLEARN_VERSION}}` |

**Preprocessing notes:** `{{DOCUMENT YOUR ACTUAL HANDLING OF NULL VALUES, INVALID DURATIONS, AND ANY OTHER CLEANING. IF YOU CHANGED THE NOTEBOOK, DESCRIBE WHY.}}`

### 2.2 Notebook versus package

The original exploratory implementation is preserved at `notebooks/00-baseline.ipynb`. The refactored pipeline is invoked with `python -m prodml.train` or `prodml-train` and writes `models/model.pkl`.

| Metric | Notebook baseline | Packaged training | Absolute difference |
|---|---:|---:|---:|
| Validation MAE (minutes) | `{{BASELINE_MAE}}` | `{{PACKAGE_MAE}}` | `{{ABS_MAE_DIFFERENCE}}` |
| Validation RMSE (minutes) | `{{BASELINE_RMSE}}` | `{{PACKAGE_RMSE}}` | `{{ABS_RMSE_DIFFERENCE}}` |

**MAE preservation criterion:** absolute difference ≤ **0.05 minutes**.
**Result:** `{{PASS_OR_FAIL}}`
**If it failed:** `{{INVESTIGATION_AND_FIX_OR_NA}}`

**Reproduction commands:**

```bash
# From the repository root, after setting up the environment and downloading the data
python -m prodml.train
```

**Evidence:** `{{LINK_TO_NOTEBOOK_OUTPUT_OR_TRAINING_LOG}}`

## 3. Package design and code quality

The `src/prodml/` package divides responsibilities as follows:

| Component | Responsibility |
|---|---|
| `config.py` | Settings and environment-variable overrides using `pydantic-settings` |
| `data.py` | Parquet loading, duration calculation, cleaning, and splitting |
| `features.py` | Route feature (`PU_DO`) and prediction feature dictionaries |
| `train.py` | Vectorizer fitting, model training, evaluation, and artifact creation |
| `predict.py` | `DurationPredictor`, artifact loading, single/batch prediction, and artifact hash |
| `logging_conf.py` | JSON formatting, correlation-ID context, logger setup, and timing decorator |
| `export.py` | ONNX export, parity validation, and inference benchmarks |
| `api/schemas.py` | Pydantic request and response schemas |
| `api/main.py` | FastAPI lifespan, middleware, error handling, and endpoints |

- **Editable installation verified:** `{{YES_NO_AND_COMMAND_OUTPUT_REFERENCE}}`
- **CLI entry point verified:** `{{YES_NO}}`
- **Ruff:** `{{PASS_FAIL_AND_VERSION}}`
- **Black:** `{{PASS_FAIL_AND_VERSION}}`
- **Pre-commit:** `{{PASS_FAIL_OR_NOT_CONFIGURED}}`
- **Type-hint/decorator notes:** `{{WHAT YOU IMPLEMENTED_AND_ANY_LIMITATIONS}}`

**Evidence:** `{{LINK_TO_COMMIT_OR_CI_OUTPUT}}`

## 4. Structured logging and request tracing

### 4.1 Design

Logs are emitted as JSON to standard output. The application uses standard logging levels and includes `timestamp`, `level`, `logger`, `message`, and `correlation_id` in each application log record. A FastAPI HTTP middleware associates a request ID with request processing via `contextvars` and returns it in the `X-Request-ID` response header. A `@timed` decorator records function execution latency.

| Level | Expected event | Observed? |
|---|---|---|
| `DEBUG` | Feature dictionary/vector information when debug logging is enabled | `{{YES_NO}}` |
| `INFO` | Request start/completion and successful prediction with latency | `{{YES_NO}}` |
| `WARNING` | Trip distance outside the chosen training-range warning threshold | `{{YES_NO}}` |
| `ERROR` | Model load failure or request validation rejection | `{{YES_NO}}` |

**Configured warning threshold:** `{{DISTANCE_THRESHOLD_AND_UNIT}}`
**Log transport:** `{{STDOUT_OR_OTHER}}`
**Correlation-ID policy:** `{{UUID_GENERATED_OR_CLIENT_HEADER_BEHAVIOR}}`

### 4.2 Trace demonstration

Make one valid request and capture its `X-Request-ID` response header. Then find the matching application log records.

```bash
curl -i -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "pickup_location_id": 74,
    "dropoff_location_id": 236,
    "trip_distance": 4.2
  }'
```

| Evidence | Actual result |
|---|---|
| Response `X-Request-ID` | `{{REQUEST_ID}}` |
| `request.started` correlation ID | `{{START_LOG_REQUEST_ID}}` |
| `prediction.served` correlation ID | `{{PREDICTION_LOG_REQUEST_ID}}` |
| `request.completed` correlation ID | `{{COMPLETION_LOG_REQUEST_ID}}` |
| Matching IDs across response and logs? | `{{YES_NO}}` |
| JSON parses successfully (e.g. via `jq`)? | `{{YES_NO}}` |

**Redacted log excerpt:**

```json
{
  "timestamp": "{{ACTUAL_TIMESTAMP}}",
  "level": "INFO",
  "logger": "{{ACTUAL_LOGGER}}",
  "message": "prediction.served",
  "correlation_id": "{{ACTUAL_REQUEST_ID}}",
  "model_version": "{{ACTUAL_MODEL_VERSION}}",
  "latency_ms": "{{ACTUAL_LATENCY_MS}}"
}
```

> The excerpt above is a template. When adding real evidence, preserve JSON types: a numeric `latency_ms` should be a JSON number, not a quoted string.

**Error-path evidence (422 and load failure):** `{{LINK_OR_DESCRIPTION}}`

## 5. Model serialization and runtime comparison

### 5.1 Serialization format comparison

| Format | Human-readable | Cross-language use | Schema-enforced | Safe to load from an untrusted source? | Role in this project |
|---|---|---|---|---|---|
| JSON | Yes | Yes | Not by default; add JSON Schema/Pydantic | Generally safe to parse **with input validation and resource limits** | API requests/responses, logs |
| Protobuf | No | Yes | Yes, via `.proto` definitions | Designed for data parsing; still validate size/content and secure the parser | Comparison only |
| Pickle | No | Python-specific | No external schema | **No — deserialization can execute code** | Trusted Python model artifact |
| ONNX | No | Yes, with compatible runtimes | Standardized typed model graph | Do not assume arbitrary untrusted models are safe; treat as trusted artifacts | Portable model export and runtime benchmark |

**Serving format and rationale:** `{{STATE_WHICH_ARTIFACT_THE_API_ACTUALLY_LOADS_AND_WHY}}`

**Security decision:** Only load Pickle artifacts created and controlled by this project's trusted training process. Never accept an arbitrary uploaded `.pkl` as a model.

### 5.2 Artifacts

| Artifact | Path | Size | SHA-256 (optional for ONNX) |
|---|---|---:|---|
| Pickle | `models/model.pkl` | `{{PICKLE_SIZE_KB}}` KB | `{{PICKLE_SHA256}}` |
| ONNX | `models/model.onnx` | `{{ONNX_SIZE_KB}}` KB | `{{ONNX_SHA256_OR_NA}}` |

### 5.3 Prediction parity

For an apples-to-apples comparison, use the **same fitted `DictVectorizer`** to transform the same validation inputs, then run the resulting numeric matrices through the scikit-learn estimator and exported ONNX estimator.

| Test | Result |
|---|---|
| Validation rows compared | `{{NUMBER_OF_ROWS — target: 500}}` |
| ONNX input dtype | `{{ACTUAL_DTYPE}}` |
| Absolute tolerance (`atol`) | `1e-4` |
| Relative tolerance (`rtol`) | `{{ACTUAL_RTOL — explicitly record library default or chosen value}}` |
| Maximum absolute prediction difference | `{{MAX_ABS_DIFFERENCE}}` minutes |
| `np.allclose` result | `{{PASS_OR_FAIL}}` |

**If parity failed:** `{{INVESTIGATE_DTYPE_FLOAT_PRECISION_UNSUPPORTED_CONVERTERS_OR_CHANGED_PREPROCESSING}}`

**Reproduction command:**

```bash
python -m prodml.export
```

**Evidence:** `{{LINK_TO_PARITY_LOG_OR_TEST_OUTPUT}}`

### 5.4 Inference latency

Benchmark the two runtimes on the same **500 transformed validation rows**. Measure model inference alone in the comparison below; if feature transformation is included, clearly identify it and measure both paths consistently.

| Runtime | Mean (ms/request) | p95 (ms/request) |
|---|---:|---:|
| scikit-learn estimator loaded from Pickle | `{{PICKLE_MEAN_MS}}` | `{{PICKLE_P95_MS}}` |
| ONNX Runtime | `{{ONNX_MEAN_MS}}` | `{{ONNX_P95_MS}}` |

**Benchmark environment:** `{{CPU_OS_PYTHON_VERSION_ONNX_RUNTIME_VERSION}}`
**Measurement conditions:** `{{NUMBER_OF_REPEATS_WARMUP_BATCH_SIZE_AND_WHETHER_VECTORIZATION_IS_EXCLUDED}}`
**Interpretation:** `{{DESCRIBE_YOUR_ACTUAL_RESULT; DO_NOT_ASSUME_ONNX_IS_FASTER}}`

## 6. FastAPI service and validation

The API loads the model at application startup and uses the loaded instance for subsequent requests.

| Endpoint | Expected behavior | Tested status | Evidence |
|---|---|---|---|
| `GET /health` | HTTP 200 only when a model object is available; otherwise non-healthy response | `{{PASS_FAIL}}` | `{{EVIDENCE}}` |
| `GET /metadata` | Version, training date, feature names, framework, artifact hash | `{{PASS_FAIL}}` | `{{EVIDENCE}}` |
| `POST /predict` | One validated request and prediction response | `{{PASS_FAIL}}` | `{{EVIDENCE}}` |
| `POST /predict/batch` | Validated list of trips and matching prediction list | `{{PASS_FAIL}}` | `{{EVIDENCE}}` |
| Invalid `trip_distance: -5` | HTTP 422 with readable validation details | `{{PASS_FAIL}}` | `{{EVIDENCE}}` |
| OpenAPI docs at `/docs` | Working examples and schemas | `{{PASS_FAIL}}` | `{{EVIDENCE}}` |

**Example request:**

```json
{
  "pickup_location_id": 74,
  "dropoff_location_id": 236,
  "trip_distance": 4.2
}
```

**Actual response (replace with a redacted response from your running service):**

```json
{
  "prediction": "{{ACTUAL_PREDICTION}}",
  "model_version": "{{ACTUAL_MODEL_VERSION}}",
  "correlation_id": "{{ACTUAL_CORRELATION_ID}}",
  "latency_ms": "{{ACTUAL_LATENCY_MS}}"
}
```

> Replace placeholder strings with the actual numeric values for `prediction` and `latency_ms` before claiming this is an actual response.

**Startup-loading verification:** `{{HOW_YOU_VERIFIED_THE_MODEL_IS_NOT_LOADED_PER_REQUEST}}`

## 7. Automated testing and quality gates

Tests use `pytest` and cover pure feature functions, data preparation, predictor behavior, FastAPI request/response handling, and Pickle/ONNX parity. Fixtures provide reusable sample inputs and an isolated small trained model. Mocks or monkeypatching isolate API behavior from model calculations.

| Test category | Test file or suite | Result |
|---|---|---|
| Data loading/cleaning and reproducible split | `tests/test_data.py` | `{{PASS_FAIL}}` |
| Route feature, missing categories, zero-distance transformation | `tests/test_features.py` | `{{PASS_FAIL}}` |
| Single/batch prediction, deterministic behavior, unseen route | `tests/test_predict.py` | `{{PASS_FAIL}}` |
| Health, metadata, valid/invalid inputs, batch API, mocked predictor | `tests/test_api.py` | `{{PASS_FAIL}}` |
| Pickle–ONNX parity | `tests/test_serialization.py` | `{{PASS_FAIL}}` |
| Training orchestration (if implemented) | `tests/test_train.py` | `{{PASS_FAIL_NOT_IMPLEMENTED}}` |

**Test command:**

```bash
pytest --cov=src/prodml --cov-report=term-missing --cov-fail-under=80
```

| Measure | Result |
|---|---:|
| Tests collected | `{{TESTS_COLLECTED}}` |
| Passed | `{{TESTS_PASSED}}` |
| Failed | `{{TESTS_FAILED}}` |
| Skipped | `{{TESTS_SKIPPED}}` |
| Measured coverage | `{{TOTAL_COVERAGE_PERCENT}}%` |
| Handbook minimum | 70% |
| Session 1 target used for this project | 80% |
| 80% gate met? | `{{YES_NO}}` |
| Deliberately introduced defect caused a test failure? | `{{YES_NO_AND_WHICH_TEST}}` |

**Untested areas or limitations:** `{{DESCRIBE_MISSING_COVERAGE_OR_NONE_IDENTIFIED}}`
**Evidence:** `{{LINK_TO_PYTEST_OUTPUT_OR_CI_RUN}}`

## 8. Containerization and publishing

### 8.1 Docker design

The production Dockerfile uses a builder stage and a runtime stage. The runtime copies only the required installed package/dependencies and model artifact, exposes port 8000, includes a health check, and runs as a non-root user. `.dockerignore` excludes development and data files that are unnecessary in the build context.

| Check | Actual result |
|---|---|
| Multi-stage build succeeds | `{{YES_NO}}` |
| Runtime user | `{{OUTPUT_OF_DOCKER_EXEC_WHOAMI}}` |
| `GET /health` works inside published image | `{{YES_NO}}` |
| `POST /predict` works from published image | `{{YES_NO}}` |
| Docker Compose startup works | `{{YES_NO}}` |
| Health-check status | `{{HEALTHY_UNHEALTHY_NOT_CHECKED}}` |
| Docker Hub image URL | `{{DOCKER_HUB_IMAGE_URL}}` |
| Versioned tag | `{{DOCKER_IMAGE_TAG — e.g., 0.1.0}}` |
| Third-party pull/run verified | `{{YES_NO_NOT_YET_AND_BY_WHOM}}` |

### 8.2 Image-size comparison

| Build | Size (MB) | Notes |
|---|---:|---|
| Single-stage | `{{SINGLE_STAGE_MB}}` | `{{WHETHER_DEV_DEPS_INCLUDED}}` |
| Multi-stage | `{{MULTI_STAGE_MB}}` | `{{WHETHER_RUNTIME_ONLY}}` |
| Difference | `{{IMAGE_SIZE_DIFFERENCE_MB}}` | `{{PERCENT_CHANGE}}%` |
| With `.dockerignore` | `{{WITH_DOCKERIGNORE_MB}}` | `{{NOTE_BUILD_CONTEXT_VS_FINAL_IMAGE_SIZE}}` |
| Without `.dockerignore` | `{{WITHOUT_DOCKERIGNORE_MB}}` | `{{NOTE_BUILD_CONTEXT_VS_FINAL_IMAGE_SIZE}}` |

> `.dockerignore` commonly reduces *build-context transfer* and prevents accidental inclusion of files, but does not necessarily change final image size if the Dockerfile copies only explicitly selected files. Record build-context sizes separately if that is what changes in your experiment.

**Measurement command:**

```bash
docker image ls prodml-api
# For build-context size, inspect the Docker build output.
```

**Why the observed difference occurred:** `{{YOUR_EXPLANATION}}`
**Evidence:** `{{LINK_TO_BUILD_LOG_OR_SCREENSHOT}}`

## 9. MLOps maturity self-assessment

**Chosen maturity level:** `{{LEVEL_0_TO_4_OR_THE_EXACT_LEVEL_FROM_YOUR_SESSION_SLIDES}}`
**Evidence for that level:** `{{CITE_PROJECT_FACTS: REPRODUCIBLE_TRAINING_PACKAGING_TESTS_CONTAINER_DEPLOYMENT_ETC}}`

**Two-sentence assessment:**

`{{SENTENCE_1: WHAT THIS PROJECT HAS ACHIEVED AND WHY THAT MATCHES THE SELECTED LEVEL.}}`
`{{SENTENCE_2: WHAT SPECIFIC CAPABILITIES ARE STILL MISSING FOR THE NEXT LEVEL, SUCH AS AUTOMATED_PIPELINES_EXPERIMENT_TRACKING_DATA_VERSIONING_CI_CD_OR_MODEL_REGISTRY_WHERE_APPLICABLE.}}`

Keep this assessment tied to the actual maturity definitions in the Session 1 material rather than claiming a level based only on the existence of a Dockerfile or API.

## 10. Challenges, decisions, and lessons learned

| Challenge / decision | Evidence or resolution |
|---|---|
| Preserving notebook metrics after refactoring | `{{WHAT_YOU_CHANGED_OR_CONFIRMED}}` |
| `DictVectorizer` treatment of unseen `PU_DO` pairs | `{{WHAT_YOU_TESTED}}` |
| Choosing a model artifact format for Python serving | `{{YOUR_REASONING}}` |
| ONNX conversion/parity and numerical precision | `{{ANY_COMPATIBILITY_OR_DTYPE_ISSUES}}` |
| Passing request IDs through concurrent requests | `{{HOW_YOU_VALIDATED_CONTEXTVARS}}` |
| FastAPI lifespan and isolated tests | `{{HOW_YOU_TESTED_LOADING_ONCE}}` |
| Multi-stage Docker and non-root execution | `{{WHAT_YOU_LEARNED}}` |
| Other | `{{YOUR_ADDITIONAL_LESSONS}}` |

## 11. Definition of Done

Check an item only when completed and supported by evidence.

- [ ] Baseline notebook runs top to bottom and produces validation MAE/RMSE.
- [ ] Installable package and `prodml-train` work.
- [ ] Packaged MAE differs by no more than ±0.05 from notebook MAE.
- [ ] Data, feature, training, prediction, and configuration concerns are separated.
- [ ] Structured JSON logs contain timestamp, level, logger, message, and correlation ID.
- [ ] Correlation ID matches across response header and request/prediction logs.
- [ ] `models/model.pkl` and `models/model.onnx` have been generated.
- [ ] ONNX parity passes on 500 validation rows at the documented tolerance.
- [ ] Mean/p95 latency comparison is recorded for both runtimes.
- [ ] All four FastAPI endpoints work, and invalid requests return HTTP 422.
- [ ] Model loads at startup, not on every prediction.
- [ ] Tests pass at **≥80% coverage** (also exceeds the handbook's 70% minimum).
- [ ] Multi-stage Docker image runs as non-root and responds to API requests.
- [ ] Single-stage/multi-stage image sizes and `.dockerignore` findings are documented.
- [ ] Version `0.1.0` is published to Docker Hub and can be pulled independently.
- [ ] README gets a new user to a prediction in three commands.
- [ ] One module PR was reviewed by two peers, merged, and `v0.1.0` tagged.

**Reviewer 1:** `{{NAME_OR_GITHUB_HANDLE}}`
**Reviewer 2:** `{{NAME_OR_GITHUB_HANDLE}}`
**PR link:** `{{PR_URL}}`
**Docker Hub link:** `{{DOCKER_HUB_IMAGE_URL}}`
**Release link:** `{{RELEASE_URL}}`

## 12. Next module

Module 2 will build on this **same repository**, extending the packaged model toward an automated and tracked training workflow. Planned additions should be taken from the Module 2 handbook requirements rather than treated as deliverables already completed in Module 1.

**Personal next-step notes:** `{{YOUR_PRIORITIES_FOR_MODULE_2}}`
