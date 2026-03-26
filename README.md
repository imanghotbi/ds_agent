# ds-agent

CLI-first data science agent that executes numbered scenarios from disk.

## Scenario layout

Create a `scenario/` directory with numbered subdirectories:

```text
scenario/
  1/
    prompt.md
    data/
      dataset.csv
  2/
    prompt.md
```

Each scenario runs in its own fresh sandbox session with session ID `session_id_<foldername>`.

Outputs are written into each scenario folder:

```text
scenario/1/output/
  metadata.json
  state.json
  transcript.md
  final_analysis.ipynb
  sandbox_artifacts/
```

`metadata.json` is updated during execution so interrupted runs keep their latest status and timing data on disk.

## Run

```bash
python3 src/main.py
python3 src/main.py --scenario-dir /path/to/scenario
python3 src/main.py --force
```

Completed scenarios are skipped by default. Use `--force` to re-run them.

## MongoDB

MongoDB is used to persist LLM call metadata such as:

- `node_name`
- `session_id`
- `model_name`
- token usage fields returned by the model provider

Relevant environment variables:

- `MONGO_URI`
- `MONGO_DATABASE`
- `MONGO_COLLECTION`
- `MONGO_ENABLED`
- `MONGO_LOGS_TIMEOUT_MS`

## Docker Compose

Start MongoDB and mongo-express with:

```bash
docker compose up -d
```

Set these environment variables before starting:

- `MONGO_ROOT_USER`
- `MONGO_ROOT_PASSWORD`
