<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->

## Default Conda Environment
- Environment name: `rc-llm-eval`
- Environment path: `/home/xuelin/miniconda3/envs/rc-llm-eval`
- Prefer running Python commands with `conda run -n rc-llm-eval ...` or `/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python`.

<!-- codex-agent-runtime:start -->

## Runtime Ports And Database Configuration

- Keep this section aligned with the root README when database names, ports, or service defaults change.
- Do not copy secrets from local `.env` files into commits; document only placeholders or compose defaults.

### Database
- Core benchmark pipeline does not require a database.
- The optional annotation system uses PostgreSQL.
- Annotation database name: `railway_annotation`.
- Default PostgreSQL host and port: `localhost:5432`.
- Default annotation DB user: `deipss`; password must come from local `.env` and must not be committed.
- Annotation RAG helper uses local Ollama by default at `http://127.0.0.1:11434`.

### Default Ports
- Annotation backend FastAPI service: `8000`.
- Annotation frontend local Vite port: `5173`; project frontend mode also uses `4005` for local/FRP exposure.
- PostgreSQL: `5432`.
- Ollama: `11434`.

### Notes For Codex Agents
- This repository may contain large local datasets and generated outputs; do not stage unrelated data changes with documentation edits.
- Before committing, check `git status --short --branch` and avoid staging unrelated runtime artifacts.

### Source Files Checked
- `annotation_system/backend/.env.example`
- `annotation_system/backend/app/config.py`
- `annotation_system/frontend/vite.config.js`
- `annotation_system/README.md`

<!-- codex-agent-runtime:end -->

## GitHub Commit Language

- Use English for all GitHub commit messages and pull/push related commit notes.
