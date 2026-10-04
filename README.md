# Sempra Protocol
A high-efficiency, compressed symbolic language and token-reduction framework designed for AI-to-AI interaction, system specification, and zero-loss context compaction.

## Core Philosophy
Natural language is bloated and inefficient for AI code execution and complex project planning. Sempra replaces conversational prose with a dense, structured system of root logograms and vector modifiers to maximize information density per token while retaining strict logical execution capability.

## Directory Layout
- `/prompts/` -> System prompts and engine configurations.
- `/specs/` -> Syntax grammar and core specifications.
- `/examples/` -> Project blueprints and MVP translations.

## Benchmark (`/bench/`)
Measures whether Sempra actually saves tokens on a local LM Studio model, against plain and terse English.

```
python3 bench/benchmark.py --base-url http://<lm-studio-host>:1234 --model <model-id> --runs 3
```

- Modes: `plain`, `terse` (`prompts/terse_english.txt`), `sempra` (`prompts/sempra_engine.txt`), `sempra_fewshot` (`prompts/sempra_engine_fewshot.txt`).
- Tasks live in `bench/tasks.json`. Token counts come from the server's `usage` block. Full outputs go to `bench/results.json`.
- Reports per mode: correctness, mean output tokens (and change vs plain), system+input tokens, and how often output is Sempra-shaped.
