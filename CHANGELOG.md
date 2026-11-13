# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-13

### Added
- `specpair check TARGET DRAFT`: pre-load verdict for the `llama.cpp@b12360` rule set from GGUF headers or
  Hugging Face tokenizer folders, listing every failing rule with the first mismatching tokens.
- `specpair scan TARGET DIR`: verdicts for every GGUF in a folder; corrupt files become error rows.
- `specpair bench`: tokens/s, acceptance rate and speedup from llama-server timings, with warnings for a draft
  that is not used and for settings that are slower than the baseline.
- `specpair demo` and `bench --mock`: offline practice mode with synthetic GGUF headers and replayed timings.
- English and Korean output (`--lang`, `SPECPAIR_LANG`, locale), `--json` output, stable exit codes.
- Small public Python API: `load_vocab`, `check_pair`, `recommended_command`, exception classes.
