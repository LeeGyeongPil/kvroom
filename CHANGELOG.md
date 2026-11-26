# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-27

### Fixed
- CI: bump workflow actions to Node 24 runtimes, restrict token permissions to read-only contents.
- Bug report template placeholder now matches this project (Node, browser).

## [0.1.0] - 2026-11-27

### Added
- Config parser for Hugging Face `config.json`: nested `text_config`, missing `head_dim`, `layer_types` / `sliding_window_pattern` / `full_attention_interval`, sliding-window-only models, MLA, Gemma 4 global layers, Gated DeltaNet linear layers.
- vLLM 0.34.0 block rules: hybrid layer groups with padding, page-size unification, sliding-window in-flight tokens, TP replication, reverse search for the longest length per concurrency.
- SGLang 0.5.23 pool rules: cell size, split full / sliding-window pools, page size, start-up failure check.
- KV budget from an estimate, from a fixed byte count, or calibrated from a pasted vLLM start-up log (both token-count definitions).
- Length x concurrency table, up to three compared setups, copy-ready `vllm serve` and `sglang.launch_server` commands.
- English and Korean screen text with a language menu (`?lang=`, saved choice, browser detection on first visit).
- Hugging Face Hub loading with gated / missing / offline handling and eight bundled config snapshots.
- Tests that reproduce two public vLLM start-up logs.
