// Bundled config snapshots (structure numbers only) used when the Hugging Face Hub is unreachable or
// VITE_KVROOM_OFFLINE=1. This is a practice stand-in for the Hub fetch: the calculation that follows is
// exactly the same. Config values were copied 2026-11-27 from the URLs in `source`.
// `weightBytes` is approximate (parameter count x dtype), not a file-size sum: edit it in the UI.
// Generated from the test fixtures; the model files themselves are not included.

export interface Preset {
  /** Short key used in `?preset=` URLs. */
  key: string
  /** Hugging Face repo shown to the user (and used in the printed serve command). */
  repo: string
  name: string
  config: Record<string, unknown>
  weightBytes: number
  source: string
  checked: string
}

export const PRESETS: readonly Preset[] = [
  {
    key: "llama-3.1-8b",
    repo: "meta-llama/Llama-3.1-8B-Instruct",
    name: "Llama-3.1-8B",
    config: {"head_dim": 128, "hidden_size": 4096, "max_position_embeddings": 131072, "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8, "torch_dtype": "bfloat16"},
    weightBytes: 16060000000,
    source: "https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "qwen3-8b",
    repo: "Qwen/Qwen3-8B",
    name: "Qwen3-8B",
    config: {"head_dim": 128, "hidden_size": 4096, "max_position_embeddings": 40960, "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8, "sliding_window": null, "torch_dtype": "bfloat16", "use_sliding_window": false},
    weightBytes: 16400000000,
    source: "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "llama-3.3-70b",
    repo: "meta-llama/Llama-3.3-70B-Instruct",
    name: "Llama-3.3-70B",
    config: {"torch_dtype": "bfloat16", "head_dim": 128, "hidden_size": 8192, "max_position_embeddings": 131072, "num_attention_heads": 64, "num_hidden_layers": 80, "num_key_value_heads": 8},
    weightBytes: 141200000000,
    source: "https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "gemma-3-27b",
    repo: "google/gemma-3-27b-it",
    name: "Gemma-3-27B",
    config: {"text_config": {"head_dim": 128, "hidden_size": 5376, "max_position_embeddings": 131072, "num_attention_heads": 32, "num_hidden_layers": 62, "num_key_value_heads": 16, "sliding_window": 1024, "sliding_window_pattern": 6, "torch_dtype": "bfloat16"}},
    weightBytes: 54800000000,
    source: "https://huggingface.co/unsloth/gemma-3-27b-it/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "gemma-4-31b",
    repo: "google/gemma-4-31B-it",
    name: "Gemma-4-31B",
    config: {"text_config": {"attention_k_eq_v": true, "dtype": "bfloat16", "global_head_dim": 512, "head_dim": 256, "hidden_size": 5376, "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"], "max_position_embeddings": 262144, "num_attention_heads": 32, "num_global_key_value_heads": 4, "num_hidden_layers": 60, "num_key_value_heads": 16, "sliding_window": 1024}},
    weightBytes: 62000000000,
    source: "https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "gpt-oss-20b",
    repo: "openai/gpt-oss-20b",
    name: "gpt-oss-20b",
    config: {"head_dim": 64, "hidden_size": 2880, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"], "max_position_embeddings": 131072, "num_attention_heads": 64, "num_hidden_layers": 24, "num_key_value_heads": 8, "sliding_window": 128},
    weightBytes: 13760000000,
    source: "https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "deepseek-v3",
    repo: "deepseek-ai/DeepSeek-V3",
    name: "DeepSeek-V3",
    config: {"hidden_size": 7168, "kv_lora_rank": 512, "max_position_embeddings": 163840, "num_attention_heads": 128, "num_hidden_layers": 61, "num_key_value_heads": 128, "qk_rope_head_dim": 64, "torch_dtype": "bfloat16"},
    weightBytes: 685000000000,
    source: "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/config.json",
    checked: '2026-11-27',
  },
  {
    key: "qwen3-next-80b",
    repo: "Qwen/Qwen3-Next-80B-A3B-Instruct",
    name: "Qwen3-Next-80B",
    config: {"full_attention_interval": 4, "head_dim": 256, "hidden_size": 2048, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_value_head_dim": 128, "max_position_embeddings": 262144, "num_attention_heads": 16, "num_hidden_layers": 48, "num_key_value_heads": 2, "torch_dtype": "bfloat16", "use_sliding_window": false},
    weightBytes: 159000000000,
    source: "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/resolve/main/config.json",
    checked: '2026-11-27',
  },
]

export function findPreset(idOrKey: string): Preset | undefined {
  const q = idOrKey.trim().toLowerCase()
  return PRESETS.find((p) => p.key === q || p.repo.toLowerCase() === q)
}
