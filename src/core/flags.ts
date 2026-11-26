import { sglangDtypeName, vllmDtypeName } from './dtype.ts'

export interface FlagOptions {
  /** Hugging Face model id, for example `openai/gpt-oss-20b`. */
  model: string
  tp: number
  maxModelLen: number
  /** Concurrent requests to allow (`--max-num-seqs` / `--max-running-requests`). */
  maxSeqs: number
  /** `auto`, `fp8`, `fp8_e4m3` or `fp8_e5m2`. */
  kvDtype: string
  /** vLLM `--max-num-batched-tokens`. */
  maxNumBatchedTokens?: number
  /** vLLM `--kv-cache-memory-bytes` / SGLang `--max-total-tokens` pin the KV size so the plan is deterministic. */
  kvCacheMemoryBytes?: number
  /** vLLM `--block-size`, only printed when it is not the default 16. */
  blockSize?: number
  /** SGLang `--page-size`, only printed when it is not the default 1. */
  pageSize?: number
  /** SGLang `--swa-full-tokens-ratio`, printed for hybrid models. */
  swaFullTokensRatio?: number
  /** SGLang `--max-total-tokens`. */
  maxTotalTokens?: number
}

/** A ready-to-paste `vllm serve` command line. */
export function vllmFlags(o: FlagOptions): string {
  const parts = [
    `vllm serve ${o.model}`,
    `--tensor-parallel-size ${o.tp}`,
    `--max-model-len ${o.maxModelLen}`,
    `--max-num-seqs ${o.maxSeqs}`,
  ]
  const dtype = vllmDtypeName(o.kvDtype)
  if (dtype) parts.push(`--kv-cache-dtype ${dtype}`)
  if (o.maxNumBatchedTokens !== undefined) parts.push(`--max-num-batched-tokens ${o.maxNumBatchedTokens}`)
  if (o.kvCacheMemoryBytes !== undefined) parts.push(`--kv-cache-memory-bytes ${Math.floor(o.kvCacheMemoryBytes)}`)
  if (o.blockSize !== undefined && o.blockSize !== 16) parts.push(`--block-size ${o.blockSize}`)
  return parts.join(' ')
}

/** A ready-to-paste `sglang.launch_server` command line. */
export function sglangFlags(o: FlagOptions): string {
  const parts = [
    `python -m sglang.launch_server --model-path ${o.model}`,
    `--tp-size ${o.tp}`,
    `--context-length ${o.maxModelLen}`,
    `--max-running-requests ${o.maxSeqs}`,
  ]
  const dtype = sglangDtypeName(o.kvDtype)
  if (dtype) parts.push(`--kv-cache-dtype ${dtype}`)
  if (o.swaFullTokensRatio !== undefined) parts.push(`--swa-full-tokens-ratio ${o.swaFullTokensRatio}`)
  if (o.pageSize !== undefined && o.pageSize !== 1) parts.push(`--page-size ${o.pageSize}`)
  if (o.maxTotalTokens !== undefined) parts.push(`--max-total-tokens ${Math.floor(o.maxTotalTokens)}`)
  return parts.join(' ')
}
