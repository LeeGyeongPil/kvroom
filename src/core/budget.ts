import { GIB } from './vllm.ts'

export type BudgetSource = 'estimate' | 'fixed' | 'log'

export interface BudgetInput {
  /** Memory of one GPU in GiB. */
  gpuGiB: number
  /** Number of GPUs = tensor-parallel size (pipeline parallelism is not modelled). */
  tp: number
  /** vLLM `--gpu-memory-utilization` (0-1). */
  utilization: number
  /** Total weight bytes of the model (all ranks together). */
  weightBytes: number
  /** GiB per rank set aside for activations, CUDA graphs and non-torch memory. Only an estimate. */
  reserveGiB: number
  /** `--kv-cache-memory-bytes`: when given the KV budget is exactly this, per rank. */
  fixedKvBytes?: number
}

export interface Budget {
  /** KV cache bytes on one rank. */
  kvBytesPerRank: number
  source: BudgetSource
  /** Weight bytes one rank holds (estimate mode). */
  weightBytesPerRank: number
  /** Set by log calibration: the exact block count the engine reported. */
  numBlocks?: number
}

/**
 * KV budget per rank. Fixed mode returns the given bytes (deterministic, recommended for planning);
 * estimate mode is `GPU x utilization - weights / TP - reserve` and can be off by a few GiB.
 */
export function estimateBudget(input: BudgetInput): Budget {
  const weightBytesPerRank = input.weightBytes / input.tp
  if (input.fixedKvBytes !== undefined) {
    return { kvBytesPerRank: Math.max(0, Math.floor(input.fixedKvBytes)), source: 'fixed', weightBytesPerRank }
  }
  const usable = input.gpuGiB * GIB * input.utilization
  const kv = usable - weightBytesPerRank - input.reserveGiB * GIB
  return { kvBytesPerRank: Math.max(0, Math.floor(kv)), source: 'estimate', weightBytesPerRank }
}
