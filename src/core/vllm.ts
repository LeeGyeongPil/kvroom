// vLLM block-counting rules, re-written from the Apache-2.0 sources (read 2026-11-27):
//   vllm/v1/kv_cache_interface.py        page size per layer
//   vllm/v1/core/kv_cache_planning.py    TP replication, page unification, grouping, fit check
//   vllm/v1/core/single_type_kv_cache_manager.py   sliding-window blocks per request
//   vllm/config/vllm.py                  in-flight tokens = max_concurrent_batches * max_num_batched_tokens
// Rules follow vLLM 0.34.0. Log token counts follow v0.12.0 (and the current definition, see below).
import { kvDtypeBytes } from './dtype.ts'
import { KvroomError } from './errors.ts'
import { fixedStateBytes } from './parse.ts'
import type { LayerKind, LayerSpec, ParsedModel } from './parse.ts'
import { bytesPerTokenPerRank, headsPerRank } from './tp.ts'

export const VLLM_RULES_VERSION = '0.34.0'
export const GIB = 1024 ** 3

export interface VllmInput {
  model: ParsedModel
  /** `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. */
  kvDtype: string
  tp: number
  /** Tokens per block (vLLM default 16). */
  blockSize: number
  maxModelLen: number
  /** Sliding-window layers keep `window - 1` plus this many in-flight tokens. */
  maxNumBatchedTokens: number
  /** 1 unless async scheduling or pipeline parallelism overlaps batches. */
  maxConcurrentBatches?: number
  /** KV cache bytes available on one rank. */
  kvBytesPerRank: number
  /** Use this block count instead of `kvBytesPerRank / bytesPerBlock` (log calibration). */
  numBlocks?: number
}

export interface VllmGroup {
  kind: LayerKind
  window?: number
  /** Real layers in this group (the rest of the group size is padding). */
  layers: number
  blocksPerRequest: number
}

export interface VllmPlan {
  groupSize: number
  numGroups: number
  groups: VllmGroup[]
  /** Group slots that hold no layer (memory vLLM reserves but cannot use). */
  paddingSlots: number
  bytesPerBlock: number
  numBlocks: number
  /** Blocks one `maxModelLen` request holds in the worst case. */
  blocksPerRequest: number
  /** Blocks of the request that come from fixed linear-layer state (estimate). */
  stateBlocksPerRequest: number
  /** Block pool size divided by `blocksPerRequest` (the "Maximum concurrency" in vLLM's start-up log). */
  maxConcurrency: number
  /** "GPU KV cache size" as vLLM v0.12 logged it: `numBlocks // numGroups * blockSize`. */
  kvTokensV012: number
  /** "GPU KV cache size" as current vLLM logs it: `maxConcurrency * maxModelLen`. */
  kvTokensCurrent: number
  /** True when a layer's page size does not divide the largest one (vLLM would refuse; numbers are rough). */
  pageMismatch: boolean
}

const cdiv = (a: number, b: number): number => Math.ceil(a / b)

function bucketKey(l: LayerSpec): string {
  return `${l.kind}:${l.window ?? 0}`
}

/** Split the model's attention layers into vLLM's "n:1" groups and count blocks. */
export function planVllm(input: VllmInput): VllmPlan {
  const { model, tp, blockSize, maxModelLen } = input
  const elem = kvDtypeBytes(input.kvDtype, model.modelDtypeBytes)
  const attention = model.layers.filter((l) => l.kind !== 'linear')
  if (attention.length === 0) {
    throw new KvroomError('CONFIG_INVALID', 'The model has no attention layers to plan a KV cache for.', {
      field: 'layers',
    })
  }

  const buckets = new Map<string, LayerSpec[]>()
  for (const l of attention) {
    const key = bucketKey(l)
    buckets.set(key, [...(buckets.get(key) ?? []), l])
  }
  const list = [...buckets.values()]
  const groupSize = Math.min(...list.map((v) => v.length))

  // Page = bytes one layer needs for one block. Smaller pages get a bigger block so all pages match.
  const pages = list.map((v) => {
    const l = v[0] as LayerSpec
    return headsPerRank(l, tp) * blockSize * (l.kDim + l.vDim) * elem
  })
  const maxPage = Math.max(...pages)
  let pageMismatch = false

  const inFlight = (input.maxConcurrentBatches ?? 1) * input.maxNumBatchedTokens
  const groups: VllmGroup[] = []
  list.forEach((v, i) => {
    const l = v[0] as LayerSpec
    const page = pages[i] as number
    let multiple = maxPage / page
    if (!Number.isInteger(multiple)) {
      pageMismatch = true
      multiple = Math.ceil(multiple)
    }
    const effBlock = blockSize * multiple
    let perGroup: number
    if (l.kind === 'sliding') {
      // SlidingWindowManager: window - 1 tokens plus the tokens being processed, plus one block of slack.
      const tokens = Math.min((l.window ?? 1) - 1 + inFlight, maxModelLen)
      perGroup = cdiv(tokens, effBlock) + 1
    } else {
      perGroup = cdiv(maxModelLen, effBlock)
    }
    const count = cdiv(v.length, groupSize)
    for (let g = 0; g < count; g++) {
      const left = v.length - g * groupSize
      groups.push({
        kind: l.kind,
        ...(l.window !== undefined ? { window: l.window } : {}),
        layers: Math.min(groupSize, left),
        blocksPerRequest: perGroup,
      })
    }
  })

  const bytesPerBlock = groupSize * maxPage
  const numBlocks = input.numBlocks ?? Math.floor(input.kvBytesPerRank / bytesPerBlock)

  // Linear layers (Gated DeltaNet): fixed state per request, estimated as extra blocks.
  const stateBlocksPerRequest = cdiv(fixedStateBytes(model) / tp, bytesPerBlock)
  const attentionBlocks = groups.reduce((s, g) => s + g.blocksPerRequest, 0)
  const blocksPerRequest = attentionBlocks + (fixedStateBytes(model) > 0 ? stateBlocksPerRequest : 0)
  const maxConcurrency = numBlocks / blocksPerRequest
  const numGroups = groups.length

  return {
    groupSize,
    numGroups,
    groups,
    paddingSlots: numGroups * groupSize - attention.length,
    bytesPerBlock,
    numBlocks,
    blocksPerRequest,
    stateBlocksPerRequest: fixedStateBytes(model) > 0 ? stateBlocksPerRequest : 0,
    maxConcurrency,
    kvTokensV012: Math.floor(numBlocks / numGroups) * blockSize,
    kvTokensCurrent: Math.floor(maxConcurrency * maxModelLen),
    pageMismatch,
  }
}

/** Largest whole number of `maxModelLen` requests the pool guarantees without preemption. */
export function maxSeqsForLen(input: VllmInput, len: number): number {
  const p = planVllm({ ...input, maxModelLen: len })
  return Math.floor(p.numBlocks / p.blocksPerRequest)
}

/**
 * Largest `max_model_len` (at most `cap`) for which `seqs` full-length requests fit.
 * Binary search, like vLLM's own estimate in its start-up error. Returns 0 when not even 1 token fits.
 */
export function maxLenForSeqs(input: VllmInput, seqs: number, cap: number): number {
  let lo = 1
  let hi = cap
  let best = 0
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2)
    const p = planVllm({ ...input, maxModelLen: mid })
    if (p.numBlocks >= p.blocksPerRequest * seqs) {
      best = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return best
}

/**
 * Reproduce vLLM's start-up check: one request of `maxModelLen` must fit.
 * @throws KvroomError `KV_TOO_SMALL` with the estimated maximum length in `details.estimatedMaxLen`.
 */
export function assertFitsOne(input: VllmInput): VllmPlan {
  const plan = planVllm(input)
  if (plan.numBlocks >= plan.blocksPerRequest) return plan
  const estimatedMaxLen = maxLenForSeqs(input, 1, input.maxModelLen)
  const neededGiB = (plan.blocksPerRequest * plan.bytesPerBlock) / GIB
  const availableGiB = (plan.numBlocks * plan.bytesPerBlock) / GIB
  throw new KvroomError(
    'KV_TOO_SMALL',
    `To serve at least one request with max length ${input.maxModelLen}, ${neededGiB.toFixed(2)} GiB of KV cache is needed, ` +
      `which is larger than the available ${availableGiB.toFixed(2)} GiB. ` +
      `Based on the available memory, the estimated maximum model length is ${estimatedMaxLen}.`,
    { needed: neededGiB.toFixed(2), available: availableGiB.toFixed(2), maxLen: input.maxModelLen, estimatedMaxLen },
  )
}

/** KV bytes per token if every attention layer kept every token (before window savings). */
export function structuredBytesPerToken(model: ParsedModel, kvDtype: string, tp = 1): number {
  return bytesPerTokenPerRank(model, kvDtypeBytes(kvDtype, model.modelDtypeBytes), tp)
}

/** KV bytes per token by the single dense formula `2 * layers * kvHeads * headDim * bytes`. */
export function denseBytesPerToken(model: ParsedModel, kvDtype: string): number {
  return model.denseElemsPerToken * kvDtypeBytes(kvDtype, model.modelDtypeBytes)
}

/** What a one-formula calculator would say about concurrency for the same memory (for comparison only). */
export function denseConcurrency(input: VllmInput): number {
  const perToken = denseBytesPerToken(input.model, input.kvDtype)
  const totalBytes = input.kvBytesPerRank * input.tp
  return Math.floor(totalBytes / perToken) / input.maxModelLen
}
