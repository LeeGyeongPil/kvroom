// Glue between the two engine models and the length x concurrency table the UI draws.
import { planSglang, sglangMaxLen, sglangMaxSeqs } from './sglang.ts'
import type { SglangPlan } from './sglang.ts'
import { maxLenForSeqs, maxSeqsForLen, planVllm } from './vllm.ts'
import type { VllmPlan } from './vllm.ts'
import type { ParsedModel } from './parse.ts'

export type Engine = 'vllm' | 'sglang'

/** Everything either engine needs except the request length. */
export interface Settings {
  model: ParsedModel
  kvDtype: string
  tp: number
  kvBytesPerRank: number
  /** Exact vLLM block count from a log; overrides `kvBytesPerRank` for vLLM. */
  numBlocks?: number
  blockSize: number
  maxNumBatchedTokens: number
  maxConcurrentBatches?: number
  pageSize: number
  swaFullTokensRatio: number
  maxTotalTokens?: number
}

export interface Cell {
  len: number
  seqs: number
  /** All `seqs` requests of `len` tokens fit at once (no preemption). */
  ok: boolean
  /** vLLM: blocks left over (negative = missing). SGLang: full-pool tokens left over. */
  spare: number
}

export const DEFAULT_SEQS: readonly number[] = [1, 2, 4, 8, 16, 32, 64]

/** Row lengths: 4k ... 128k that the model supports, plus the model's own maximum. */
export function defaultLens(maxPosition: number): number[] {
  const base = [4096, 8192, 16384, 32768, 65536, 131072].filter((n) => n < maxPosition)
  return [...base, maxPosition]
}

function vllmInput(s: Settings, len: number) {
  return {
    model: s.model,
    kvDtype: s.kvDtype,
    tp: s.tp,
    blockSize: s.blockSize,
    maxModelLen: len,
    maxNumBatchedTokens: s.maxNumBatchedTokens,
    ...(s.maxConcurrentBatches !== undefined ? { maxConcurrentBatches: s.maxConcurrentBatches } : {}),
    kvBytesPerRank: s.kvBytesPerRank,
    ...(s.numBlocks !== undefined ? { numBlocks: s.numBlocks } : {}),
  }
}

export function vllmPlanAt(s: Settings, len: number): VllmPlan {
  return planVllm(vllmInput(s, len))
}

/** `maxRunningRequests` only matters for linear-layer state; the table passes the column's seqs. */
export function sglangPlanAt(s: Settings, maxRunning?: number): SglangPlan {
  return planSglang({
    model: s.model,
    kvDtype: s.kvDtype,
    tp: s.tp,
    kvBytesPerRank: s.kvBytesPerRank,
    pageSize: s.pageSize,
    swaFullTokensRatio: s.swaFullTokensRatio,
    ...(s.maxTotalTokens !== undefined ? { maxTotalTokens: s.maxTotalTokens } : {}),
    ...(maxRunning !== undefined ? { maxRunningRequests: maxRunning } : {}),
  })
}

/** Whole table: one row per length, one cell per concurrency. */
export function capacityTable(engine: Engine, s: Settings, lens: readonly number[], seqs: readonly number[]): Cell[][] {
  if (engine === 'vllm') {
    return lens.map((len) => {
      const p = vllmPlanAt(s, len)
      return seqs.map((n) => ({ len, seqs: n, ok: p.numBlocks >= p.blocksPerRequest * n, spare: p.numBlocks - p.blocksPerRequest * n }))
    })
  }
  return lens.map((len) =>
    seqs.map((n) => {
      const p = sglangPlanAt(s, n)
      const ok = sglangMaxSeqs(p, len) >= n
      return { len, seqs: n, ok, spare: p.fullTokens - len * n }
    }),
  )
}

/** "How many requests of `len` tokens fit?" for the chosen engine. */
export function maxSeqsAt(engine: Engine, s: Settings, len: number): number {
  if (engine === 'vllm') return maxSeqsForLen(vllmInput(s, len), len)
  return sglangMaxSeqs(sglangPlanAt(s), len)
}

/** "How long can each of `seqs` concurrent requests be?" for the chosen engine. */
export function maxLenAt(engine: Engine, s: Settings, seqs: number, cap: number): number {
  if (engine === 'vllm') return maxLenForSeqs(vllmInput(s, cap), seqs, cap)
  return sglangMaxLen(sglangPlanAt(s, seqs), seqs, cap)
}
