// SGLang pool rules, re-written from the Apache-2.0 source (read 2026-11-27):
//   python/sglang/srt/model_executor/pool_configurator.py   cell size, full/SWA split pools
// Rules follow SGLang 0.5.23.
import { kvDtypeBytes } from './dtype.ts'
import { KvroomError } from './errors.ts'
import { fixedStateBytes } from './parse.ts'
import type { ParsedModel } from './parse.ts'
import { bytesPerTokenPerRank, headsPerRank } from './tp.ts'

export const SGLANG_RULES_VERSION = '0.5.23'

export interface SglangInput {
  model: ParsedModel
  kvDtype: string
  tp: number
  /** KV bytes available on one rank. */
  kvBytesPerRank: number
  /** `--page-size` (SGLang default 1). */
  pageSize: number
  /** `--swa-full-tokens-ratio` (default 0.8): SWA pool tokens divided by full pool tokens. */
  swaFullTokensRatio: number
  /** `--max-total-tokens`: fixes the full pool size and ignores `kvBytesPerRank`. */
  maxTotalTokens?: number
  /** `--max-running-requests`: used to set aside linear-layer state (estimate). */
  maxRunningRequests?: number
}

export interface SglangPlan {
  /** True when the model mixes sliding and full layers, so SGLang keeps two pools. */
  hybrid: boolean
  /** Bytes one token costs per rank across the layers of the full pool (every layer when not hybrid). */
  fullBytesPerToken: number
  /** Bytes one token costs per rank across the sliding layers (0 when not hybrid). */
  swaBytesPerToken: number
  fullTokens: number
  swaTokens: number
  /** Smallest sliding window in the model (hybrid only). */
  window: number
  pageSize: number
}

const floorToPage = (n: number, page: number): number => Math.floor(n / page) * page

/**
 * Size SGLang's token pools.
 * @throws KvroomError `SGLANG_SWA_POOL_TOO_SMALL` when the SWA pool cannot hold `window + page_size` tokens.
 */
export function planSglang(input: SglangInput): SglangPlan {
  const { model, tp, pageSize, swaFullTokensRatio } = input
  const elem = kvDtypeBytes(input.kvDtype, model.modelDtypeBytes)
  const sliding = model.layers.filter((l) => l.kind === 'sliding')
  const attention = model.layers.filter((l) => l.kind !== 'linear')
  const hybrid = sliding.length > 0 && sliding.length < attention.length

  const perToken = (kinds: (l: { kind: string }) => boolean): number =>
    model.layers.filter(kinds).reduce((s, l) => s + headsPerRank(l, tp) * (l.kDim + l.vDim) * elem, 0)

  const swaBytesPerToken = hybrid ? perToken((l) => l.kind === 'sliding') : 0
  const fullBytesPerToken = hybrid ? perToken((l) => l.kind !== 'sliding' && l.kind !== 'linear') : bytesPerTokenPerRank(model, elem, tp)

  let budget = input.kvBytesPerRank
  const state = fixedStateBytes(model)
  if (state > 0 && input.maxRunningRequests) budget -= (state / tp) * input.maxRunningRequests
  budget = Math.max(0, budget)

  let fullTokens: number
  if (input.maxTotalTokens !== undefined) {
    fullTokens = floorToPage(input.maxTotalTokens, pageSize)
  } else {
    const denom = fullBytesPerToken + (hybrid ? swaFullTokensRatio * swaBytesPerToken : 0)
    fullTokens = floorToPage(Math.floor(budget / denom), pageSize)
  }
  const swaTokens = hybrid ? floorToPage(Math.floor(fullTokens * swaFullTokensRatio), pageSize) : 0
  const window = hybrid ? Math.min(...sliding.map((l) => l.window ?? Infinity)) : 0

  if (hybrid && swaTokens < window + pageSize) {
    throw new KvroomError(
      'SGLANG_SWA_POOL_TOO_SMALL',
      `The SWA pool (${swaTokens} tokens) cannot hold even one request (window ${window} + page size ${pageSize}). ` +
        'Give the KV cache more memory or raise --swa-full-tokens-ratio.',
      { swaTokens, window, pageSize },
    )
  }
  return { hybrid, fullBytesPerToken, swaBytesPerToken, fullTokens, swaTokens, window, pageSize }
}

/** Requests of `len` tokens both pools can hold at the same time. */
export function sglangMaxSeqs(plan: SglangPlan, len: number): number {
  const byFull = Math.floor(plan.fullTokens / len)
  if (!plan.hybrid) return byFull
  const bySwa = Math.floor(plan.swaTokens / Math.min(len, plan.window + plan.pageSize))
  return Math.min(byFull, bySwa)
}

/** Longest request length for which `seqs` requests fit in the pools (capped at `cap`). */
export function sglangMaxLen(plan: SglangPlan, seqs: number, cap: number): number {
  const bySeqs = Math.floor(plan.fullTokens / seqs)
  if (plan.hybrid && plan.swaTokens < seqs * (plan.window + plan.pageSize)) return 0
  return Math.min(bySeqs, cap)
}
