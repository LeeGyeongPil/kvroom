// Read the lines vLLM prints at start-up and turn them back into a KV budget.
// Log formats seen in public logs: vLLM issue #12330 (early v1-era text) and a v0.12.0 server log.
import type { Budget } from './budget.ts'
import { planVllm } from './vllm.ts'
import type { VllmInput } from './vllm.ts'

export interface LogFacts {
  /** "Available KV cache memory: X GiB" (or the older "the rest of the memory reserved for KV Cache is X GiB"). */
  availableKvGiB?: number
  /** "# GPU blocks: N". */
  gpuBlocks?: number
  /** "GPU KV cache size: N tokens". */
  kvTokens?: number
  /** "Maximum concurrency for L tokens per request: Cx" -> C. */
  concurrency?: number
  /** ... and L. */
  maxModelLen?: number
  /** "vLLM version X.Y.Z" when present. */
  vllmVersion?: string
}

const toNumber = (s: string): number => Number(s.replace(/,/g, ''))

/** Extract facts from pasted log text. Unknown formats give an empty object, never an exception. */
export function parseVllmLog(text: string): LogFacts {
  const facts: LogFacts = {}
  const gib = /(?:Available KV cache memory:|the rest of the memory reserved for KV Cache is)\s*([\d.]+)\s*GiB/i.exec(text)
  if (gib?.[1]) facts.availableKvGiB = Number(gib[1])
  const blocks = /#\s*GPU blocks:\s*(\d[\d,]*)/i.exec(text)
  if (blocks?.[1]) facts.gpuBlocks = toNumber(blocks[1])
  const tokens = /GPU KV cache size:\s*(\d[\d,]*)\s*tokens/i.exec(text)
  if (tokens?.[1]) facts.kvTokens = toNumber(tokens[1])
  const conc = /Maximum concurrency for\s*(\d[\d,]*)\s*tokens per request:\s*([\d.]+)x/i.exec(text)
  if (conc?.[1] && conc[2]) {
    facts.maxModelLen = toNumber(conc[1])
    facts.concurrency = Number(conc[2])
  }
  const ver = /vLLM(?: API server)? version\s*v?(\d+\.\d+(?:\.\d+)?)/i.exec(text)
  if (ver?.[1]) facts.vllmVersion = ver[1]
  return facts
}

export type TokenDefinition = 'v012' | 'current' | 'auto'

export interface CalibrateOptions {
  /**
   * What "GPU KV cache size" meant in the engine that printed the log:
   * `v012` = `blocks // groups * block`, `current` = `concurrency * max_model_len`.
   * `auto` picks the one that reproduces the logged concurrency, else decides by version.
   */
  tokenDefinition?: TokenDefinition
}

export interface LogBudget extends Budget {
  numBlocks: number
  /** Which log fact the budget came from. */
  basis: 'blocks' | 'tokens' | 'gib' | 'concurrency'
  /** Max model length the numbers belong to (from the concurrency line when present). */
  maxModelLen: number
}

/**
 * Work out the real KV budget from log facts. Priority: block count > token count > GiB > concurrency
 * (exact counts beat rounded GiB). Returns `null` when the log held nothing usable.
 */
export function calibrateFromLog(
  facts: LogFacts,
  input: Omit<VllmInput, 'kvBytesPerRank' | 'numBlocks'>,
  options: CalibrateOptions = {},
): LogBudget | null {
  const len = facts.maxModelLen ?? input.maxModelLen
  const base = planVllm({ ...input, maxModelLen: len, kvBytesPerRank: 0 })
  const make = (blocks: number, basis: LogBudget['basis']): LogBudget => ({
    kvBytesPerRank: Math.round(blocks) * base.bytesPerBlock,
    source: 'log',
    weightBytesPerRank: 0,
    numBlocks: Math.round(blocks),
    basis,
    maxModelLen: len,
  })

  if (facts.gpuBlocks !== undefined) return make(facts.gpuBlocks, 'blocks')

  if (facts.kvTokens !== undefined) {
    const fromV012 = (facts.kvTokens / input.blockSize) * base.numGroups
    const fromCurrent = (facts.kvTokens / len) * base.blocksPerRequest
    let def = options.tokenDefinition ?? 'auto'
    if (def === 'auto') {
      if (facts.concurrency !== undefined) {
        const errOld = Math.abs(fromV012 / base.blocksPerRequest - facts.concurrency)
        const errNew = Math.abs(fromCurrent / base.blocksPerRequest - facts.concurrency)
        def = errOld < errNew ? 'v012' : 'current'
      } else if (facts.vllmVersion) {
        const minor = Number(facts.vllmVersion.split('.')[1] ?? '99')
        def = minor <= 12 ? 'v012' : 'current'
      } else {
        def = 'current'
      }
    }
    return make(def === 'v012' ? fromV012 : fromCurrent, 'tokens')
  }

  if (facts.availableKvGiB !== undefined) {
    return make(Math.floor((facts.availableKvGiB * 1024 ** 3) / base.bytesPerBlock), 'gib')
  }
  if (facts.concurrency !== undefined) return make(facts.concurrency * base.blocksPerRequest, 'concurrency')
  return null
}
