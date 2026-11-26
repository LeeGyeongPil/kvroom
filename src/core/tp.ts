import { KvroomError } from './errors.ts'
import type { LayerSpec, ParsedModel } from './parse.ts'

/** KV heads one TP rank keeps for a layer: split when possible, replicated when TP > KV heads. */
export function headsPerRank(layer: LayerSpec, tp: number): number {
  if (!layer.tpSplittable) return layer.kvHeads
  if (layer.kvHeads >= tp) return layer.kvHeads / tp
  return 1
}

export interface TpPlacement {
  tp: number
  /** KV heads held by one rank (largest over the attention layers). */
  headsPerRank: number
  /** Total KV bytes across all ranks divided by the unsharded size (1 = no duplication). */
  replication: number
}

/**
 * Check that the engine accepts this TP size and report how much KV cache gets duplicated.
 * @throws KvroomError `TP_NOT_DIVISIBLE` when heads cannot be split evenly (the engine rejects it too).
 */
export function placeOnTp(model: ParsedModel, tp: number): TpPlacement {
  if (!Number.isInteger(tp) || tp < 1) {
    throw new KvroomError('TP_NOT_DIVISIBLE', `Tensor parallel size must be a positive integer, got ${tp}.`, { tp })
  }
  if (model.numAttentionHeads % tp !== 0) {
    throw new KvroomError(
      'TP_NOT_DIVISIBLE',
      `${model.numAttentionHeads} attention heads cannot be split across ${tp} GPUs. Pick a tensor-parallel size that divides ${model.numAttentionHeads}.`,
      { heads: model.numAttentionHeads, tp },
    )
  }
  let maxHeads = 0
  let sharded = 0
  let whole = 0
  for (const l of model.layers) {
    if (l.kind === 'linear') continue
    if (l.tpSplittable) {
      const ok = l.kvHeads >= tp ? l.kvHeads % tp === 0 : tp % l.kvHeads === 0
      if (!ok) {
        throw new KvroomError(
          'TP_NOT_DIVISIBLE',
          `${l.kvHeads} KV heads do not fit tensor-parallel size ${tp}. Pick a size that divides ${l.kvHeads} or is a multiple of it.`,
          { heads: l.kvHeads, tp },
        )
      }
    }
    const h = headsPerRank(l, tp)
    maxHeads = Math.max(maxHeads, h)
    const unit = l.kDim + l.vDim
    sharded += h * unit * tp
    whole += l.kvHeads * unit
  }
  const replication = whole === 0 ? 1 : Math.round((sharded / whole) * 100) / 100
  return { tp, headsPerRank: maxHeads, replication }
}

/** KV bytes per token held by one rank, summing every attention layer (nothing is windowed out). */
export function bytesPerTokenPerRank(model: ParsedModel, elemBytes: number, tp: number): number {
  let total = 0
  for (const l of model.layers) {
    if (l.kind === 'linear') continue
    total += headsPerRank(l, tp) * (l.kDim + l.vDim) * elemBytes
  }
  return total
}
