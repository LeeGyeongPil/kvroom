import { KvroomError } from './errors.ts'

export type LayerKind = 'full' | 'sliding' | 'mla' | 'linear'

/** What one decoder layer stores per request. Dimensions are element counts, not bytes. */
export interface LayerSpec {
  kind: LayerKind
  /** Total KV heads in the model (1 for the MLA latent vector, 0 for linear layers). */
  kvHeads: number
  /** Elements per head for K (MLA: kv_lora_rank + qk_rope_head_dim). */
  kDim: number
  /** Elements per head for V (MLA and Gemma 4 `attention_k_eq_v`: 0). */
  vDim: number
  /** Sliding-window length in tokens (sliding layers only). */
  window?: number
  /** Fixed per-request state in bytes (linear attention layers only). */
  stateBytes?: number
  /** False when TP cannot split this layer's cache (MLA latent): every rank keeps all of it. */
  tpSplittable: boolean
}

/** A fact the parser noticed. `key` is an i18n key under `note.*`. */
export interface Note {
  key: string
  vars?: Record<string, string | number>
  level: 'info' | 'warn'
}

export type Confidence = 'high' | 'medium' | 'low'

export interface ParsedModel {
  layers: LayerSpec[]
  maxPosition: number
  numAttentionHeads: number
  /** Bytes per element of the model's own dtype (2 for bf16/fp16, 4 for fp32). */
  modelDtypeBytes: number
  /** `2 * layers * kvHeads * headDim` elements per token: the single-formula estimate many calculators use. */
  denseElemsPerToken: number
  notes: Note[]
  confidence: Confidence
}

type Json = Record<string, unknown>

function isObject(v: unknown): v is Json {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function num(c: Json, key: string): number | undefined {
  const v = c[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

function posInt(c: Json, key: string): number {
  const v = num(c, key)
  if (v === undefined || !Number.isInteger(v) || v <= 0) {
    throw new KvroomError(
      'CONFIG_INVALID',
      `config.json field "${key}" is missing or not a positive integer. Paste the model's full config.json.`,
      { field: key },
    )
  }
  return v
}

function dtypeBytesOf(root: Json, c: Json): number {
  const raw = c['torch_dtype'] ?? c['dtype'] ?? root['torch_dtype'] ?? root['dtype']
  return typeof raw === 'string' && /32/.test(raw) ? 4 : 2
}

/**
 * Turn a Hugging Face `config.json` into per-layer cache specs.
 * Handles nested `text_config`, missing `head_dim`, three ways of describing layer types,
 * MLA, Gemma 4 global layers and Gated DeltaNet linear layers.
 * @throws KvroomError `CONFIG_INVALID` when required fields are missing.
 */
export function parseConfig(raw: unknown): ParsedModel {
  if (!isObject(raw)) {
    throw new KvroomError('CONFIG_INVALID', 'config.json must be a JSON object.', { field: '(root)' })
  }
  const nested = raw['text_config']
  const c: Json = isObject(nested) ? nested : raw
  const notes: Note[] = []
  let confidence: Confidence = 'high'
  const lower = (to: Confidence) => {
    if (confidence === 'high' || to === 'low') confidence = to
  }

  const numLayers = posInt(c, 'num_hidden_layers')
  const heads = posInt(c, 'num_attention_heads')
  const kvHeads = num(c, 'num_key_value_heads') ?? heads
  const modelDtypeBytes = dtypeBytesOf(raw, c)

  let maxPosition = num(c, 'max_position_embeddings')
  if (maxPosition === undefined) {
    maxPosition = 4096
    notes.push({ key: 'note.maxPositionMissing', vars: { value: maxPosition }, level: 'warn' })
  }

  // MLA (DeepSeek V2/V3, Kimi K2 ...): one latent vector per token and layer, never split by TP.
  const kvLoraRank = num(c, 'kv_lora_rank')
  if (kvLoraRank) {
    const rope = num(c, 'qk_rope_head_dim') ?? 0
    const dim = kvLoraRank + rope
    if (c['head_dim'] == null) notes.push({ key: 'note.headDimMissing', level: 'info' })
    notes.push({ key: 'note.mla', vars: { dim, rank: kvLoraRank, rope }, level: 'info' })
    const hd = num(c, 'head_dim') ?? Math.floor(posInt(c, 'hidden_size') / heads)
    return {
      layers: Array.from({ length: numLayers }, () => ({
        kind: 'mla' as const,
        kvHeads: 1,
        kDim: dim,
        vDim: 0,
        tpSplittable: false,
      })),
      maxPosition,
      numAttentionHeads: heads,
      modelDtypeBytes,
      denseElemsPerToken: 2 * numLayers * kvHeads * hd,
      notes,
      confidence,
    }
  }

  let headDim = num(c, 'head_dim')
  if (headDim === undefined) {
    headDim = Math.floor(posInt(c, 'hidden_size') / heads)
    notes.push({ key: 'note.headDimMissing', level: 'info' })
  }

  const windowRaw = num(c, 'sliding_window')
  const slidingOff = c['use_sliding_window'] === false

  // Decide the type of every layer.
  let types: string[]
  const layerTypes = c['layer_types']
  const interval = num(c, 'full_attention_interval')
  const pattern = num(c, 'sliding_window_pattern')
  if (Array.isArray(layerTypes)) {
    types = layerTypes.map((t) => String(t))
    if (types.length !== numLayers) {
      throw new KvroomError(
        'CONFIG_INVALID',
        `layer_types has ${types.length} entries but num_hidden_layers is ${numLayers}.`,
        { field: 'layer_types' },
      )
    }
    notes.push({ key: 'note.layerTypes', level: 'info' })
  } else if (interval) {
    types = Array.from({ length: numLayers }, (_, i) => ((i + 1) % interval === 0 ? 'full_attention' : 'linear_attention'))
    notes.push({ key: 'note.fromInterval', vars: { n: interval }, level: 'info' })
  } else if (pattern && windowRaw !== undefined) {
    types = Array.from({ length: numLayers }, (_, i) => ((i + 1) % pattern === 0 ? 'full_attention' : 'sliding_attention'))
    notes.push({ key: 'note.fromPattern', vars: { n: pattern }, level: 'info' })
  } else if (windowRaw !== undefined && !slidingOff) {
    // Mistral style: only `sliding_window` is given and every layer uses it.
    types = Array.from({ length: numLayers }, () => 'sliding_attention')
    notes.push({ key: 'note.allSliding', vars: { window: windowRaw }, level: 'info' })
  } else {
    types = Array.from({ length: numLayers }, () => 'full_attention')
  }

  if (slidingOff && windowRaw !== undefined) {
    notes.push({ key: 'note.slidingDisabled', level: 'info' })
  }

  let sawGlobal = false
  const layers: LayerSpec[] = types.map((t): LayerSpec => {
    if (t === 'linear_attention') {
      // Gated DeltaNet (Qwen3-Next): conv state + recurrent state, fixed size per request.
      const kh = posInt(c, 'linear_num_key_heads')
      const vh = posInt(c, 'linear_num_value_heads')
      const kd = posInt(c, 'linear_key_head_dim')
      const vd = posInt(c, 'linear_value_head_dim')
      const conv = posInt(c, 'linear_conv_kernel_dim')
      const convElems = (conv - 1) * (2 * kh * kd + vh * vd)
      const ssmElems = vh * kd * vd
      return {
        kind: 'linear',
        kvHeads: 0,
        kDim: 0,
        vDim: 0,
        stateBytes: (convElems + ssmElems) * modelDtypeBytes,
        tpSplittable: true,
      }
    }
    if (t === 'sliding_attention' && !slidingOff) {
      if (windowRaw === undefined) {
        throw new KvroomError('CONFIG_INVALID', 'Layers are marked sliding_attention but "sliding_window" is missing.', {
          field: 'sliding_window',
        })
      }
      return { kind: 'sliding', kvHeads, kDim: headDim, vDim: headDim, window: windowRaw, tpSplittable: true }
    }
    const globalDim = num(c, 'global_head_dim')
    if (globalDim !== undefined) {
      // Gemma 4 style: global layers use another head count and head dim.
      sawGlobal = true
      const gh = num(c, 'num_global_key_value_heads') ?? kvHeads
      return {
        kind: 'full',
        kvHeads: gh,
        kDim: globalDim,
        vDim: c['attention_k_eq_v'] === true ? 0 : globalDim,
        tpSplittable: true,
      }
    }
    return { kind: 'full', kvHeads, kDim: headDim, vDim: headDim, tpSplittable: true }
  })

  if (sawGlobal) {
    notes.push({ key: 'note.gemma4Global', level: 'warn' })
    lower('medium')
  }
  if (layers.some((l) => l.kind === 'linear')) {
    notes.push({ key: 'note.linear', level: 'warn' })
    lower('medium')
  }

  return {
    layers,
    maxPosition,
    numAttentionHeads: heads,
    modelDtypeBytes,
    denseElemsPerToken: 2 * numLayers * kvHeads * headDim,
    notes,
    confidence,
  }
}

/** Count layers per kind, for display and tests. */
export function layerMix(model: ParsedModel): Partial<Record<LayerKind, number>> {
  const mix: Partial<Record<LayerKind, number>> = {}
  for (const l of model.layers) mix[l.kind] = (mix[l.kind] ?? 0) + 1
  return mix
}

/** Total per-request fixed state (linear layers) in bytes. */
export function fixedStateBytes(model: ParsedModel): number {
  return model.layers.reduce((s, l) => s + (l.stateBytes ?? 0), 0)
}
