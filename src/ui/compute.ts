// Turns the form values into numbers by calling the core. No React here, so tests can run it in Node.
import { estimateBudget } from '../core/budget.ts'
import type { Budget } from '../core/budget.ts'
import { kvDtypeBytes } from '../core/dtype.ts'
import { KvroomError } from '../core/errors.ts'
import { sglangFlags, vllmFlags } from '../core/flags.ts'
import { calibrateFromLog } from '../core/logparse.ts'
import type { LogBudget, LogFacts, TokenDefinition } from '../core/logparse.ts'
import type { ParsedModel } from '../core/parse.ts'
import type { SglangPlan } from '../core/sglang.ts'
import { capacityTable, defaultLens, DEFAULT_SEQS, maxLenAt, maxSeqsAt, sglangPlanAt, vllmPlanAt } from '../core/table.ts'
import type { Cell, Engine, Settings } from '../core/table.ts'
import { placeOnTp } from '../core/tp.ts'
import type { TpPlacement } from '../core/tp.ts'
import { denseConcurrency, GIB } from '../core/vllm.ts'
import type { VllmPlan } from '../core/vllm.ts'
import { gbToGiB, GPUS } from '../data/gpus.ts'
import { REPO_RE } from '../hub.ts'
import { t } from '../i18n/index.ts'
import type { Lang, MessageKey } from '../i18n/index.ts'

export type BudgetMode = 'estimate' | 'fixed' | 'log'

/** Values shared by every compared setup. */
export interface Common {
  engine: Engine
  maxNumBatchedTokens: number
  blockSize: number
  pageSize: number
  swaRatio: number
  utilization: number
  reserveGiB: number
  budgetMode: BudgetMode
  fixedGiB: number
  /** Length the summary, flags and "max seqs" use. */
  len: number
  /** Concurrency the flags and "max length" use. */
  seqs: number
  /** Total model weight bytes. */
  weightBytes: number
  logFacts: LogFacts | null
  tokenDefinition: TokenDefinition
}

/** What differs between compared columns. */
export interface Setup {
  gpu: string
  customGB: number
  tp: number
  kvDtype: string
}

export interface Computed {
  gpuGiB: number
  placement: TpPlacement | null
  budget: Budget | null
  settings: Settings | null
  vllm: VllmPlan | null
  sglang: SglangPlan | null
  /** Engine-specific problem, shown translated. The rest of the result is still filled when possible. */
  error: KvroomError | null
  /** vLLM: one request of `len` does not fit; longest length that does. */
  tooSmallMaxLen: number | null
  table: Cell[][]
  lens: number[]
  seqsList: readonly number[]
  maxSeqs: number
  maxLenForSeqs: number
  denseConcurrency: number | null
  logBudget: LogBudget | null
}

export interface Commands {
  vllm: string
  sglang: string
  /** True when the KV size is pinned in the flags (the plan is then deterministic). */
  pinned: boolean
}

/** Every number the form validates. */
export type InputField =
  | 'blockSize'
  | 'pageSize'
  | 'maxNumBatchedTokens'
  | 'len'
  | 'seqs'
  | 'tp'
  | 'utilization'
  | 'swaRatio'
  | 'reserveGiB'
  | 'weightBytes'
  | 'customGB'
  | 'fixedGiB'

/**
 * The on-screen label (i18n key) of each validated field. The error message shows this label,
 * never the internal field name, so the user can find the box on the form.
 */
export const INPUT_LABELS: Record<InputField, MessageKey> = {
  blockSize: 'serving.blockSize',
  pageSize: 'serving.pageSize',
  maxNumBatchedTokens: 'serving.batched',
  len: 'summary.len',
  seqs: 'summary.seqs',
  tp: 'serving.count',
  utilization: 'serving.util',
  swaRatio: 'serving.swaRatio',
  reserveGiB: 'serving.reserve',
  weightBytes: 'model.weights',
  customGB: 'serving.gpuCustom',
  fixedGiB: 'serving.fixedGiB',
}

interface Rule {
  field: InputField
  value: number
  min: number
  max: number
  /** Smallest allowed value is excluded (value must be > min). */
  exclusiveMin?: boolean
  integer?: boolean
}

/**
 * Reject values that would turn the output into NaN or into flags the engines refuse.
 * The error carries structured details (label key, min, max, flags); the translated text is
 * built in `errorText` so no English range text leaks into the Korean screen.
 * @throws KvroomError `INPUT_OUT_OF_RANGE` naming the field and the allowed range.
 */
export function validateInputs(c: Common, s: Setup): void {
  const rules: Rule[] = [
    { field: 'blockSize', value: c.blockSize, min: 1, max: Infinity, integer: true },
    { field: 'pageSize', value: c.pageSize, min: 1, max: Infinity, integer: true },
    { field: 'maxNumBatchedTokens', value: c.maxNumBatchedTokens, min: 1, max: Infinity, integer: true },
    { field: 'len', value: c.len, min: 1, max: Infinity, integer: true },
    { field: 'seqs', value: c.seqs, min: 1, max: Infinity, integer: true },
    { field: 'tp', value: s.tp, min: 1, max: Infinity, integer: true },
    { field: 'utilization', value: c.utilization, min: 0, max: 1, exclusiveMin: true },
    { field: 'swaRatio', value: c.swaRatio, min: 0, max: 1, exclusiveMin: true },
    { field: 'reserveGiB', value: c.reserveGiB, min: 0, max: Infinity },
    { field: 'weightBytes', value: c.weightBytes, min: 0, max: Infinity },
    { field: 'customGB', value: s.customGB, min: 0, max: Infinity, exclusiveMin: true },
  ]
  if (c.budgetMode === 'fixed') rules.push({ field: 'fixedGiB', value: c.fixedGiB, min: 0, max: Infinity, exclusiveMin: true })
  for (const r of rules) {
    const bad =
      !Number.isFinite(r.value) ||
      (r.exclusiveMin ? r.value <= r.min : r.value < r.min) ||
      r.value > r.max ||
      (r.integer === true && !Number.isInteger(r.value))
    if (bad) {
      // 화면 문구는 i18n 이 만든다: 여기서는 숫자와 표시용 라벨 키만 넘긴다.
      const details: Record<string, string | number> = {
        field: r.field,
        labelKey: INPUT_LABELS[r.field],
        min: r.min,
        exclusiveMin: r.exclusiveMin ? 1 : 0,
        integer: r.integer ? 1 : 0,
        value: r.value,
      }
      if (Number.isFinite(r.max)) details.max = r.max
      throw new KvroomError('INPUT_OUT_OF_RANGE', `${r.field} = ${r.value} is out of range.`, details)
    }
  }
}

/**
 * Both engines' launch commands for the chosen length and concurrency.
 * @throws KvroomError `INPUT_OUT_OF_RANGE` when a form value is invalid (no flags are produced then).
 */
export function buildCommands(repo: string, model: ParsedModel, c: Common, s: Setup, comp: Computed): Commands {
  validateInputs(c, s)
  const fixed = c.budgetMode === 'fixed'
  const hybrid = model.layers.some((l) => l.kind === 'sliding') && model.layers.some((l) => l.kind === 'full' || l.kind === 'mla')
  let maxTotalTokens: number | undefined
  if (fixed && comp.settings) {
    try {
      maxTotalTokens = sglangPlanAt(comp.settings, c.seqs).fullTokens
    } catch {
      maxTotalTokens = undefined
    }
  }
  const base = { model: repo, tp: s.tp, maxModelLen: c.len, maxSeqs: c.seqs, kvDtype: s.kvDtype }
  return {
    vllm: vllmFlags({
      ...base,
      maxNumBatchedTokens: c.maxNumBatchedTokens,
      ...(fixed ? { kvCacheMemoryBytes: c.fixedGiB * GIB } : {}),
      blockSize: c.blockSize,
    }),
    sglang: sglangFlags({
      ...base,
      ...(hybrid ? { swaFullTokensRatio: c.swaRatio } : {}),
      pageSize: c.pageSize,
      ...(maxTotalTokens !== undefined ? { maxTotalTokens } : {}),
    }),
    pinned: fixed,
  }
}

export function gpuGiBOf(s: Setup): number {
  if (s.gpu === 'custom') return gbToGiB(s.customGB)
  const g = GPUS.find((x) => x.name === s.gpu)
  return gbToGiB(g ? g.memoryGB : s.customGB)
}

export function compute(model: ParsedModel, c: Common, s: Setup): Computed {
  const lens = defaultLens(model.maxPosition)
  const empty: Computed = {
    gpuGiB: gpuGiBOf(s),
    placement: null,
    budget: null,
    settings: null,
    vllm: null,
    sglang: null,
    error: null,
    tooSmallMaxLen: null,
    table: [],
    lens,
    seqsList: DEFAULT_SEQS,
    maxSeqs: 0,
    maxLenForSeqs: 0,
    denseConcurrency: null,
    logBudget: null,
  }
  try {
    validateInputs(c, s)
    kvDtypeBytes(s.kvDtype, model.modelDtypeBytes)
    const placement = placeOnTp(model, s.tp)
    const out: Computed = { ...empty, placement }

    // 1. KV budget per GPU.
    let budget: Budget
    let logBudget: LogBudget | null = null
    if (c.budgetMode === 'fixed') {
      budget = estimateBudget({
        gpuGiB: out.gpuGiB,
        tp: s.tp,
        utilization: c.utilization,
        weightBytes: c.weightBytes,
        reserveGiB: c.reserveGiB,
        fixedKvBytes: c.fixedGiB * GIB,
      })
    } else if (c.budgetMode === 'log' && c.engine === 'vllm' && c.logFacts) {
      logBudget = calibrateFromLog(
        c.logFacts,
        {
          model,
          kvDtype: s.kvDtype,
          tp: s.tp,
          blockSize: c.blockSize,
          maxModelLen: c.len,
          maxNumBatchedTokens: c.maxNumBatchedTokens,
        },
        { tokenDefinition: c.tokenDefinition },
      )
      budget = logBudget ?? estimateBudget({ gpuGiB: out.gpuGiB, tp: s.tp, utilization: c.utilization, weightBytes: c.weightBytes, reserveGiB: c.reserveGiB })
    } else {
      budget = estimateBudget({
        gpuGiB: out.gpuGiB,
        tp: s.tp,
        utilization: c.utilization,
        weightBytes: c.weightBytes,
        reserveGiB: c.reserveGiB,
      })
    }
    out.budget = budget
    out.logBudget = logBudget

    const settings: Settings = {
      model,
      kvDtype: s.kvDtype,
      tp: s.tp,
      kvBytesPerRank: budget.kvBytesPerRank,
      ...(budget.numBlocks !== undefined && c.engine === 'vllm' ? { numBlocks: budget.numBlocks } : {}),
      blockSize: c.blockSize,
      maxNumBatchedTokens: c.maxNumBatchedTokens,
      pageSize: c.pageSize,
      swaFullTokensRatio: c.swaRatio,
    }
    out.settings = settings

    // 2. Engine plan at the chosen length, then the table.
    if (c.engine === 'vllm') {
      const plan = vllmPlanAt(settings, c.len)
      out.vllm = plan
      if (plan.numBlocks < plan.blocksPerRequest) out.tooSmallMaxLen = maxLenAt('vllm', settings, 1, c.len)
      out.denseConcurrency = denseConcurrency({ ...settings, maxModelLen: c.len })
    } else {
      out.sglang = sglangPlanAt(settings, c.seqs)
    }
    out.table = capacityTable(c.engine, settings, lens, DEFAULT_SEQS)
    out.maxSeqs = maxSeqsAt(c.engine, settings, c.len)
    out.maxLenForSeqs = maxLenAt(c.engine, settings, c.seqs, model.maxPosition)
    return out
  } catch (e) {
    if (e instanceof KvroomError) return { ...empty, error: e }
    throw e
  }
}

/** What was read from a vLLM log, in the screen language (units are translated, numbers are not). */
export function describeFacts(lang: Lang, f: LogFacts): string {
  const n = (v: number): string => v.toLocaleString('en-US')
  const parts: string[] = []
  if (f.availableKvGiB !== undefined) parts.push(`${f.availableKvGiB} GiB`)
  if (f.gpuBlocks !== undefined) parts.push(t(lang, 'log.fact.blocks', { n: n(f.gpuBlocks) }))
  if (f.kvTokens !== undefined) parts.push(t(lang, 'log.fact.tokens', { n: n(f.kvTokens) }))
  if (f.concurrency !== undefined) {
    parts.push(f.maxModelLen !== undefined ? t(lang, 'log.fact.concurrency', { c: f.concurrency, len: n(f.maxModelLen) }) : `${f.concurrency}x`)
  }
  return parts.join(', ')
}

/** Placeholder shown in the flags when a pasted config has no usable Hub id. */
export const PLACEHOLDER_MODEL = '<your-model>'

/**
 * Structure key of a parsed model: the numbers every KV calculation reads (layer list, max length,
 * heads, dtype). Two configs with the same key need the same KV cache, so they count as one model.
 */
export function modelFingerprint(m: ParsedModel): string {
  return JSON.stringify({ layers: m.layers, maxPosition: m.maxPosition, heads: m.numAttentionHeads, dtype: m.modelDtypeBytes })
}

/**
 * Form changes when another model is put on screen. The length is capped at the new model's maximum,
 * and a pasted vLLM log is dropped when the structure changed: its block and token counts belong
 * to the previous model, so calibrating the new one with them would give a wrong budget.
 */
export function commonForModel(c: Common, previous: ParsedModel | null, next: ParsedModel): { patch: Partial<Common>; logCleared: boolean } {
  const patch: Partial<Common> = { len: Math.min(next.maxPosition, c.len) }
  const changed = previous === null || modelFingerprint(previous) !== modelFingerprint(next)
  const logCleared = changed && c.logFacts !== null
  if (logCleared) {
    patch.logFacts = null
    if (c.budgetMode === 'log') patch.budgetMode = 'estimate'
  }
  return { patch, logCleared }
}

export interface PasteInput {
  /** Text in the Hugging Face id box at the moment of pasting. */
  repoText: string
  /** True only when the user typed in the id box after the last model was loaded (the box starts prefilled). */
  repoEdited: boolean
  /** The pasted config, already parsed. */
  pasted: ParsedModel
  /** The model on screen before the paste. */
  previous: { repo: string; weightBytes: number; model: ParsedModel } | null
}

export interface PasteTarget {
  repo: string
  weightBytes: number
  /** The pasted config has the same structure as the model already loaded. */
  sameModel: boolean
}

/**
 * Model id and weight size for a pasted config.json.
 *
 * Whether it is "the same model" is decided from the config structure, never from the id box:
 * the box is prefilled with the loaded model's id, so its text alone says nothing about the paste.
 * - Same structure: keep the previous weight size (and id, unless the user typed a new one).
 * - Different structure: weight size goes back to 0 (the screen asks for it), and the id is the one
 *   the user typed after the last load if it looks like owner/name, else `<your-model>`.
 */
export function pasteTarget(input: PasteInput): PasteTarget {
  const { previous, pasted } = input
  const typed = input.repoText.trim()
  // 상자에 원래 있던(= 이전 모델) ID 는 사용자가 쓴 값으로 치지 않는다.
  const typedOk =
    input.repoEdited && REPO_RE.test(typed) && (previous === null || typed.toLowerCase() !== previous.repo.toLowerCase())
  const sameModel = previous !== null && modelFingerprint(previous.model) === modelFingerprint(pasted)
  if (sameModel) return { repo: typedOk ? typed : previous.repo, weightBytes: previous.weightBytes, sameModel }
  return { repo: typedOk ? typed : PLACEHOLDER_MODEL, weightBytes: 0, sameModel }
}
