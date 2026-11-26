// Public API of the calculation core. No DOM, no network: safe to run in Node or the browser.
export { KvroomError, ERROR_CODES } from './errors.ts'
export type { KvroomErrorCode } from './errors.ts'
export { KV_DTYPES, kvDtypeBytes, vllmDtypeName, sglangDtypeName } from './dtype.ts'
export type { KvDtype } from './dtype.ts'
export { parseConfig, layerMix, fixedStateBytes } from './parse.ts'
export type { LayerKind, LayerSpec, Note, Confidence, ParsedModel } from './parse.ts'
export { placeOnTp, headsPerRank, bytesPerTokenPerRank } from './tp.ts'
export type { TpPlacement } from './tp.ts'
export {
  planVllm,
  maxLenForSeqs,
  maxSeqsForLen,
  assertFitsOne,
  structuredBytesPerToken,
  denseBytesPerToken,
  denseConcurrency,
  VLLM_RULES_VERSION,
  GIB,
} from './vllm.ts'
export type { VllmInput, VllmPlan, VllmGroup } from './vllm.ts'
export { planSglang, sglangMaxSeqs, sglangMaxLen, SGLANG_RULES_VERSION } from './sglang.ts'
export type { SglangInput, SglangPlan } from './sglang.ts'
export { estimateBudget } from './budget.ts'
export type { Budget, BudgetInput, BudgetSource } from './budget.ts'
export { parseVllmLog, calibrateFromLog } from './logparse.ts'
export type { LogFacts, LogBudget, CalibrateOptions, TokenDefinition } from './logparse.ts'
export { vllmFlags, sglangFlags } from './flags.ts'
export type { FlagOptions } from './flags.ts'
export { capacityTable, defaultLens, maxSeqsAt, maxLenAt, vllmPlanAt, sglangPlanAt, DEFAULT_SEQS } from './table.ts'
export type { Cell, Engine, Settings } from './table.ts'
