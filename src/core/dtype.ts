import { KvroomError } from './errors.ts'

/** KV cache dtypes offered in the UI. `auto` means "same as the model dtype". */
export const KV_DTYPES = ['auto', 'fp8', 'fp8_e4m3', 'fp8_e5m2'] as const
export type KvDtype = (typeof KV_DTYPES)[number]

const BYTES: Record<string, number> = {
  fp8: 1,
  fp8_e4m3: 1,
  fp8_e5m2: 1,
  bf16: 2,
  bfloat16: 2,
  fp16: 2,
  float16: 2,
  fp32: 4,
  float32: 4,
}

/** Bytes per KV element. `auto` uses `modelDtypeBytes` (2 for bf16/fp16 models). */
export function kvDtypeBytes(dtype: string, modelDtypeBytes = 2): number {
  if (dtype === 'auto') return modelDtypeBytes
  const b = BYTES[dtype]
  if (b === undefined) {
    throw new KvroomError('UNKNOWN_DTYPE', `Unknown KV cache dtype "${dtype}". Use one of: ${KV_DTYPES.join(', ')}.`, {
      dtype,
    })
  }
  return b
}

/** Value for vLLM `--kv-cache-dtype`, or `undefined` when the flag should be left out. */
export function vllmDtypeName(dtype: string): string | undefined {
  kvDtypeBytes(dtype)
  if (dtype === 'auto' || dtype === 'bf16' || dtype === 'bfloat16' || dtype === 'fp16' || dtype === 'float16') {
    return undefined
  }
  return dtype
}

/** Value for SGLang `--kv-cache-dtype` (vLLM `fp8` is spelled `fp8_e4m3` there). */
export function sglangDtypeName(dtype: string): string | undefined {
  const v = vllmDtypeName(dtype)
  if (v === undefined) return undefined
  return v === 'fp8' ? 'fp8_e4m3' : v
}
