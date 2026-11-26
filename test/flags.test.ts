import assert from 'node:assert/strict'
import { test } from 'node:test'
import { KvroomError } from '../src/core/errors.ts'
import { sglangDtypeName } from '../src/core/dtype.ts'
import { sglangFlags, vllmFlags } from '../src/core/flags.ts'
import type { FlagOptions } from '../src/core/flags.ts'

const planB: FlagOptions = {
  model: 'openai/gpt-oss-20b',
  tp: 1,
  maxModelLen: 128000,
  maxSeqs: 2,
  kvDtype: 'fp8',
  maxNumBatchedTokens: 8128,
  swaFullTokensRatio: 0.8,
}

test('vLLM flags for the receipt B plan are exact', () => {
  assert.equal(
    vllmFlags(planB),
    'vllm serve openai/gpt-oss-20b --tensor-parallel-size 1 --max-model-len 128000 --max-num-seqs 2 --kv-cache-dtype fp8 --max-num-batched-tokens 8128',
  )
})

test('SGLang flags use fp8_e4m3 and the SWA ratio', () => {
  const s = sglangFlags(planB)
  assert.ok(s.includes('--tp-size 1 --context-length 128000 --max-running-requests 2 --kv-cache-dtype fp8_e4m3 --swa-full-tokens-ratio 0.8'), s)
  assert.ok(s.startsWith('python -m sglang.launch_server --model-path openai/gpt-oss-20b'))
})

test('dtype auto leaves --kv-cache-dtype out in both engines', () => {
  const o = { ...planB, kvDtype: 'auto' }
  assert.ok(!vllmFlags(o).includes('--kv-cache-dtype'))
  assert.ok(!sglangFlags(o).includes('--kv-cache-dtype'))
})

test('a fixed KV budget adds --kv-cache-memory-bytes as an integer', () => {
  const f = vllmFlags({ ...planB, kvCacheMemoryBytes: 4.17 * 1024 ** 3 })
  assert.match(f, /--kv-cache-memory-bytes \d+$/)
})

test('non-default block and page sizes are printed, defaults are not', () => {
  assert.ok(!vllmFlags({ ...planB, blockSize: 16 }).includes('--block-size'))
  assert.ok(vllmFlags({ ...planB, blockSize: 32 }).includes('--block-size 32'))
  assert.ok(sglangFlags({ ...planB, pageSize: 64 }).includes('--page-size 64'))
  assert.ok(sglangFlags({ ...planB, maxTotalTokens: 4096 }).includes('--max-total-tokens 4096'))
})

test('unknown dtype names are rejected with UNKNOWN_DTYPE', () => {
  assert.throws(() => sglangDtypeName('int4'), (e: unknown) => e instanceof KvroomError && e.code === 'UNKNOWN_DTYPE')
  assert.throws(() => vllmFlags({ ...planB, kvDtype: 'nope' }), (e: unknown) => e instanceof KvroomError)
})
