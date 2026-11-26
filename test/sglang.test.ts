import assert from 'node:assert/strict'
import { test } from 'node:test'
import { KvroomError } from '../src/core/errors.ts'
import { planSglang, sglangMaxLen, sglangMaxSeqs } from '../src/core/sglang.ts'
import type { SglangInput } from '../src/core/sglang.ts'
import { GiB, loadModel } from './helpers.ts'

function input(model: string, kvDtype: string, kvGiB: number, tp = 1): SglangInput {
  return { model: loadModel(model), kvDtype, tp, kvBytesPerRank: kvGiB * GiB, pageSize: 1, swaFullTokensRatio: 0.8 }
}

test('gpt-oss-20b fp8 splits into a full pool and an SWA pool', () => {
  const p = planSglang(input('gpt-oss-20b', 'fp8', 4.17))
  assert.equal(p.hybrid, true)
  assert.equal(p.fullTokens, 202433)
  assert.equal(p.swaTokens, 161946)
  assert.equal(p.window, 128)
})

test('dense and MLA cell sizes', () => {
  assert.equal(planSglang(input('llama-3.1-8b', 'auto', 10)).fullBytesPerToken, 131072)
  const ds = planSglang(input('deepseek-v3', 'auto', 10, 8))
  assert.equal(ds.fullBytesPerToken, 70272) // MLA latent is not split by TP
  assert.equal(ds.hybrid, false)
})

test('SWA pool smaller than window + page size fails like the engine', () => {
  // 1 MiB of KV for gpt-oss: far fewer than 128 + 1 SWA tokens fit.
  assert.throws(
    () => planSglang(input('gpt-oss-20b', 'fp8', 0.001)),
    (e: unknown) => e instanceof KvroomError && e.code === 'SGLANG_SWA_POOL_TOO_SMALL',
  )
})

test('page size rounds pool sizes down to a page multiple', () => {
  const p = planSglang({ ...input('llama-3.1-8b', 'auto', 1), pageSize: 64 })
  assert.equal(p.fullTokens % 64, 0)
})

test('max-total-tokens pins the pool and ignores the memory budget', () => {
  const p = planSglang({ ...input('gpt-oss-20b', 'fp8', 0.001), maxTotalTokens: 100000 })
  assert.equal(p.fullTokens, 100000)
  assert.equal(p.swaTokens, 80000)
})

test('capacity helpers agree with the pools', () => {
  const p = planSglang(input('llama-3.1-8b', 'auto', 10))
  assert.equal(sglangMaxSeqs(p, 8192), Math.floor(p.fullTokens / 8192))
  assert.equal(sglangMaxLen(p, 4, 1_000_000), Math.floor(p.fullTokens / 4))
})
