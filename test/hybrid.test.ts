import assert from 'node:assert/strict'
import { test } from 'node:test'
import { KvroomError } from '../src/core/errors.ts'
import { assertFitsOne, maxLenForSeqs, maxSeqsForLen, planVllm } from '../src/core/vllm.ts'
import type { VllmInput } from '../src/core/vllm.ts'
import { GiB, loadModel } from './helpers.ts'

function gemma(maxNumBatchedTokens: number, kvGiB = 20, maxModelLen = 131072): VllmInput {
  return {
    model: loadModel('gemma-3-27b'),
    kvDtype: 'auto',
    tp: 1,
    blockSize: 16,
    maxModelLen,
    maxNumBatchedTokens,
    kvBytesPerRank: kvGiB * GiB,
  }
}

test('gemma-3-27b: 7 groups of 10 layers, 8 padding slots, 11654 blocks per request', () => {
  const p = planVllm(gemma(8192))
  assert.equal(p.numGroups, 7)
  assert.equal(p.groupSize, 10)
  assert.equal(p.paddingSlots, 8)
  // 1 full group: 131072/16 = 8192 blocks. 6 sliding groups: cdiv(1023 + 8192, 16) + 1 = 577 each.
  assert.equal(p.blocksPerRequest, 8192 + 6 * 577)
  assert.equal(p.blocksPerRequest, 11654)
  assert.equal(p.maxConcurrency.toFixed(2), '1.41')
})

test('gemma-3-27b: trade-off between concurrency and length, and max_num_batched_tokens', () => {
  const at8192 = maxLenForSeqs(gemma(8192), 8, 131072)
  assert.equal(at8192, 4656)
  const at2048 = maxLenForSeqs(gemma(2048), 8, 131072)
  assert.ok(at2048 > at8192, `${at2048} should exceed ${at8192}`)
  assert.equal(maxSeqsForLen(gemma(8192), 4656), 8)
})

test('a KV cache that cannot hold one request throws KV_TOO_SMALL with an estimated max length', () => {
  const input = gemma(8192, 0.5)
  assert.throws(
    () => assertFitsOne(input),
    (e: unknown) =>
      e instanceof KvroomError &&
      e.code === 'KV_TOO_SMALL' &&
      typeof e.details['estimatedMaxLen'] === 'number' &&
      e.details['estimatedMaxLen'] > 0 &&
      e.details['estimatedMaxLen'] < 131072,
  )
  // The estimate it reports really fits.
  const est = maxLenForSeqs(input, 1, 131072)
  assert.doesNotThrow(() => assertFitsOne({ ...input, maxModelLen: est }))
})

test('maxConcurrentBatches multiplies the in-flight tokens sliding layers hold', () => {
  const one = planVllm(gemma(2048))
  const two = planVllm({ ...gemma(2048), maxConcurrentBatches: 2 })
  assert.ok(two.blocksPerRequest > one.blocksPerRequest)
})

test('dense GQA model is a single group with no padding', () => {
  const p = planVllm({ ...gemma(8192), model: loadModel('llama-3.3-70b'), tp: 8 })
  assert.equal(p.numGroups, 1)
  assert.equal(p.paddingSlots, 0)
  assert.equal(p.groupSize, 80)
})

test('mixed page sizes (gemma-4) unify by enlarging the smaller page block size', () => {
  const p = planVllm({ ...gemma(8192), model: loadModel('gemma-4-31b') })
  assert.equal(p.groupSize, 10)
  assert.equal(p.pageMismatch, false)
  // sliding page (16 heads x 512) is 4x the full page (4 heads x 512): full layers use 4x larger blocks.
  const full = p.groups.find((g) => g.kind === 'full')
  assert.equal(full?.blocksPerRequest, Math.ceil(131072 / 64))
})

test('qwen3-next linear state adds blocks per request (estimate)', () => {
  const p = planVllm({ ...gemma(8192), model: loadModel('qwen3-next-80b'), maxModelLen: 8192 })
  assert.ok(p.stateBlocksPerRequest > 0)
  assert.equal(p.blocksPerRequest, p.groups.reduce((s, g) => s + g.blocksPerRequest, 0) + p.stateBlocksPerRequest)
})
