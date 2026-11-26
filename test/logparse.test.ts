import assert from 'node:assert/strict'
import { test } from 'node:test'
import { calibrateFromLog, parseVllmLog } from '../src/core/logparse.ts'
import { planVllm } from '../src/core/vllm.ts'
import type { VllmInput } from '../src/core/vllm.ts'
import { loadLog, loadModel } from './helpers.ts'

const llama: Omit<VllmInput, 'kvBytesPerRank'> = {
  model: loadModel('llama-3.1-8b'),
  kvDtype: 'auto',
  tp: 1,
  blockSize: 16,
  maxModelLen: 8192,
  maxNumBatchedTokens: 8192,
}
const gptOss: Omit<VllmInput, 'kvBytesPerRank'> = {
  model: loadModel('gpt-oss-20b'),
  kvDtype: 'fp8',
  tp: 1,
  blockSize: 16,
  maxModelLen: 128000,
  maxNumBatchedTokens: 8128,
}

test('receipt A log: facts are extracted and the budget reproduces the logged concurrency', () => {
  const facts = parseVllmLog(loadLog('receipt-a'))
  assert.equal(facts.availableKvGiB, 11.12)
  assert.equal(facts.gpuBlocks, 5691)
  assert.equal(facts.concurrency, 11.12)
  assert.equal(facts.maxModelLen, 8192)
  const b = calibrateFromLog(facts, llama)
  assert.ok(b)
  assert.equal(b.basis, 'blocks')
  const plan = planVllm({ ...llama, kvBytesPerRank: b.kvBytesPerRank, numBlocks: b.numBlocks })
  assert.ok(Math.abs(plan.maxConcurrency - 11.12) <= 0.01)
})

test('receipt B log: v0.12 token count converts back to 22792 blocks and 2.68x', () => {
  const facts = parseVllmLog(loadLog('receipt-b'))
  assert.equal(facts.availableKvGiB, 4.17)
  assert.equal(facts.kvTokens, 182336)
  assert.equal(facts.concurrency, 2.68)
  assert.equal(facts.maxModelLen, 128000)
  const b = calibrateFromLog(facts, gptOss)
  assert.ok(b)
  assert.equal(b.basis, 'tokens')
  assert.equal(b.numBlocks, 22792)
  const plan = planVllm({ ...gptOss, kvBytesPerRank: b.kvBytesPerRank, numBlocks: b.numBlocks })
  assert.ok(Math.abs(plan.maxConcurrency - 2.68) <= 0.01)
})

test('token definition can be forced to the current meaning (concurrency x length)', () => {
  const b = calibrateFromLog({ kvTokens: 342250, maxModelLen: 128000 }, gptOss, { tokenDefinition: 'current' })
  assert.ok(b)
  const plan = planVllm({ ...gptOss, kvBytesPerRank: b.kvBytesPerRank, numBlocks: b.numBlocks })
  assert.ok(Math.abs(plan.kvTokensCurrent - 342250) / 342250 < 0.001)
})

test('with only GiB or only concurrency the budget still comes out', () => {
  const g = calibrateFromLog({ availableKvGiB: 4.17 }, gptOss)
  assert.equal(g?.basis, 'gib')
  const c = calibrateFromLog({ concurrency: 2.68, maxModelLen: 128000 }, gptOss)
  assert.equal(c?.basis, 'concurrency')
})

test('lines in an unknown format give empty facts and no calibration', () => {
  const facts = parseVllmLog('INFO started\nsomething else: 42\n')
  assert.deepEqual(facts, {})
  assert.equal(calibrateFromLog(facts, gptOss), null)
  assert.deepEqual(parseVllmLog(''), {})
})
