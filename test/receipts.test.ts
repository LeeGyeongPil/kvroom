// Accuracy receipts: the calculation must reproduce numbers printed by real vLLM servers.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { denseConcurrency, planVllm } from '../src/core/vllm.ts'
import { GiB, loadModel } from './helpers.ts'

// Receipt A: https://github.com/vllm-project/vllm/issues/12330 (Meta-Llama-3-8B, A100 40GB)
//   "the rest of the memory reserved for KV Cache is 11.12GiB", "# GPU blocks: 5691",
//   "Maximum concurrency for 8192 tokens per request: 11.12x"
test('receipt A: llama-3-8b A100 matches vLLM log', () => {
  const plan = planVllm({
    model: loadModel('llama-3.1-8b'), // same KV shape as Llama-3-8B: 32 layers, 8 KV heads, head_dim 128
    kvDtype: 'auto',
    tp: 1,
    blockSize: 16,
    maxModelLen: 8192,
    maxNumBatchedTokens: 8192,
    kvBytesPerRank: 11.12 * GiB,
  })
  assert.ok(Math.abs(plan.numBlocks - 5691) / 5691 <= 0.001, `numBlocks ${plan.numBlocks}`)
  assert.equal(plan.maxConcurrency.toFixed(2), '11.12')
  assert.equal(plan.numGroups, 1)
})

// Receipt B: https://dev.to/ratibor78/running-openais-gpt-oss-20b-with-128k-context-on-a-single-l4-gpu-27ac
//   vLLM v0.12.0: "Available KV cache memory: 4.17 GiB", "GPU KV cache size: 182,336 tokens",
//   "Maximum concurrency for 128,000 tokens per request: 2.68x", --max-num-batched-tokens 8128
test('receipt B: gpt-oss-20b L4 hybrid SWA matches vLLM log', () => {
  const input = {
    model: loadModel('gpt-oss-20b'),
    kvDtype: 'fp8',
    tp: 1,
    blockSize: 16,
    maxModelLen: 128000,
    maxNumBatchedTokens: 8128,
    kvBytesPerRank: 4.17 * GiB,
  }
  const plan = planVllm(input)
  assert.equal(plan.numGroups, 2)
  assert.equal(plan.groupSize, 12)
  assert.equal(plan.paddingSlots, 0)
  assert.equal(plan.blocksPerRequest, 8517) // full 8000 + sliding cdiv(127 + 8128, 16) + 1 = 517
  assert.ok(plan.maxConcurrency >= 2.67 && plan.maxConcurrency <= 2.68, `concurrency ${plan.maxConcurrency}`)
  assert.ok(Math.abs(plan.kvTokensV012 - 182336) / 182336 <= 0.001, `tokens ${plan.kvTokensV012}`)

  // The log's token count converts back to the exact block count: 22792.
  const exact = planVllm({ ...input, numBlocks: 22792 })
  assert.equal(exact.maxConcurrency.toFixed(2), '2.68')
  assert.equal(exact.kvTokensV012, 182336)

  // One dense formula (24576 B/token) is far off: 1.42x instead of 2.68x.
  const dense = denseConcurrency(input)
  assert.equal(dense.toFixed(2), '1.42')
  assert.ok((dense - 2.68) / 2.68 < -0.4)
})
