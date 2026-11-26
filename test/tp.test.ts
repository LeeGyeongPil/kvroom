import assert from 'node:assert/strict'
import { test } from 'node:test'
import { KvroomError } from '../src/core/errors.ts'
import { parseConfig } from '../src/core/parse.ts'
import { placeOnTp } from '../src/core/tp.ts'
import { loadModel } from './helpers.ts'

test('llama-3.3-70b: KV heads 8 split up to TP8, duplicated x2 at TP16', () => {
  const m = loadModel('llama-3.3-70b')
  assert.equal(placeOnTp(m, 1).replication, 1)
  assert.equal(placeOnTp(m, 8).replication, 1)
  assert.equal(placeOnTp(m, 8).headsPerRank, 1)
  assert.equal(placeOnTp(m, 16).replication, 2)
})

test('qwen3-next: 2 KV heads, TP4 duplicates x2', () => {
  assert.equal(placeOnTp(loadModel('qwen3-next-80b'), 4).replication, 2)
})

test('deepseek-v3: MLA latent is kept on every rank (x8 at TP8)', () => {
  const p = placeOnTp(loadModel('deepseek-v3'), 8)
  assert.equal(p.replication, 8)
  assert.equal(p.headsPerRank, 1)
})

test('heads that TP cannot divide throw TP_NOT_DIVISIBLE', () => {
  const m = loadModel('gpt-oss-20b') // 64 attention heads
  assert.throws(
    () => placeOnTp(m, 3),
    (e: unknown) => e instanceof KvroomError && e.code === 'TP_NOT_DIVISIBLE' && e.details['tp'] === 3,
  )
  assert.throws(() => placeOnTp(m, 0), (e: unknown) => e instanceof KvroomError && e.code === 'TP_NOT_DIVISIBLE')
})

test('KV heads that do not divide TP are rejected', () => {
  const m = parseConfig({
    num_hidden_layers: 2,
    num_attention_heads: 24,
    num_key_value_heads: 6,
    head_dim: 64,
    max_position_embeddings: 4096,
  })
  assert.equal(placeOnTp(m, 6).replication, 1)
  assert.throws(() => placeOnTp(m, 4), (e: unknown) => e instanceof KvroomError && e.code === 'TP_NOT_DIVISIBLE')
})
