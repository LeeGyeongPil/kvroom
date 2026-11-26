import assert from 'node:assert/strict'
import { test } from 'node:test'
import { KvroomError } from '../src/core/errors.ts'
import { fixedStateBytes, layerMix, parseConfig } from '../src/core/parse.ts'
import { denseBytesPerToken, structuredBytesPerToken } from '../src/core/vllm.ts'
import { loadConfig, loadModel } from './helpers.ts'

// Expected values come from the research probe (work/<id>/probe/probe-output.md), computed from the real configs.
const CASES: [string, Record<string, number>, number][] = [
  ['llama-3.1-8b', { full: 32 }, 131072],
  ['qwen3-8b', { full: 36 }, 147456],
  ['llama-3.3-70b', { full: 80 }, 327680],
  ['gemma-3-27b', { sliding: 52, full: 10 }, 507904],
  ['gemma-4-31b', { sliding: 50, full: 10 }, 860160],
  ['gpt-oss-20b', { sliding: 12, full: 12 }, 49152],
  ['deepseek-v3', { mla: 61 }, 70272],
  ['qwen3-next-80b', { linear: 36, full: 12 }, 24576],
]

for (const [name, mix, bytes] of CASES) {
  test(`parseConfig ${name}: layer mix and bytes per token`, () => {
    const m = loadModel(name)
    assert.deepEqual(layerMix(m), mix)
    assert.equal(structuredBytesPerToken(m, 'auto', 1), bytes)
  })
}

test('deepseek-v3: head_dim missing is noted and dense formula is 24.9x too large', () => {
  const m = loadModel('deepseek-v3')
  assert.ok(m.notes.some((n) => n.key === 'note.headDimMissing'))
  assert.equal(denseBytesPerToken(m, 'auto'), 1748992)
  assert.equal((denseBytesPerToken(m, 'auto') / structuredBytesPerToken(m, 'auto')).toFixed(1), '24.9')
  assert.equal(m.confidence, 'high')
})

test('qwen3-next: linear state per request and medium confidence', () => {
  const m = loadModel('qwen3-next-80b')
  assert.equal(fixedStateBytes(m), 39518208)
  assert.equal(m.confidence, 'medium')
})

test('gemma-4: global layers use their own head count and dim, flagged medium', () => {
  const m = loadModel('gemma-4-31b')
  const g = m.layers.find((l) => l.kind === 'full')
  assert.deepEqual([g?.kvHeads, g?.kDim, g?.vDim], [4, 512, 0])
  assert.equal(m.confidence, 'medium')
  assert.ok(m.notes.some((n) => n.key === 'note.gemma4Global'))
})

test('gemma-3: layer types derived from sliding_window_pattern', () => {
  const m = loadModel('gemma-3-27b')
  assert.ok(m.notes.some((n) => n.key === 'note.fromPattern'))
  assert.equal(m.layers[5]?.kind, 'full')
  assert.equal(m.layers[0]?.window, 1024)
})

test('use_sliding_window: false turns every layer into full attention', () => {
  const m = parseConfig({
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    hidden_size: 512,
    sliding_window: 4096,
    use_sliding_window: false,
    max_position_embeddings: 32768,
  })
  assert.deepEqual(layerMix(m), { full: 4 })
})

test('only sliding_window given (Mistral style): all layers slide', () => {
  const m = parseConfig({
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    hidden_size: 512,
    sliding_window: 4096,
    max_position_embeddings: 32768,
  })
  assert.deepEqual(layerMix(m), { sliding: 4 })
})

test('missing required fields throw CONFIG_INVALID with the field name', () => {
  const bad = loadConfig('llama-3.1-8b') as Record<string, unknown>
  delete bad['num_hidden_layers']
  assert.throws(
    () => parseConfig(bad),
    (e: unknown) => e instanceof KvroomError && e.code === 'CONFIG_INVALID' && e.details['field'] === 'num_hidden_layers',
  )
  assert.throws(() => parseConfig([]), (e: unknown) => e instanceof KvroomError && e.code === 'CONFIG_INVALID')
  assert.throws(() => parseConfig({ num_hidden_layers: 2, num_attention_heads: 4 }), (e: unknown) => e instanceof KvroomError)
})
