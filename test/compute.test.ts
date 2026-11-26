// The screen logic (form values -> numbers -> flags) without React.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseVllmLog } from '../src/core/logparse.ts'
import { capacityTable, defaultLens } from '../src/core/table.ts'
import { buildCommands, compute } from '../src/ui/compute.ts'
import type { Common, Setup } from '../src/ui/compute.ts'
import { loadLog, loadModel } from './helpers.ts'

const common: Common = {
  engine: 'vllm',
  maxNumBatchedTokens: 8128,
  blockSize: 16,
  pageSize: 1,
  swaRatio: 0.8,
  utilization: 0.9,
  reserveGiB: 2,
  budgetMode: 'fixed',
  fixedGiB: 4.17,
  len: 128000,
  seqs: 2,
  weightBytes: 13.76e9,
  logFacts: null,
  tokenDefinition: 'auto',
}
const l4: Setup = { gpu: 'L4', customGB: 24, tp: 1, kvDtype: 'fp8' }

test('scenario 1: gpt-oss-20b on L4 with a fixed 4.17 GiB budget', () => {
  const m = loadModel('gpt-oss-20b')
  const c = compute(m, common, l4)
  assert.equal(c.error, null)
  assert.equal(c.maxSeqs, 2)
  assert.equal(c.maxLenForSeqs, 131072) // two requests fit even at the model maximum, so the search hits its cap
})

test('scenario 1b: longest length for 4 concurrent requests is 82,816 tokens', () => {
  const m = loadModel('gpt-oss-20b')
  const c = compute(m, { ...common, seqs: 4 }, l4)
  assert.equal(c.maxLenForSeqs, 82816)
  const cmds = buildCommands('openai/gpt-oss-20b', m, { ...common, seqs: 4, len: 82816 }, l4, c)
  assert.ok(cmds.vllm.includes('--max-num-seqs 4'))
  assert.ok(cmds.vllm.includes('--kv-cache-memory-bytes 4477'), cmds.vllm)
  assert.equal(cmds.pinned, true)
})

test('estimate mode: GPU x utilization - weights/TP - reserve', () => {
  const m = loadModel('gpt-oss-20b')
  const c = compute(m, { ...common, budgetMode: 'estimate' }, l4)
  const gib = 1024 ** 3
  const gpu = (24e9 / gib) * gib
  const expected = Math.floor(gpu * 0.9 - 13.76e9 - 2 * gib)
  assert.equal(c.budget?.kvBytesPerRank, expected)
  assert.equal(c.budget?.source, 'estimate')
})

test('TP that cannot split the heads gives an error result instead of throwing', () => {
  const c = compute(loadModel('gpt-oss-20b'), common, { ...l4, tp: 3 })
  assert.equal(c.error?.code, 'TP_NOT_DIVISIBLE')
  assert.equal(c.table.length, 0)
})

test('SGLang with a tiny budget reports the SWA pool error', () => {
  const c = compute(loadModel('gpt-oss-20b'), { ...common, engine: 'sglang', fixedGiB: 0.001 }, l4)
  assert.equal(c.error?.code, 'SGLANG_SWA_POOL_TOO_SMALL')
})

test('log mode: budget comes from the pasted log and reproduces the logged concurrency', () => {
  const facts = parseVllmLog(loadLog('receipt-b'))
  const c = compute(loadModel('gpt-oss-20b'), { ...common, budgetMode: 'log', logFacts: facts }, l4)
  assert.equal(c.logBudget?.numBlocks, 22792)
  assert.equal(c.vllm?.maxConcurrency.toFixed(2), '2.68')
})

test('SGLang flags pin --max-total-tokens in fixed mode and include the SWA ratio for hybrid models', () => {
  const m = loadModel('gpt-oss-20b')
  const c = compute(m, { ...common, engine: 'sglang' }, l4)
  const cmds = buildCommands('openai/gpt-oss-20b', m, { ...common, engine: 'sglang' }, l4, c)
  assert.match(cmds.sglang, /--swa-full-tokens-ratio 0\.8/)
  assert.match(cmds.sglang, /--max-total-tokens \d+/)
})

test('table rows are ordered lengths and cells are monotonic in concurrency', () => {
  const m = loadModel('llama-3.1-8b')
  const c = compute(m, { ...common, fixedGiB: 10 }, { ...l4, kvDtype: 'auto' })
  assert.deepEqual(c.lens, defaultLens(131072))
  for (const row of c.table) {
    let seenNo = false
    for (const cell of row) {
      if (!cell.ok) seenNo = true
      else assert.equal(seenNo, false, 'a bigger concurrency fits after a smaller one failed')
    }
  }
  assert.ok(capacityTable('vllm', c.settings!, [4096], [1]).length === 1)
})

// --- Round 1 fixes: input validation (Q-2) and pasted configs (Q-1) ---
import { commonForModel, describeFacts, INPUT_LABELS, modelFingerprint, PLACEHOLDER_MODEL, pasteTarget } from '../src/ui/compute.ts'
import type { InputField } from '../src/ui/compute.ts'
import { KvroomError } from '../src/core/errors.ts'
import { parseConfig } from '../src/core/parse.ts'
import { errorText, rangeText, t } from '../src/i18n/index.ts'

const bad: Array<[string, Partial<Common>, Partial<Setup>]> = [
  ['blockSize', { blockSize: 0 }, {}],
  ['pageSize', { pageSize: 0 }, {}],
  ['maxNumBatchedTokens', { maxNumBatchedTokens: 0 }, {}],
  ['utilization', { utilization: 5 }, {}],
  ['utilization', { utilization: 0 }, {}],
  ['swaRatio', { swaRatio: 0 }, {}],
  ['swaRatio', { swaRatio: 5 }, {}],
  ['fixedGiB', { fixedGiB: 0 }, {}],
  ['customGB', {}, { customGB: 0 }],
  ['len', { len: 0 }, {}],
  ['seqs', { seqs: 0 }, {}],
  ['blockSize', { blockSize: Number.NaN }, {}],
]

test('invalid numbers give INPUT_OUT_OF_RANGE and no NaN flags', () => {
  const m = loadModel('llama-3.1-8b')
  for (const [field, cp, sp] of bad) {
    const c = { ...common, ...cp }
    const s = { ...l4, ...sp }
    const comp = compute(m, c, s)
    assert.equal(comp.error?.code, 'INPUT_OUT_OF_RANGE', field)
    assert.equal(comp.error?.details.field, field)
    assert.equal(comp.table.length, 0)
    assert.throws(() => buildCommands('a/b', m, c, s, comp), (e: unknown) => e instanceof KvroomError && e.code === 'INPUT_OUT_OF_RANGE')
    // The message names the on-screen label, not the internal key (Q-8).
    const label = INPUT_LABELS[field as InputField]
    assert.equal(comp.error?.details.labelKey, label)
    for (const lang of ['en', 'ko'] as const) {
      const text = errorText(lang, comp.error)
      assert.ok(text.includes(t(lang, label)), `${lang}: ${text}`)
      assert.ok(!text.includes(`"${field}"`), `${lang}: internal key shown: ${text}`)
    }
  }
})

// Korean text may keep digits and symbols. The form label itself is checked separately (some Korean
// labels are engine flag names or units on purpose: max-num-batched-tokens, KV, GiB); once the label
// is removed, nothing in Latin letters may remain.

test('every validated field gives a fully Korean message in ko and the English label in en', () => {
  const fields = Object.keys(INPUT_LABELS) as InputField[]
  const m = loadModel('llama-3.1-8b')
  const breakOne: Record<InputField, [Partial<Common>, Partial<Setup>]> = {
    blockSize: [{ blockSize: 0 }, {}],
    pageSize: [{ pageSize: 1.5 }, {}],
    maxNumBatchedTokens: [{ maxNumBatchedTokens: -1 }, {}],
    len: [{ len: 0 }, {}],
    seqs: [{ seqs: 0.5 }, {}],
    tp: [{}, { tp: 0 }],
    utilization: [{ utilization: 1.5 }, {}],
    swaRatio: [{ swaRatio: 0 }, {}],
    reserveGiB: [{ reserveGiB: -1 }, {}],
    weightBytes: [{ weightBytes: -1 }, {}],
    customGB: [{}, { customGB: 0 }],
    fixedGiB: [{ fixedGiB: 0 }, {}],
  }
  for (const field of fields) {
    const [cp, sp] = breakOne[field]
    const comp = compute(m, { ...common, ...cp }, { ...l4, ...sp })
    assert.equal(comp.error?.details.field, field)
    const ko = errorText('ko', comp.error)
    const en = errorText('en', comp.error)
    assert.ok(ko.includes(t('ko', INPUT_LABELS[field])), ko)
    assert.ok(en.includes(t('en', INPUT_LABELS[field])), en)
    assert.ok(!/[A-Za-z]/.test(ko.replace(t('ko', INPUT_LABELS[field]), '')), `English left in ko: ${ko}`)
    assert.ok(!/이상|초과|이하|정수/.test(en), `Korean left in en: ${en}`)
  }
})

test('range text is translated: utilization > 0 and <= 1, block size whole number >= 1', () => {
  const util = compute(loadModel('llama-3.1-8b'), { ...common, utilization: 5 }, l4).error
  assert.ok(errorText('ko', util).includes('0 초과, 1 이하'), errorText('ko', util))
  assert.ok(errorText('en', util).includes('more than 0, 1 or less'), errorText('en', util))
  const block = compute(loadModel('llama-3.1-8b'), { ...common, blockSize: 0 }, l4).error
  assert.ok(errorText('ko', block).includes('1 이상, 정수만'), errorText('ko', block))
  assert.ok(errorText('en', block).includes('1 or more, whole numbers only'), errorText('en', block))
  assert.equal(rangeText('ko', { min: 0, exclusiveMin: 1, max: 1, integer: 0 }), '0 초과, 1 이하')
})

test('invalid JSON and a non-object config get their own translated messages, not "(JSON)"', () => {
  const notJson = new KvroomError('CONFIG_INVALID', 'Not valid JSON.', { field: '(JSON)' })
  let notObject: unknown = null
  try {
    parseConfig([1, 2])
  } catch (e) {
    notObject = e
  }
  for (const lang of ['en', 'ko'] as const) {
    assert.ok(!errorText(lang, notJson).includes('(JSON)'))
    assert.ok(!errorText(lang, notObject).includes('(root)'))
  }
  assert.equal(errorText('ko', notJson), t('ko', 'error.CONFIG_NOT_JSON'))
  assert.equal(errorText('en', notObject), t('en', 'error.CONFIG_NOT_OBJECT'))
})

test('valid numbers still produce flags without NaN', () => {
  const m = loadModel('llama-3.1-8b')
  const comp = compute(m, common, l4)
  assert.equal(comp.error, null)
  const cmd = buildCommands('a/b', m, common, l4, comp)
  assert.ok(!cmd.vllm.includes('NaN') && !cmd.sglang.includes('NaN'))
})

// The screen starts on the gpt-oss-20b preset, so the id box is prefilled with its id.
const gptOss = () => ({ repo: 'openai/gpt-oss-20b', weightBytes: 13.76e9, model: loadModel('gpt-oss-20b') })

test('untouched id box + different config -> placeholder id and weight 0 (QA round-2 Q-7)', () => {
  const llama = loadModel('llama-3.1-8b')
  const target = pasteTarget({ repoText: 'openai/gpt-oss-20b', repoEdited: false, pasted: llama, previous: gptOss() })
  assert.deepEqual(target, { repo: PLACEHOLDER_MODEL, weightBytes: 0, sameModel: false })
  // The flags carry no trace of the previous model, and the weight box is 0 so the screen asks for it.
  const c = { ...common, budgetMode: 'estimate' as const, len: 8192, weightBytes: target.weightBytes }
  const cmd = buildCommands(target.repo, llama, c, l4, compute(llama, c, l4))
  assert.ok(!cmd.vllm.includes('gpt-oss') && !cmd.sglang.includes('gpt-oss'), cmd.vllm)
  assert.ok(cmd.vllm.startsWith(`vllm serve ${PLACEHOLDER_MODEL} `), cmd.vllm)
  // Estimate mode no longer subtracts the 13.76 GB of gpt-oss weights.
  assert.equal(compute(llama, c, l4).budget?.weightBytesPerRank, 0)
})

test('typing the previous id by hand still does not attach it to another model', () => {
  const llama = loadModel('llama-3.1-8b')
  const t1 = pasteTarget({ repoText: ' openai/gpt-oss-20b ', repoEdited: true, pasted: llama, previous: gptOss() })
  assert.deepEqual(t1, { repo: PLACEHOLDER_MODEL, weightBytes: 0, sameModel: false })
  const t2 = pasteTarget({ repoText: 'OpenAI/GPT-OSS-20B', repoEdited: true, pasted: llama, previous: gptOss() })
  assert.equal(t2.repo, PLACEHOLDER_MODEL)
})

test('id typed after the last load is used for a different config (gated-model flow)', () => {
  const llama = loadModel('llama-3.1-8b')
  const target = pasteTarget({ repoText: 'meta-llama/Llama-3.1-8B-Instruct', repoEdited: true, pasted: llama, previous: gptOss() })
  assert.deepEqual(target, { repo: 'meta-llama/Llama-3.1-8B-Instruct', weightBytes: 0, sameModel: false })
  const cmd = buildCommands(target.repo, llama, common, l4, compute(llama, common, l4))
  assert.ok(cmd.vllm.includes('meta-llama/Llama-3.1-8B-Instruct') && !cmd.vllm.includes('gpt-oss'))
  // Not an owner/name id -> placeholder.
  assert.equal(pasteTarget({ repoText: 'not an id', repoEdited: true, pasted: llama, previous: gptOss() }).repo, PLACEHOLDER_MODEL)
  assert.deepEqual(pasteTarget({ repoText: '', repoEdited: false, pasted: llama, previous: null }), { repo: PLACEHOLDER_MODEL, weightBytes: 0, sameModel: false })
})

test('a stale id from an earlier load (box not edited) is never reused after a placeholder paste', () => {
  // Paste 1 left the screen on "<your-model>" (llama structure); the box text is still the old preset id.
  const llama = loadModel('llama-3.1-8b')
  const afterPaste = { repo: PLACEHOLDER_MODEL, weightBytes: 0, model: llama }
  const t2 = pasteTarget({ repoText: 'openai/gpt-oss-20b', repoEdited: false, pasted: loadModel('qwen3-8b'), previous: afterPaste })
  assert.equal(t2.repo, PLACEHOLDER_MODEL)
})

test('same structure as the loaded model keeps its id and weight size', () => {
  // Re-pasting the loaded model's own config (for example after editing a non-structural field).
  const same = pasteTarget({ repoText: 'openai/gpt-oss-20b', repoEdited: false, pasted: loadModel('gpt-oss-20b'), previous: gptOss() })
  assert.deepEqual(same, { repo: 'openai/gpt-oss-20b', weightBytes: 13.76e9, sameModel: true })
  // A fine-tune with the same structure and a typed id: new id, same weight size.
  const tuned = pasteTarget({ repoText: 'someone/gpt-oss-20b-tuned', repoEdited: true, pasted: loadModel('gpt-oss-20b'), previous: gptOss() })
  assert.deepEqual(tuned, { repo: 'someone/gpt-oss-20b-tuned', weightBytes: 13.76e9, sameModel: true })
})

test('fingerprint tells models apart and ignores parser notes', () => {
  const names = ['deepseek-v3', 'gemma-3-27b', 'gemma-4-31b', 'gpt-oss-20b', 'llama-3.1-8b', 'llama-3.3-70b', 'qwen3-8b', 'qwen3-next-80b']
  const prints = new Set(names.map((n) => modelFingerprint(loadModel(n))))
  assert.equal(prints.size, names.length)
  const m = loadModel('llama-3.1-8b')
  assert.equal(modelFingerprint({ ...m, notes: [] }), modelFingerprint(m))
})

test('a vLLM log read for one model is dropped when a different model is loaded', () => {
  const facts = parseVllmLog(loadLog('receipt-b'))
  const withLog: Common = { ...common, budgetMode: 'log', logFacts: facts }
  const gpt = loadModel('gpt-oss-20b')
  const llama = loadModel('llama-3.1-8b')
  const changed = commonForModel(withLog, gpt, llama)
  assert.equal(changed.logCleared, true)
  assert.equal(changed.patch.logFacts, null)
  assert.equal(changed.patch.budgetMode, 'estimate')
  // Same structure (re-load of the same model): the log stays.
  const same = commonForModel(withLog, gpt, loadModel('gpt-oss-20b'))
  assert.equal(same.logCleared, false)
  assert.ok(!('logFacts' in same.patch))
  // No log: nothing to clear, but the length is still capped at the new maximum.
  const noLog = commonForModel({ ...common, len: 1_000_000 }, gpt, llama)
  assert.equal(noLog.logCleared, false)
  assert.equal(noLog.patch.len, llama.maxPosition)
})

test('log facts are described in the screen language', () => {
  const facts = parseVllmLog(loadLog('receipt-b'))
  const ko = describeFacts('ko', facts)
  const en = describeFacts('en', facts)
  assert.ok(!/blocks|tokens/.test(ko), ko)
  assert.ok(ko.includes('토큰'), ko)
  assert.ok(en.includes('tokens'), en)
})
