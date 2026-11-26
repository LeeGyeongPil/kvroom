import assert from 'node:assert/strict'
import { test } from 'node:test'
import { ERROR_CODES, KvroomError } from '../src/core/errors.ts'
import { parseConfig } from '../src/core/parse.ts'
import { en } from '../src/i18n/en.ts'
import { errorText, LANGS, loadStoredLang, MESSAGES, resolveLang, saveLang, STORAGE_KEY, t } from '../src/i18n/index.ts'
import type { MessageKey } from '../src/i18n/index.ts'
import { ko } from '../src/i18n/ko.ts'
import { loadModel } from './helpers.ts'

test('en and ko have the same keys and no empty text', () => {
  assert.deepEqual(Object.keys(en).sort(), Object.keys(ko).sort())
  for (const lang of LANGS) {
    for (const [key, text] of Object.entries(MESSAGES[lang])) {
      assert.ok(text.trim().length > 0, `${lang}:${key} is empty`)
    }
  }
})

test('both languages use the same {placeholders} in every message', () => {
  const names = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',')
  for (const key of Object.keys(en) as MessageKey[]) {
    assert.equal(names(en[key]), names(ko[key]), key)
  }
})

test('resolveLang: query beats saved value beats browser language', () => {
  assert.equal(resolveLang('ko', 'en', ['en-US']), 'ko')
  assert.equal(resolveLang('en', 'ko', ['ko-KR']), 'en')
  assert.equal(resolveLang(null, 'ko', ['en-US']), 'ko')
  assert.equal(resolveLang(null, 'en', ['ko-KR']), 'en')
  assert.equal(resolveLang(null, null, ['ko-KR', 'en']), 'ko')
  assert.equal(resolveLang(null, null, ['ko']), 'ko')
  assert.equal(resolveLang(null, null, ['en-US']), 'en')
  assert.equal(resolveLang(null, null, ['fr-FR']), 'en')
  assert.equal(resolveLang(null, null, []), 'en')
  assert.equal(resolveLang('xx', 'yy', ['ja']), 'en')
})

test('the chosen language is saved and read back; storage failures are ignored', () => {
  const map = new Map<string, string>()
  const storage = { getItem: (k: string) => map.get(k) ?? null, setItem: (k: string, v: string) => void map.set(k, v) }
  assert.equal(loadStoredLang(storage), null)
  saveLang(storage, 'ko')
  assert.equal(map.get(STORAGE_KEY), 'ko')
  assert.equal(resolveLang(null, loadStoredLang(storage), ['en-US']), 'ko')
  const broken = {
    getItem: (): string | null => {
      throw new Error('blocked')
    },
    setItem: (): void => {
      throw new Error('blocked')
    },
  }
  assert.equal(loadStoredLang(broken), null)
  assert.doesNotThrow(() => saveLang(broken, 'en'))
})

test('switching language really changes the text and fills placeholders', () => {
  assert.notEqual(t('en', 'flags.copy'), t('ko', 'flags.copy'))
  assert.equal(t('en', 'summary.maxLenFor', { n: 8, len: 4656 }), 'Longest length for 8 concurrent requests: 4656 tokens')
  assert.ok(t('ko', 'summary.maxLenFor', { n: 8, len: 4656 }).includes('4656'))
  assert.equal(t('en', 'summary.maxLenFor'), 'Longest length for {n} concurrent requests: {len} tokens')
})

test('every error code has a message in both languages', () => {
  for (const code of ERROR_CODES) {
    for (const lang of LANGS) {
      assert.ok(MESSAGES[lang][`error.${code}` as MessageKey], `${lang}: error.${code}`)
    }
  }
  const err = new KvroomError('TP_NOT_DIVISIBLE', 'x', { heads: 64, tp: 3 })
  assert.ok(errorText('en', err).includes('64'))
  assert.ok(errorText('ko', err).includes('64'))
  assert.notEqual(errorText('en', err), errorText('ko', err))
})

test('every parser note key has a message', () => {
  const keys = new Set<string>()
  for (const name of ['deepseek-v3', 'gemma-3-27b', 'gemma-4-31b', 'qwen3-next-80b', 'gpt-oss-20b']) {
    for (const n of loadModel(name).notes) keys.add(n.key)
  }
  keys.add(parseConfig({ num_hidden_layers: 1, num_attention_heads: 2, head_dim: 8 }).notes[0]?.key ?? '')
  keys.add('note.allSliding')
  keys.add('note.slidingDisabled')
  for (const k of keys) assert.ok(k in en, k)
})

test('pasted-weights hint exists in both languages', () => {
  assert.ok(en['model.setWeights'])
  assert.ok(MESSAGES.ko['model.setWeights'])
})
