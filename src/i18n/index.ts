// Language selection and message lookup. No DOM access except through the optional storage argument.
import { KvroomError } from '../core/errors.ts'
import { en } from './en.ts'
import type { MessageKey } from './en.ts'
import { ko } from './ko.ts'

export type Lang = 'en' | 'ko'
export const LANGS: readonly Lang[] = ['en', 'ko']
export const LANG_NAMES: Record<Lang, string> = { en: 'English', ko: '한국어' }
export const STORAGE_KEY = 'kvroom.lang'

export const MESSAGES: Record<Lang, Record<MessageKey, string>> = { en, ko }
export type { MessageKey }

function asLang(v: string | null | undefined): Lang | undefined {
  const s = v?.trim().toLowerCase()
  if (s === 'en' || s === 'ko') return s
  return undefined
}

/**
 * Pick the screen language. Order: `?lang=` query, then the saved choice, then the browser's top
 * language (Korean if it starts with `ko`), otherwise English.
 */
export function resolveLang(
  query: string | null | undefined,
  stored: string | null | undefined,
  navigatorLanguages: readonly string[] = [],
): Lang {
  const q = asLang(query)
  if (q) return q
  const s = asLang(stored)
  if (s) return s
  const first = navigatorLanguages[0]?.toLowerCase() ?? ''
  return first.startsWith('ko') ? 'ko' : 'en'
}

/** Minimal storage shape so tests can pass a Map-backed stub. */
export interface LangStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

export function loadStoredLang(storage: LangStorage | undefined): string | null {
  try {
    return storage?.getItem(STORAGE_KEY) ?? null
  } catch {
    return null
  }
}

export function saveLang(storage: LangStorage | undefined, lang: Lang): void {
  try {
    storage?.setItem(STORAGE_KEY, lang)
  } catch {
    // Private mode or blocked storage: the choice just is not remembered.
  }
}

/** Translate a message, filling `{name}` placeholders from `vars`. */
export function t(lang: Lang, key: MessageKey, vars: Record<string, string | number> = {}): string {
  const template = MESSAGES[lang][key]
  return template.replace(/\{(\w+)\}/g, (whole, name: string) => {
    const v = vars[name]
    return v === undefined ? whole : String(v)
  })
}

const isKey = (k: unknown): k is MessageKey => typeof k === 'string' && k in en

/** Allowed range in the screen language, from the numbers an `INPUT_OUT_OF_RANGE` error carries. */
export function rangeText(lang: Lang, d: Record<string, string | number>): string {
  const parts: string[] = []
  if (d.min !== undefined) parts.push(t(lang, d.exclusiveMin === 1 ? 'range.gt' : 'range.min', { min: d.min }))
  if (d.max !== undefined) parts.push(t(lang, 'range.max', { max: d.max }))
  if (d.integer === 1) parts.push(t(lang, 'range.int'))
  return parts.join(', ')
}

/** Translated text for a thrown error: `KvroomError` codes map to `error.<CODE>`. */
export function errorText(lang: Lang, err: unknown): string {
  if (err instanceof KvroomError) {
    const d = err.details
    if (err.code === 'INPUT_OUT_OF_RANGE') {
      // 내부 키(blockSize 등) 대신 화면 라벨과 번역된 범위를 넣는다.
      const field = isKey(d.labelKey) ? t(lang, d.labelKey) : String(d.field ?? '')
      return t(lang, 'error.INPUT_OUT_OF_RANGE', { field, range: rangeText(lang, d) })
    }
    if (err.code === 'CONFIG_INVALID' && d.field === '(JSON)') return t(lang, 'error.CONFIG_NOT_JSON')
    if (err.code === 'CONFIG_INVALID' && d.field === '(root)') return t(lang, 'error.CONFIG_NOT_OBJECT')
    return t(lang, `error.${err.code}` as MessageKey, d)
  }
  return err instanceof Error ? err.message : String(err)
}
