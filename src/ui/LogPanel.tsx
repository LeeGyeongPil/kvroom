import { useState } from 'react'
import { parseVllmLog } from '../core/logparse.ts'
import type { LogBudget, LogFacts, TokenDefinition } from '../core/logparse.ts'
import { t } from '../i18n/index.ts'
import type { Lang, MessageKey } from '../i18n/index.ts'
import { describeFacts } from './compute.ts'

interface Props {
  lang: Lang
  facts: LogFacts | null
  logBudget: LogBudget | null
  tokenDefinition: TokenDefinition
  /** True when a log was dropped because a model with a different structure was loaded. */
  cleared: boolean
  onFacts: (facts: LogFacts | null) => void
  onTokenDefinition: (d: TokenDefinition) => void
}

export function LogPanel({ lang, facts, logBudget, tokenDefinition, cleared, onFacts, onTokenDefinition }: Props) {
  const [text, setText] = useState('')
  const [empty, setEmpty] = useState(false)

  const read = () => {
    const f = parseVllmLog(text)
    const found = Object.keys(f).length > 0
    setEmpty(!found)
    onFacts(found ? f : null)
  }

  return (
    <section className="panel">
      <h2>{t(lang, 'log.title')}</h2>
      <p className="hint">{t(lang, 'log.hint')}</p>
      <textarea
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="INFO ... GPU KV cache size: 182,336 tokens"
        spellCheck={false}
      />
      <div className="row">
        <button type="button" disabled={text.trim() === ''} onClick={read}>
          {t(lang, 'log.apply')}
        </button>
        <button
          type="button"
          onClick={() => {
            setText('')
            setEmpty(false)
            onFacts(null)
          }}
        >
          {t(lang, 'log.clear')}
        </button>
        <label className="inline">
          {t(lang, 'log.def')}
          <select value={tokenDefinition} onChange={(e) => onTokenDefinition(e.target.value as TokenDefinition)}>
            {(['auto', 'v012', 'current'] as const).map((d) => (
              <option key={d} value={d}>
                {t(lang, `log.def.${d}` as MessageKey)}
              </option>
            ))}
          </select>
        </label>
      </div>
      {empty ? <p className="error">{t(lang, 'log.none')}</p> : null}
      {cleared ? <p className="hint">{t(lang, 'log.clearedOnModel')}</p> : null}
      {facts ? <p>{t(lang, 'log.found', { facts: describeFacts(lang, facts) })}</p> : null}
      {logBudget ? <p>{t(lang, 'log.basis', { basis: t(lang, `log.basis.${logBudget.basis}` as MessageKey) })}</p> : null}
    </section>
  )
}
