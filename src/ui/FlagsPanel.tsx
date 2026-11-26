import { useState } from 'react'
import { t } from '../i18n/index.ts'
import type { Lang } from '../i18n/index.ts'
import type { Engine } from '../core/table.ts'
import type { Commands } from './compute.ts'

interface Props {
  lang: Lang
  engine: Engine
  commands: Commands
}

function CopyBlock({ lang, title, text, active }: { lang: Lang; title: string; text: string; active: boolean }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      })
      .catch(() => undefined)
  }
  return (
    <div className={`flags ${active ? 'active' : ''}`}>
      <div className="flags-head">
        <strong>{title}</strong>
        <button type="button" onClick={copy}>
          {copied ? t(lang, 'flags.copied') : t(lang, 'flags.copy')}
        </button>
      </div>
      <pre>{text}</pre>
    </div>
  )
}

export function FlagsPanel({ lang, engine, commands }: Props) {
  return (
    <section className="panel">
      <h2>{t(lang, 'flags.title')}</h2>
      <CopyBlock lang={lang} title="vLLM" text={commands.vllm} active={engine === 'vllm'} />
      <CopyBlock lang={lang} title="SGLang" text={commands.sglang} active={engine === 'sglang'} />
      {commands.pinned ? null : <p className="hint">{t(lang, 'flags.pin')}</p>}
    </section>
  )
}
