import { useState } from 'react'
import type { DragEvent } from 'react'
import { bytesPerTokenPerRank } from '../core/tp.ts'
import { denseBytesPerToken, structuredBytesPerToken } from '../core/vllm.ts'
import { fixedStateBytes, layerMix } from '../core/parse.ts'
import type { LayerKind, ParsedModel } from '../core/parse.ts'
import { PRESETS } from '../data/presets.ts'
import { t } from '../i18n/index.ts'
import { PLACEHOLDER_MODEL } from './compute.ts'
import type { Lang, MessageKey } from '../i18n/index.ts'

export type ModelSource = 'hub' | 'preset' | 'paste'

interface Props {
  lang: Lang
  repo: string
  model: ParsedModel | null
  source: ModelSource | null
  loading: boolean
  error: string | null
  weightGB: number
  onLoad: (repo: string) => void
  onPreset: (key: string) => void
  /** Returns the model id the flags now use, or null when the paste failed. */
  onPaste: (text: string, repoText: string, repoEdited: boolean) => string | null
  onWeightGB: (gb: number) => void
}

const fmt = (n: number): string => n.toLocaleString('en-US')

export function ModelPanel(p: Props) {
  const [repoText, setRepoText] = useState(p.repo)
  // 상자는 불러온 모델 ID 로 미리 채워진다. 사용자가 직접 고쳤을 때만 붙여넣기에 그 ID 를 쓴다.
  const [repoEdited, setRepoEdited] = useState(false)
  const [paste, setPaste] = useState('')
  const lang = p.lang

  const applyPaste = (text: string) => {
    const used = p.onPaste(text, repoText, repoEdited)
    if (used === null) return
    // 상자가 플래그에 실제로 들어가는 ID 를 보여 주게 한다(모르면 비워 둔다).
    setRepoText(used === PLACEHOLDER_MODEL ? '' : used)
    setRepoEdited(false)
  }

  const onDrop = (e: DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (!file) return
    file
      .text()
      .then((text) => {
        setPaste(text)
        applyPaste(text)
      })
      .catch(() => undefined)
  }

  const mix = p.model ? layerMix(p.model) : {}
  return (
    <section className="panel">
      <h2>{t(lang, 'model.title')}</h2>
      <div className="row">
        <label className="grow">
          {t(lang, 'model.repo')}
          <input
            value={repoText}
            onChange={(e) => {
              setRepoText(e.target.value)
              setRepoEdited(true)
            }}
            placeholder="openai/gpt-oss-20b"
            spellCheck={false}
          />
        </label>
        <button type="button" disabled={p.loading} onClick={() => p.onLoad(repoText)}>
          {p.loading ? t(lang, 'model.loading') : t(lang, 'model.load')}
        </button>
      </div>
      <div className="chips" aria-label={t(lang, 'model.presets')}>
        {PRESETS.map((pr) => (
          <button
            type="button"
            key={pr.key}
            className="chip"
            onClick={() => {
              setRepoText(pr.repo)
              setRepoEdited(false)
              p.onPreset(pr.key)
            }}
          >
            {pr.name}
          </button>
        ))}
      </div>
      <label>
        {t(lang, 'model.paste')}
        <textarea
          rows={3}
          value={paste}
          onChange={(e) => setPaste(e.target.value)}
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          placeholder='{ "num_hidden_layers": 32, ... }'
          spellCheck={false}
        />
      </label>
      <p className="hint">{t(lang, 'model.pasteHint')}</p>
      <button type="button" disabled={paste.trim() === ''} onClick={() => applyPaste(paste)}>
        {t(lang, 'model.apply')}
      </button>

      {p.error ? <p className="error">{p.error}</p> : null}

      {p.model ? (
        <div className="model-info">
          <p className="hint">{t(lang, `model.source.${p.source ?? 'paste'}` as MessageKey)}</p>
          <dl>
            <dt>{t(lang, 'model.layers')}</dt>
            <dd>
              {(Object.keys(mix) as LayerKind[]).map((k) => (
                <span key={k} className={`badge kind-${k}`}>
                  {mix[k]} x {t(lang, `model.layerKind.${k}` as MessageKey)}
                </span>
              ))}
            </dd>
            <dt>{t(lang, 'model.bytesPerToken')}</dt>
            <dd>{fmt(bytesPerTokenPerRank(p.model, p.model.modelDtypeBytes, 1))} B</dd>
            <dt>{t(lang, 'model.dense')}</dt>
            <dd>
              {fmt(denseBytesPerToken(p.model, 'auto'))} B
              {denseBytesPerToken(p.model, 'auto') !== structuredBytesPerToken(p.model, 'auto') ? (
                <span className="hint"> ({(denseBytesPerToken(p.model, 'auto') / structuredBytesPerToken(p.model, 'auto')).toFixed(2)}x)</span>
              ) : null}
            </dd>
            {fixedStateBytes(p.model) > 0 ? (
              <>
                <dt>{t(lang, 'model.state')}</dt>
                <dd>{fmt(fixedStateBytes(p.model))} B</dd>
              </>
            ) : null}
            <dt>{t(lang, 'model.maxPosition')}</dt>
            <dd>{fmt(p.model.maxPosition)}</dd>
            <dt>{t(lang, 'model.confidence')}</dt>
            <dd>
              <span className={`badge conf-${p.model.confidence}`}>{t(lang, `model.confidence.${p.model.confidence}` as MessageKey)}</span>
            </dd>
          </dl>
          <ul className="notes">
            {p.model.notes.map((n, i) => (
              <li key={i} className={n.level}>
                {t(lang, n.key as MessageKey, n.vars)}
              </li>
            ))}
          </ul>
          <label>
            {t(lang, 'model.weights')}
            <input type="number" min={0} step={0.1} value={p.weightGB} onChange={(e) => p.onWeightGB(Number(e.target.value))} />
          </label>
          <p className="hint">{t(lang, 'model.weightsHint')}</p>
          {p.source === 'paste' && p.weightGB === 0 ? <p className="hint">{t(lang, 'model.setWeights')}</p> : null}
        </div>
      ) : null}
    </section>
  )
}
