import { KV_DTYPES } from '../core/dtype.ts'
import { GPUS } from '../data/gpus.ts'
import { t } from '../i18n/index.ts'
import type { Lang, MessageKey } from '../i18n/index.ts'
import type { BudgetMode, Common, Setup } from './compute.ts'

interface Props {
  lang: Lang
  common: Common
  setup: Setup
  onCommon: (patch: Partial<Common>) => void
  onSetup: (patch: Partial<Setup>) => void
}

const TP_CHOICES = [1, 2, 3, 4, 8, 16]

function NumberField(props: { label: string; value: number; min?: number; step?: number; onChange: (n: number) => void }) {
  return (
    <label>
      {props.label}
      <input
        type="number"
        min={props.min ?? 0}
        step={props.step ?? 1}
        value={props.value}
        onChange={(e) => {
          const n = Number(e.target.value)
          if (Number.isFinite(n)) props.onChange(n)
        }}
      />
    </label>
  )
}

export function ServingPanel({ lang, common, setup, onCommon, onSetup }: Props) {
  const modes: BudgetMode[] = common.engine === 'vllm' ? ['estimate', 'fixed', 'log'] : ['estimate', 'fixed']
  return (
    <section className="panel">
      <h2>{t(lang, 'serving.title')}</h2>
      <div className="grid">
        <label>
          {t(lang, 'serving.engine')}
          <select
            value={common.engine}
            onChange={(e) => {
              const engine = e.target.value === 'sglang' ? 'sglang' : 'vllm'
              onCommon({ engine, ...(engine === 'sglang' && common.budgetMode === 'log' ? { budgetMode: 'estimate' } : {}) })
            }}
          >
            <option value="vllm">vLLM</option>
            <option value="sglang">SGLang</option>
          </select>
        </label>
        <label>
          {t(lang, 'serving.gpu')}
          <select value={setup.gpu} onChange={(e) => onSetup({ gpu: e.target.value })}>
            {GPUS.map((g) => (
              <option key={g.name} value={g.name}>
                {g.name} ({g.memoryGB} GB)
              </option>
            ))}
            <option value="custom">{t(lang, 'serving.gpuCustom')}</option>
          </select>
        </label>
        {setup.gpu === 'custom' ? (
          <NumberField label={t(lang, 'serving.gpuCustom')} value={setup.customGB} min={1} onChange={(n) => onSetup({ customGB: n })} />
        ) : null}
        <label>
          {t(lang, 'serving.count')}
          <select value={setup.tp} onChange={(e) => onSetup({ tp: Number(e.target.value) })}>
            {[...new Set([...TP_CHOICES, setup.tp])].sort((a, b) => a - b).map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t(lang, 'serving.kvDtype')}
          <select value={setup.kvDtype} onChange={(e) => onSetup({ kvDtype: e.target.value })}>
            {KV_DTYPES.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        {common.engine === 'vllm' ? (
          <>
            <NumberField label={t(lang, 'serving.batched')} value={common.maxNumBatchedTokens} min={1} step={256} onChange={(n) => onCommon({ maxNumBatchedTokens: n })} />
            <NumberField label={t(lang, 'serving.blockSize')} value={common.blockSize} min={1} onChange={(n) => onCommon({ blockSize: n })} />
          </>
        ) : (
          <>
            <NumberField label={t(lang, 'serving.pageSize')} value={common.pageSize} min={1} onChange={(n) => onCommon({ pageSize: n })} />
            <NumberField label={t(lang, 'serving.swaRatio')} value={common.swaRatio} min={0.05} step={0.05} onChange={(n) => onCommon({ swaRatio: n })} />
          </>
        )}
      </div>

      <fieldset>
        <legend>{t(lang, 'serving.budget')}</legend>
        {modes.map((m) => (
          <label key={m} className="radio">
            <input type="radio" name="budget" checked={common.budgetMode === m} onChange={() => onCommon({ budgetMode: m })} />
            {t(lang, `serving.budget.${m}` as MessageKey)}
          </label>
        ))}
        {common.engine === 'sglang' ? <p className="hint">{t(lang, 'serving.vllmOnlyLog')}</p> : null}
        {common.budgetMode === 'estimate' ? (
          <div className="grid">
            <NumberField label={t(lang, 'serving.util')} value={common.utilization} min={0.1} step={0.01} onChange={(n) => onCommon({ utilization: n })} />
            <NumberField label={t(lang, 'serving.reserve')} value={common.reserveGiB} min={0} step={0.5} onChange={(n) => onCommon({ reserveGiB: n })} />
          </div>
        ) : null}
        {common.budgetMode === 'fixed' ? (
          <NumberField label={t(lang, 'serving.fixedGiB')} value={common.fixedGiB} min={0.1} step={0.5} onChange={(n) => onCommon({ fixedGiB: n })} />
        ) : null}
        {common.budgetMode === 'log' && !common.logFacts ? <p className="hint">{t(lang, 'serving.logNeeded')}</p> : null}
      </fieldset>

      <div className="grid">
        <NumberField label={t(lang, 'summary.len')} value={common.len} min={1} step={1024} onChange={(n) => onCommon({ len: Math.max(1, Math.floor(n)) })} />
        <NumberField label={t(lang, 'summary.seqs')} value={common.seqs} min={1} onChange={(n) => onCommon({ seqs: Math.max(1, Math.floor(n)) })} />
      </div>
    </section>
  )
}
