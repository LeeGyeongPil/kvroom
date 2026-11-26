import { GIB } from '../core/vllm.ts'
import { errorText, t } from '../i18n/index.ts'
import type { Lang } from '../i18n/index.ts'
import type { Common, Computed } from './compute.ts'

const fmt = (n: number): string => n.toLocaleString('en-US')

const lenLabel = (n: number): string => (n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : fmt(n))

interface TableProps {
  lang: Lang
  comp: Computed
  common: Common
}

// Length x concurrency grid: green cells are guaranteed without preemption.
export function CapacityTable({ lang, comp, common }: TableProps) {
  if (comp.error) return <p className="error">{errorText(lang, comp.error)}</p>
  if (comp.table.length === 0) return null
  return (
    <section className="panel">
      <h2>{t(lang, 'table.title')}</h2>
      <div className="table-wrap">
        <table className="capacity">
          <thead>
            <tr>
              <th scope="col">
                {t(lang, 'table.length')} \ {t(lang, 'table.seqs')}
              </th>
              {comp.seqsList.map((n) => (
                <th scope="col" key={n}>
                  {n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {comp.table.map((row, i) => (
              <tr key={comp.lens[i]}>
                <th scope="row">{lenLabel(comp.lens[i] ?? 0)}</th>
                {row.map((cell) => (
                  <td key={cell.seqs} className={cell.ok ? 'ok' : 'no'} title={`${fmt(cell.len)} x ${cell.seqs}`}>
                    {cell.ok ? fmt(cell.spare) : t(lang, 'table.no')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">{t(lang, 'table.legend', { unit: t(lang, common.engine === 'vllm' ? 'table.unit.vllm' : 'table.unit.sglang') })}</p>
      <p>{t(lang, 'summary.maxLenFor', { n: common.seqs, len: fmt(comp.maxLenForSeqs) })}</p>
      <p>{t(lang, 'summary.maxSeqsFor', { n: comp.maxSeqs, len: fmt(common.len) })}</p>
      {common.engine === 'vllm' ? <p className="hint">{t(lang, 'summary.tradeoff')}</p> : null}
    </section>
  )
}

interface CompareProps {
  lang: Lang
  rows: { label: string; comp: Computed }[]
  active: number
  canAdd: boolean
  onSelect: (i: number) => void
  onRemove: (i: number) => void
  onAdd: () => void
}

// Up to three setups side by side: same model, different GPU / count / dtype.
export function CompareTable({ lang, rows, active, canAdd, onSelect, onRemove, onAdd }: CompareProps) {
  return (
    <section className="panel">
      <h2>{t(lang, 'compare.title')}</h2>
      <div className="table-wrap">
        <table className="compare">
          <thead>
            <tr>
              <th scope="col">{t(lang, 'compare.setup')}</th>
              <th scope="col">{t(lang, 'compare.kv')}</th>
              <th scope="col">{t(lang, 'compare.concurrency')}</th>
              <th scope="col">{t(lang, 'compare.maxLen')}</th>
              <th scope="col" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={i === active ? 'active' : ''}>
                <th scope="row">{r.label}</th>
                {r.comp.error ? (
                  <td colSpan={3} className="error">
                    {errorText(lang, r.comp.error)}
                  </td>
                ) : (
                  <>
                    <td>{r.comp.budget ? `${(r.comp.budget.kvBytesPerRank / GIB).toFixed(2)} GiB` : '-'}</td>
                    <td>{fmt(r.comp.maxSeqs)}</td>
                    <td>{fmt(r.comp.maxLenForSeqs)}</td>
                  </>
                )}
                <td className="actions">
                  <button type="button" onClick={() => onSelect(i)} disabled={i === active}>
                    {t(lang, 'compare.select')}
                  </button>
                  {rows.length > 1 ? (
                    <button type="button" onClick={() => onRemove(i)}>
                      {t(lang, 'compare.remove')}
                    </button>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {canAdd ? (
        <button type="button" onClick={onAdd}>
          {t(lang, 'compare.add')}
        </button>
      ) : null}
    </section>
  )
}
