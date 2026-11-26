import { GIB } from '../core/vllm.ts'
import { errorText, t } from '../i18n/index.ts'
import type { Lang } from '../i18n/index.ts'
import type { Common, Computed } from './compute.ts'

const fmt = (n: number): string => n.toLocaleString('en-US')

interface Props {
  lang: Lang
  comp: Computed
  common: Common
}

export function SummaryPanel({ lang, comp, common }: Props) {
  if (comp.error) {
    return (
      <section className="panel">
        <h2>{t(lang, 'summary.title')}</h2>
        <p className="error">{errorText(lang, comp.error)}</p>
      </section>
    )
  }
  const { budget, vllm, sglang, placement } = comp
  if (!budget || !placement) return null
  const v = vllm
  const concurrency = v ? v.maxConcurrency : common.len > 0 && comp.maxSeqs >= 0 ? comp.maxSeqs : 0
  const dense = comp.denseConcurrency
  const diff = dense !== null && v && v.maxConcurrency > 0 ? Math.round((dense / v.maxConcurrency - 1) * 100) : null
  return (
    <section className="panel">
      <h2>{t(lang, 'summary.title')}</h2>
      {comp.tooSmallMaxLen !== null ? <p className="error">{t(lang, 'status.tooSmall', { len: fmt(comp.tooSmallMaxLen) })}</p> : null}
      <dl>
        <dt>{t(lang, 'summary.kv')}</dt>
        <dd>
          {(budget.kvBytesPerRank / GIB).toFixed(2)} GiB
          {budget.source === 'estimate' ? <span className="badge conf-medium"> ~</span> : null}
        </dd>
        {budget.source === 'estimate' ? (
          <>
            <dt>{t(lang, 'summary.weights')}</dt>
            <dd>{(budget.weightBytesPerRank / GIB).toFixed(2)} GiB</dd>
          </>
        ) : null}
        <dt>{t(lang, 'summary.replication')}</dt>
        <dd>{t(lang, 'summary.replicationValue', { value: placement.replication, heads: placement.headsPerRank })}</dd>
        {v ? (
          <>
            <dt>{t(lang, 'summary.blocks')}</dt>
            <dd>{fmt(v.numBlocks)}</dd>
            <dt>{t(lang, 'summary.groups')}</dt>
            <dd>{t(lang, 'summary.groupsValue', { groups: v.numGroups, size: v.groupSize, padding: v.paddingSlots })}</dd>
            <dt>{t(lang, 'summary.perRequest')}</dt>
            <dd>{fmt(v.blocksPerRequest)}</dd>
            <dt>{t(lang, 'summary.concurrency')}</dt>
            <dd>{concurrency.toFixed(2)}x</dd>
            <dt>{t(lang, 'summary.logTokens')}</dt>
            <dd>{t(lang, 'summary.tokensCurrent', { now: fmt(v.kvTokensCurrent), old: fmt(v.kvTokensV012) })}</dd>
            {dense !== null && diff !== null && diff !== 0 ? (
              <>
                <dt>{t(lang, 'summary.dense')}</dt>
                <dd>{t(lang, 'summary.denseValue', { value: dense.toFixed(2), diff: diff > 0 ? `+${diff}` : diff })}</dd>
              </>
            ) : null}
          </>
        ) : null}
        {sglang ? (
          <>
            <dt>{t(lang, 'summary.pools')}</dt>
            <dd>
              {sglang.hybrid
                ? t(lang, 'summary.poolsValue', { full: fmt(sglang.fullTokens), swa: fmt(sglang.swaTokens) })
                : t(lang, 'summary.poolsDense', { full: fmt(sglang.fullTokens) })}
            </dd>
          </>
        ) : null}
      </dl>
      {v?.pageMismatch ? <p className="error">{t(lang, 'summary.pageMismatch')}</p> : null}
    </section>
  )
}
