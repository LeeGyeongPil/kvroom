import { useEffect, useMemo, useState } from 'react'
import { KV_DTYPES } from '../core/dtype.ts'
import { KvroomError } from '../core/errors.ts'
import type { LogFacts } from '../core/logparse.ts'
import { parseConfig } from '../core/parse.ts'
import type { ParsedModel } from '../core/parse.ts'
import { SGLANG_RULES_VERSION } from '../core/sglang.ts'
import type { Engine } from '../core/table.ts'
import { VLLM_RULES_VERSION } from '../core/vllm.ts'
import { GPUS } from '../data/gpus.ts'
import { findPreset } from '../data/presets.ts'
import { loadFromHub } from '../hub.ts'
import { errorText, loadStoredLang, resolveLang, saveLang, t } from '../i18n/index.ts'
import type { Lang } from '../i18n/index.ts'
import { CapacityTable, CompareTable } from './CapacityTable.tsx'
import { FlagsPanel } from './FlagsPanel.tsx'
import { LangMenu } from './LangMenu.tsx'
import { LogPanel } from './LogPanel.tsx'
import { ModelPanel } from './ModelPanel.tsx'
import type { ModelSource } from './ModelPanel.tsx'
import { ServingPanel } from './ServingPanel.tsx'
import { SummaryPanel } from './SummaryPanel.tsx'
import { buildCommands, commonForModel, compute, pasteTarget } from './compute.ts'
import type { Common, Setup } from './compute.ts'

const OFFLINE = import.meta.env.VITE_KVROOM_OFFLINE === '1'
const DEFAULT_PRESET = 'gpt-oss-20b'
const MAX_SETUPS = 3

interface Loaded {
  repo: string
  model: ParsedModel
  source: ModelSource
  weightBytes: number
  /** True when the Hub could not be reached and a bundled snapshot stands in. */
  fallback: boolean
}

function numParam(q: URLSearchParams, key: string): number | undefined {
  const raw = q.get(key)
  if (raw === null || raw.trim() === '') return undefined
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : undefined
}

/** Settings for the screenshots and for sharing: `?preset=&lang=&engine=&gpu=&tp=&dtype=&batched=&len=&seqs=`. */
function initialState(q: URLSearchParams): { common: Common; setup: Setup } {
  const engine: Engine = q.get('engine') === 'sglang' ? 'sglang' : 'vllm'
  const gpuParam = q.get('gpu')
  const gpu = gpuParam && GPUS.some((g) => g.name === gpuParam) ? gpuParam : 'L4'
  const dtypeParam = q.get('dtype')
  const dtype = dtypeParam && (KV_DTYPES as readonly string[]).includes(dtypeParam) ? dtypeParam : 'auto'
  const fixed = numParam(q, 'kvgib')
  return {
    common: {
      engine,
      maxNumBatchedTokens: numParam(q, 'batched') ?? 8192,
      blockSize: 16,
      pageSize: 1,
      swaRatio: 0.8,
      utilization: 0.9,
      reserveGiB: 2,
      budgetMode: fixed !== undefined ? 'fixed' : 'estimate',
      fixedGiB: fixed ?? 4,
      len: numParam(q, 'len') ?? 32768,
      seqs: numParam(q, 'seqs') ?? 2,
      weightBytes: 0,
      logFacts: null,
      tokenDefinition: 'auto',
    },
    setup: { gpu, customGB: 24, tp: Math.floor(numParam(q, 'tp') ?? 1), kvDtype: dtype },
  }
}

function fromPreset(key: string): Loaded | null {
  const p = findPreset(key)
  if (!p) return null
  return { repo: p.repo, model: parseConfig(p.config), source: 'preset', weightBytes: p.weightBytes, fallback: false }
}

function setupLabel(s: Setup): string {
  const gpu = s.gpu === 'custom' ? `${s.customGB} GB` : s.gpu
  return `${gpu} x${s.tp} ${s.kvDtype}`
}

export function App() {
  const query = useMemo(() => new URLSearchParams(window.location.search), [])
  const init = useMemo(() => initialState(query), [query])

  const [lang, setLangState] = useState<Lang>(() =>
    resolveLang(query.get('lang'), loadStoredLang(window.localStorage), navigator.languages ?? [navigator.language]),
  )
  const [loaded, setLoaded] = useState<Loaded | null>(() => fromPreset(query.get('preset') ?? DEFAULT_PRESET) ?? fromPreset(DEFAULT_PRESET))
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [weightGB, setWeightGB] = useState(() => (loaded ? loaded.weightBytes / 1e9 : 0))
  const [common, setCommonState] = useState<Common>(init.common)
  const [setups, setSetups] = useState<Setup[]>([init.setup])
  const [active, setActive] = useState(0)
  const [logCleared, setLogCleared] = useState(false)
  /** Bumped to empty the log box when a log is dropped on a model change. */
  const [logEpoch, setLogEpoch] = useState(0)

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const setLang = (l: Lang) => {
    setLangState(l)
    saveLang(window.localStorage, l)
  }
  const setCommon = (patch: Partial<Common>) => setCommonState((c) => ({ ...c, ...patch }))
  const setSetup = (patch: Partial<Setup>) => setSetups((list) => list.map((s, i) => (i === active ? { ...s, ...patch } : s)))

  const adopt = (next: Loaded) => {
    // 구조가 다른 모델이면 이전 모델의 로그 보정값도 남기지 않는다.
    const { patch, logCleared: cleared } = commonForModel(common, loaded?.model ?? null, next.model)
    setLoaded(next)
    setWeightGB(next.weightBytes / 1e9)
    setError(null)
    setCommon(patch)
    if (cleared) {
      setLogCleared(true)
      setLogEpoch((n) => n + 1)
    }
  }

  const onLoad = (repo: string) => {
    setLoading(true)
    setError(null)
    loadFromHub(repo, { offline: OFFLINE })
      .then((m) => {
        const model = parseConfig(m.config)
        adopt({ repo: m.repo, model, source: m.source, weightBytes: m.weightBytes ?? 0, fallback: m.source === 'preset' && !OFFLINE })
      })
      .catch((e: unknown) => setError(e))
      .finally(() => setLoading(false))
  }

  const onPreset = (key: string) => {
    const next = fromPreset(key)
    if (next) adopt(next)
  }

  /** Returns the model id the flags will use, or null when the paste failed. */
  const onPaste = (text: string, repoText: string, repoEdited: boolean): string | null => {
    try {
      const model = parseConfig(JSON.parse(text))
      // 같은 모델인지는 ID 칸이 아니라 config 구조로 판단한다(ID 칸은 이전 모델 ID 로 채워져 있다).
      const target = pasteTarget({
        repoText,
        repoEdited,
        pasted: model,
        previous: loaded ? { repo: loaded.repo, weightBytes: weightGB * 1e9, model: loaded.model } : null,
      })
      adopt({ repo: target.repo, model, source: 'paste', weightBytes: target.weightBytes, fallback: false })
      return target.repo
    } catch (e) {
      setError(e instanceof SyntaxError ? new KvroomError('CONFIG_INVALID', 'Not valid JSON.', { field: '(JSON)' }) : e)
      return null
    }
  }

  const fullCommon: Common = { ...common, weightBytes: weightGB * 1e9 }
  const model = loaded?.model ?? null
  const setup = setups[active] ?? init.setup
  const comp = model ? compute(model, fullCommon, setup) : null
  const rows = model ? setups.map((s) => ({ label: setupLabel(s), comp: compute(model, fullCommon, s) })) : []
  const commands = model && comp && comp.error?.code !== 'INPUT_OUT_OF_RANGE' && loaded ? buildCommands(loaded.repo, model, fullCommon, setup, comp) : null

  return (
    <div className="app">
      <header>
        <div>
          <h1>{t(lang, 'app.title')}</h1>
          <p>{t(lang, 'app.tagline')}</p>
        </div>
        <LangMenu lang={lang} onChange={setLang} />
      </header>

      {loaded && loaded.source === 'preset' ? <p className="banner">{t(lang, 'app.mock')}{loaded.fallback ? ` ${t(lang, 'status.reload')}` : ''}</p> : null}

      <main>
        <div className="col">
          <ModelPanel
            lang={lang}
            repo={loaded?.repo ?? ''}
            model={model}
            source={loaded?.source ?? null}
            loading={loading}
            error={error ? errorText(lang, error) : null}
            weightGB={weightGB}
            onLoad={onLoad}
            onPreset={onPreset}
            onPaste={onPaste}
            onWeightGB={setWeightGB}
          />
          {model ? <ServingPanel lang={lang} common={common} setup={setup} onCommon={setCommon} onSetup={setSetup} /> : null}
          <LogPanel
            key={logEpoch}
            lang={lang}
            facts={common.logFacts}
            logBudget={comp?.logBudget ?? null}
            tokenDefinition={common.tokenDefinition}
            cleared={logCleared}
            onFacts={(f: LogFacts | null) => {
              setLogCleared(false)
              setCommon({ logFacts: f, ...(f ? { budgetMode: 'log' as const, engine: 'vllm' as const } : {}) })
            }}
            onTokenDefinition={(d) => setCommon({ tokenDefinition: d })}
          />
        </div>
        <div className="col">
          {model && comp ? (
            <>
              <SummaryPanel lang={lang} comp={comp} common={fullCommon} />
              <CapacityTable lang={lang} comp={comp} common={fullCommon} />
              <CompareTable
                lang={lang}
                rows={rows}
                active={active}
                canAdd={setups.length < MAX_SETUPS}
                onSelect={setActive}
                onRemove={(i) => {
                  setSetups((list) => list.filter((_, k) => k !== i))
                  setActive((a) => Math.max(0, a >= i ? a - 1 : a))
                }}
                onAdd={() => {
                  setSetups((list) => [...list, { ...setup }])
                  setActive(setups.length)
                }}
              />
              {commands ? <FlagsPanel lang={lang} engine={common.engine} commands={commands} /> : null}
            </>
          ) : (
            <p className="hint">{t(lang, 'status.none')}</p>
          )}
        </div>
      </main>

      <footer>
        <p>{t(lang, 'app.rules', { vllm: VLLM_RULES_VERSION, sglang: SGLANG_RULES_VERSION, date: '2026-11-27' })}</p>
        <p>{t(lang, 'app.footer')}</p>
      </footer>
    </div>
  )
}
