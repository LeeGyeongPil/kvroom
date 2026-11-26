// Fetch a model's config.json (and weight file sizes) from the Hugging Face Hub in the browser.
// No token is ever sent. Gated models answer 401 and must be pasted by hand.
import { KvroomError } from './core/errors.ts'
import { findPreset } from './data/presets.ts'
import type { Preset } from './data/presets.ts'

export interface HubOptions {
  /** Inject a fetch (tests, custom proxies). Defaults to `globalThis.fetch`. */
  fetchImpl?: typeof fetch
  /** True = never call the network; only the bundled snapshots are used. */
  offline?: boolean
  /** Abort a request after this many milliseconds (default 15000). Slow Hubs are not an error until then. */
  timeoutMs?: number
  /** Hub base URL (default https://huggingface.co). */
  baseUrl?: string
}

export interface HubModel {
  repo: string
  /** The raw config.json content. */
  config: unknown
  /** Sum of `.safetensors` file sizes, when the Hub reported them. */
  weightBytes?: number
  /** `hub` = fetched live, `preset` = bundled snapshot used instead (offline mode or network failure). */
  source: 'hub' | 'preset'
  preset?: Preset
}

export const REPO_RE = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/

function fromPreset(repo: string): HubModel | undefined {
  const p = findPreset(repo)
  if (!p) return undefined
  return { repo: p.repo, config: p.config, weightBytes: p.weightBytes, source: 'preset', preset: p }
}

async function timedFetch(f: typeof fetch, url: string, timeoutMs: number): Promise<Response> {
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort(), timeoutMs)
  try {
    return await f(url, { signal: ctl.signal })
  } finally {
    clearTimeout(timer)
  }
}

function sumSafetensors(body: unknown): number | undefined {
  if (typeof body !== 'object' || body === null) return undefined
  const siblings = (body as { siblings?: unknown }).siblings
  if (!Array.isArray(siblings)) return undefined
  let total = 0
  for (const s of siblings) {
    if (typeof s !== 'object' || s === null) continue
    const { rfilename, size } = s as { rfilename?: unknown; size?: unknown }
    if (typeof rfilename === 'string' && rfilename.endsWith('.safetensors') && typeof size === 'number') total += size
  }
  return total > 0 ? total : undefined
}

/**
 * Load a model from the Hub, falling back to a bundled snapshot when the network fails.
 * @throws KvroomError `HUB_GATED` (401/403: paste config.json), `HUB_NOT_FOUND` (404),
 *   `HUB_NETWORK` (no network and no snapshot), `HUB_BAD_RESPONSE` (not JSON / unexpected status).
 */
export async function loadFromHub(repo: string, options: HubOptions = {}): Promise<HubModel> {
  const id = repo.trim()
  if (!REPO_RE.test(id)) {
    throw new KvroomError('HUB_NOT_FOUND', `"${repo}" is not a Hub model id. Use the form owner/name, for example openai/gpt-oss-20b.`, {
      repo,
    })
  }
  const noNetwork = (): KvroomError =>
    new KvroomError('HUB_NETWORK', `Could not reach the Hugging Face Hub for ${id} and no bundled snapshot exists. Paste its config.json instead.`, {
      repo: id,
    })

  if (options.offline) {
    const snap = fromPreset(id)
    if (snap) return snap
    throw noNetwork()
  }

  const f = options.fetchImpl ?? globalThis.fetch
  const base = options.baseUrl ?? 'https://huggingface.co'
  const timeoutMs = options.timeoutMs ?? 15000

  let res: Response
  try {
    res = await timedFetch(f, `${base}/${id}/resolve/main/config.json`, timeoutMs)
  } catch {
    const snap = fromPreset(id)
    if (snap) return snap
    throw noNetwork()
  }

  if (res.status === 401 || res.status === 403) {
    throw new KvroomError('HUB_GATED', `${id} is gated on the Hub. Accept its terms there, then paste or drop its config.json here.`, {
      repo: id,
    })
  }
  if (res.status === 404) {
    throw new KvroomError('HUB_NOT_FOUND', `${id} was not found on the Hub. Check the spelling.`, { repo: id })
  }
  if (res.status >= 500) {
    const snap = fromPreset(id)
    if (snap) return snap
    throw noNetwork()
  }
  if (!res.ok) {
    throw new KvroomError('HUB_BAD_RESPONSE', `The Hub answered HTTP ${res.status} for ${id}.`, { repo: id, status: res.status })
  }

  let config: unknown
  try {
    config = await res.json()
  } catch {
    throw new KvroomError('HUB_BAD_RESPONSE', `The Hub did not return JSON for ${id}/config.json.`, { repo: id })
  }

  // File sizes are optional: when they cannot be read the UI asks for the weight size by hand.
  let weightBytes: number | undefined
  try {
    const info = await timedFetch(f, `${base}/api/models/${id}?blobs=true`, timeoutMs)
    if (info.ok) weightBytes = sumSafetensors(await info.json())
  } catch {
    weightBytes = undefined
  }
  return { repo: id, config, ...(weightBytes !== undefined ? { weightBytes } : {}), source: 'hub' }
}
