import assert from 'node:assert/strict'
import { test } from 'node:test'
import { KvroomError } from '../src/core/errors.ts'
import { parseConfig } from '../src/core/parse.ts'
import { PRESETS } from '../src/data/presets.ts'
import { loadFromHub } from '../src/hub.ts'
import { loadConfig } from './helpers.ts'

type Handler = (url: string) => Response | Promise<Response>

function stub(handler: Handler): { f: typeof fetch; calls: string[] } {
  const calls: string[] = []
  const f = (async (input: string | URL | Request) => {
    const url = String(input)
    calls.push(url)
    return handler(url)
  }) as typeof fetch
  return { f, calls }
}

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

test('200: config is returned and parses; safetensors sizes are summed for the weights', async () => {
  const s = stub((url) => {
    if (url.endsWith('/config.json')) return json(loadConfig('gpt-oss-20b'))
    return json({
      siblings: [
        { rfilename: 'model-00000-of-00002.safetensors', size: 4_792_272_488 },
        { rfilename: 'model-00001-of-00002.safetensors', size: 4_798_702_184 },
        { rfilename: 'model-00002-of-00002.safetensors', size: 4_170_000_000 },
        { rfilename: 'config.json', size: 1806 },
        { rfilename: 'original/model.bin', size: 99 },
      ],
    })
  })
  const m = await loadFromHub('openai/gpt-oss-20b', { fetchImpl: s.f })
  assert.equal(m.source, 'hub')
  assert.equal(m.weightBytes, 13_760_974_672)
  assert.equal(parseConfig(m.config).layers.length, 24)
  assert.deepEqual(s.calls, [
    'https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json',
    'https://huggingface.co/api/models/openai/gpt-oss-20b?blobs=true',
  ])
})

test('401 raises HUB_GATED, 404 raises HUB_NOT_FOUND', async () => {
  const gated = stub(() => new Response('', { status: 401 }))
  await assert.rejects(
    loadFromHub('meta-llama/Llama-3.1-8B-Instruct-gated', { fetchImpl: gated.f }),
    (e: unknown) => e instanceof KvroomError && e.code === 'HUB_GATED',
  )
  const missing = stub(() => new Response('', { status: 404 }))
  await assert.rejects(
    loadFromHub('nobody/nothing', { fetchImpl: missing.f }),
    (e: unknown) => e instanceof KvroomError && e.code === 'HUB_NOT_FOUND',
  )
})

test('network error falls back to the bundled snapshot, or fails clearly when there is none', async () => {
  const down = stub(() => {
    throw new TypeError('network down')
  })
  const m = await loadFromHub('openai/gpt-oss-20b', { fetchImpl: down.f })
  assert.equal(m.source, 'preset')
  assert.equal(m.weightBytes, 13_760_000_000)
  await assert.rejects(
    loadFromHub('nobody/unknown-model', { fetchImpl: down.f }),
    (e: unknown) => e instanceof KvroomError && e.code === 'HUB_NETWORK',
  )
})

test('offline mode never calls fetch', async () => {
  const s = stub(() => json({}))
  const m = await loadFromHub('Qwen/Qwen3-8B', { fetchImpl: s.f, offline: true })
  assert.equal(m.source, 'preset')
  await assert.rejects(loadFromHub('nobody/unknown-model', { fetchImpl: s.f, offline: true }), KvroomError)
  assert.equal(s.calls.length, 0)
})

test('a slow Hub is aborted after the timeout instead of hanging', async () => {
  const slow = (async (_url: string | URL | Request, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
    })) as typeof fetch
  const m = await loadFromHub('openai/gpt-oss-20b', { fetchImpl: slow, timeoutMs: 20 })
  assert.equal(m.source, 'preset')
})

test('malformed ids and non-JSON answers are reported', async () => {
  await assert.rejects(loadFromHub('not a repo'), (e: unknown) => e instanceof KvroomError && e.code === 'HUB_NOT_FOUND')
  const bad = stub(() => new Response('<html>', { status: 200 }))
  await assert.rejects(
    loadFromHub('a/b', { fetchImpl: bad.f }),
    (e: unknown) => e instanceof KvroomError && e.code === 'HUB_BAD_RESPONSE',
  )
})

test('every bundled preset parses and has a source URL', () => {
  assert.equal(PRESETS.length, 8)
  for (const p of PRESETS) {
    const m = parseConfig(p.config)
    assert.ok(m.layers.length > 0, p.key)
    assert.match(p.source, /^https:\/\/huggingface\.co\//)
  }
})
