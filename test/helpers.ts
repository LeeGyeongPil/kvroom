import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseConfig } from '../src/core/parse.ts'
import type { ParsedModel } from '../src/core/parse.ts'

const dir = import.meta.dirname

export const GiB = 1024 ** 3

export function loadConfig(name: string): unknown {
  return JSON.parse(readFileSync(join(dir, 'fixtures', 'configs', `${name}.json`), 'utf8'))
}

export function loadModel(name: string): ParsedModel {
  return parseConfig(loadConfig(name))
}

export function loadLog(name: string): string {
  return readFileSync(join(dir, 'fixtures', 'logs', `${name}.txt`), 'utf8')
}
