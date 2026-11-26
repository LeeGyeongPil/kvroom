// Every failure the core can report. The UI maps `code` to a translated message
// (key `error.<CODE>`), so the code list is part of the public contract.
export type KvroomErrorCode =
  | 'CONFIG_INVALID'
  | 'TP_NOT_DIVISIBLE'
  | 'KV_TOO_SMALL'
  | 'SGLANG_SWA_POOL_TOO_SMALL'
  | 'UNKNOWN_DTYPE'
  | 'INPUT_OUT_OF_RANGE'
  | 'HUB_GATED'
  | 'HUB_NOT_FOUND'
  | 'HUB_NETWORK'
  | 'HUB_BAD_RESPONSE'

export const ERROR_CODES: readonly KvroomErrorCode[] = [
  'CONFIG_INVALID',
  'TP_NOT_DIVISIBLE',
  'KV_TOO_SMALL',
  'SGLANG_SWA_POOL_TOO_SMALL',
  'UNKNOWN_DTYPE',
  'INPUT_OUT_OF_RANGE',
  'HUB_GATED',
  'HUB_NOT_FOUND',
  'HUB_NETWORK',
  'HUB_BAD_RESPONSE',
]

export class KvroomError extends Error {
  readonly code: KvroomErrorCode
  /** Numbers and names the UI needs to fill the translated message. */
  readonly details: Record<string, string | number>

  constructor(code: KvroomErrorCode, message: string, details: Record<string, string | number> = {}) {
    super(message)
    this.name = 'KvroomError'
    this.code = code
    this.details = details
  }
}
