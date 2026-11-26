// GPU memory sizes as printed on the vendors' public spec sheets (decimal GB, checked 2026-11-27):
//   NVIDIA product pages for T4, L4, A10, RTX 3090/4090/5090, L40S, A100, H100, H200;
//   AMD Instinct MI300X page. The UI lets users type any other size.

export interface Gpu {
  name: string
  /** Vendor-quoted memory in GB (1 GB = 10^9 bytes). */
  memoryGB: number
}

export const GPUS: readonly Gpu[] = [
  { name: 'T4', memoryGB: 16 },
  { name: 'L4', memoryGB: 24 },
  { name: 'A10G', memoryGB: 24 },
  { name: 'RTX 3090', memoryGB: 24 },
  { name: 'RTX 4090', memoryGB: 24 },
  { name: 'RTX 5090', memoryGB: 32 },
  { name: 'L40S', memoryGB: 48 },
  { name: 'A100 40GB', memoryGB: 40 },
  { name: 'A100 80GB', memoryGB: 80 },
  { name: 'H100 80GB', memoryGB: 80 },
  { name: 'H200', memoryGB: 141 },
  { name: 'MI300X', memoryGB: 192 },
]

/** Convert vendor GB to GiB, the unit engines report. */
export function gbToGiB(gb: number): number {
  return (gb * 1e9) / 1024 ** 3
}
