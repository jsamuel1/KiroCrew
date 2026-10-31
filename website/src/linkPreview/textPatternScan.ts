/**
 * The matching half of `textPatterns`, shared by the worker (`textPatternWorker.ts`)
 * and the tests. Pure: no DOM, no timers.
 *
 * A pattern is app input that meets chat text, and no static analysis of a regex
 * can prove it cheap, so this code never runs on the dashboard's thread: the
 * worker runs it, and the main thread's time budget (`textPatterns.ts`) is what
 * bounds it. Before each pattern starts, `onStart(index)` is called, so the main
 * thread knows which pattern was running if it has to terminate the worker.
 *
 * @module linkPreview/textPatternScan
 */

/** Longest text-node slice scanned; a longer node is skipped whole. */
export const MAX_SCAN_SUBJECT = 2000
/** Ceiling on hits collected from one text node. */
export const MAX_HITS_PER_NODE = 200

export interface ScanPattern {
  source: string
  flags: string
}

export interface ScanHit {
  start: number
  end: number
  /** Index into the patterns the scan was given. */
  pattern: number
}

/**
 * Non-overlapping hits per text, earlier patterns first. A pattern that matches
 * the empty string is not scanned and is reported through `onReject`: that test
 * runs an app's regex too, so it runs here, under the budget, not on the
 * dashboard's thread.
 *
 * Non-overlapping hits per text, earlier patterns first. A hit counts only when
 * its pattern also matches the hit's text on its own: the hover resolves a span
 * from its text alone, so a match that needs surrounding text (a positive
 * lookbehind) would draw a span that opens nothing.
 */
export function scanTexts(
  patterns: readonly ScanPattern[],
  texts: readonly string[],
  onStart: (index: number) => void = () => {},
  onReject: (index: number) => void = () => {},
): ScanHit[][] {
  const out: ScanHit[][] = texts.map(() => [])
  patterns.forEach((p, index) => {
    onStart(index)
    if (new RegExp(p.source, p.flags.replace(/g/g, '')).test('')) {
      onReject(index)
      return
    }
    const scan = new RegExp(p.source, p.flags.includes('g') ? p.flags : `${p.flags}g`)
    const whole = new RegExp(`^(?:${p.source})$`, p.flags.replace(/g/g, ''))
    texts.forEach((value, t) => {
      if (!value || value.length > MAX_SCAN_SUBJECT) return
      const hits = out[t]
      scan.lastIndex = 0
      let m: RegExpExecArray | null
      while (hits.length < MAX_HITS_PER_NODE && (m = scan.exec(value)) !== null) {
        // A zero-width match cannot advance the scan; abandon the pattern here.
        if (!m[0]) break
        const start = m.index
        const end = start + m[0].length
        if (value[start - 1] === '\\') continue
        if (hits.some((h) => start < h.end && h.start < end)) continue
        if (!whole.test(m[0])) continue
        hits.push({ start, end, pattern: index })
      }
    })
  })
  for (const hits of out) hits.sort((a, b) => a.start - b.start)
  return out
}

/** Messages between the main thread and the worker. */
export type ToWorker = { type: 'scan'; batch: number; patterns: ScanPattern[]; texts: string[] }
export type FromWorker =
  | { type: 'start'; batch: number; pattern: number }
  | { type: 'done'; batch: number; hits: ScanHit[][]; rejected?: number[] }
