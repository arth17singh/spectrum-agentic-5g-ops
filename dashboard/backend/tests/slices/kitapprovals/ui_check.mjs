// Live UI check + screenshots for /approvals against a FIXTURE backend (never AWS).
//   SPECTRUM_OFFLINE=1 KITAPPROVALS_FIXTURE=<demo.json> APPROVALS_STATE_DIR=<tmp with a PIN> uvicorn ... --port 879x
//   node ui_check.mjs --base http://127.0.0.1:879x --mode empty|demo --pin 246810 --out <dir>
// demo mode also checks review F8: with a button in the confirm sheet focused, the shell's hotkeys
// (digits = demo beats, ArrowRight = next beat, p = pause) do nothing; Space activates the button instead.
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const require = createRequire(join(here, '../../../../frontend/package.json'))
const { chromium } = require('playwright-core')

const args = process.argv.slice(2)
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d }
const base = opt('base'), mode = opt('mode', 'demo'), pin = opt('pin', '246810'), out = opt('out', '.')
const CHROME = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const fail = (m) => { console.error(`ui_check FAIL: ${m}`); process.exitCode = 1 }

const browser = await chromium.launch({ executablePath: CHROME, headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errors = []
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
page.on('pageerror', (e) => errors.push(String(e)))
const sims = []
page.on('request', (r) => { const p = new URL(r.url()).pathname; if (r.method() === 'POST' && !p.startsWith('/api/kitapprovals')) sims.push(p) })

await page.goto(`${base}/approvals`)
await page.waitForSelector('[data-slice="kitapprovals"]')
await page.waitForTimeout(1200)
if (await page.locator('[aria-label="Demo Director"]').isVisible()) { await page.keyboard.press('d'); await page.waitForTimeout(300) }

if (mode === 'empty') {
  if (!(await page.locator('[data-empty="pending"]').isVisible())) fail('empty state not shown')
  await page.screenshot({ path: join(out, '1-empty.png'), fullPage: true })
} else {
  await page.click('[data-row="chg-3f9a1c2b7d10"]')
  await page.waitForSelector('[data-detail="chg-3f9a1c2b7d10"]')
  await page.waitForTimeout(300)
  await page.screenshot({ path: join(out, '2-pending-row.png'), fullPage: true })

  await page.click('[data-action="review"]')
  await page.waitForSelector('[role="dialog"]')
  await page.fill('[role="dialog"] textarea', 'checked: coverage gain predicted, trust 0.91')
  await page.fill('[role="dialog"] input[type="password"]', pin)
  await page.waitForTimeout(1700) // the decision buttons arm after 1.5 s
  await page.screenshot({ path: join(out, '3-confirm-sheet.png'), fullPage: true })

  // hotkeys must not fire from inside the sheet
  await page.focus('[role="dialog"] [data-action="approve"]')
  for (const k of ['3', 'ArrowRight', 'p', 'g', 'm']) await page.keyboard.press(k)
  await page.keyboard.down('r'); await page.waitForTimeout(1200); await page.keyboard.up('r')
  await page.waitForTimeout(300)
  if (new URL(page.url()).pathname !== '/approvals') fail(`a hotkey navigated to ${page.url()}`)
  if (!(await page.locator('[role="dialog"]').isVisible())) fail('the sheet closed on a hotkey')
  if (sims.length) fail(`a hotkey reached the simulator: ${sims.join(', ')}`)
  // Space on the focused Approve button activates it (and does not run nextBeat)
  await page.keyboard.press(' ')
  await page.waitForSelector('[role="dialog"]', { state: 'detached', timeout: 5000 }).catch(() => fail('Space did not approve'))
  await page.waitForTimeout(800)
  if (new URL(page.url()).pathname !== '/approvals') fail(`Space navigated to ${page.url()}`)
  await page.screenshot({ path: join(out, '4-decided-approved-not-applied.png'), fullPage: true })

  await page.click('[data-row="chg-c71d09e5f2ab"]')
  await page.waitForSelector('[data-detail="chg-c71d09e5f2ab"]')
  await page.waitForTimeout(300)
  await page.screenshot({ path: join(out, '5-applied.png'), fullPage: true })
}
if (errors.length) fail(`console errors: ${errors.join(' | ')}`)
await browser.close()
if (!process.exitCode) console.log(`ui_check ${mode}: ok`)
