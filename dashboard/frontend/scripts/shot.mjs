// Headless screenshot helper for visual QA (uses the system Google Chrome; no browser download).
// usage: node scripts/shot.mjs <path> <out.png> [--w 1440] [--h 900] [--wait 2500] [--scroll "<css selector>"]
//        [--keys "d,ArrowRight"] [--theme projector] [--full]   (--full scrolls <main> and stitches nothing: it
//        resizes the viewport to the main content height)
import { chromium } from 'playwright-core'

const args = process.argv.slice(2)
const path = args[0] ?? '/'
const out = args[1] ?? 'shot.png'
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d }
const flag = (k) => args.includes(`--${k}`)
const w = Number(opt('w', 1440)), h = Number(opt('h', 900))
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true })
const page = await browser.newPage({ viewport: { width: w, height: h } })
const errors = []
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
page.on('pageerror', (e) => errors.push(String(e)))
await page.addInitScript((theme) => {
  const ui = JSON.parse(localStorage.getItem('spectrum.ui') ?? '{}')
  ui.directorOpen = false
  if (theme) ui.theme = theme
  localStorage.setItem('spectrum.ui', JSON.stringify(ui))
}, opt('theme', null))
await page.goto(`http://localhost:5173${path}`, { waitUntil: 'networkidle' })
await page.waitForTimeout(Number(opt('wait', 2500)))
for (const k of (opt('keys', '') || '').split(',').filter(Boolean)) { await page.keyboard.press(k); await page.waitForTimeout(400) }
const sel = opt('scroll', null)
if (sel) { await page.locator(sel).first().scrollIntoViewIfNeeded(); await page.waitForTimeout(500) }
if (flag('full')) {
  const H = await page.evaluate(() => (document.querySelector('main')?.scrollHeight ?? 900) + 180)
  await page.setViewportSize({ width: w, height: Math.min(H, 6000) })
  await page.waitForTimeout(800)
}
await page.screenshot({ path: out })
if (errors.length) console.log('CONSOLE ERRORS:\n' + errors.slice(0, 10).join('\n'))
console.log('saved', out)
await browser.close()
