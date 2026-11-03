// Route smoke for the merge gate: load each route of a RUNNING backend (serving the built UI) in
// headless system Chrome and fail unless, for every route,
//   - the final URL path equals the route (unknown or flag-off routes silently redirect to /),
//   - there are no console errors and no uncaught page errors,
//   - every /api/ response is JSON (a route the SPA catch-all swallowed answers text/html) and not 5xx.
// usage: node scripts/smoke.mjs --base http://127.0.0.1:8790 --routes /,/evidence,/trace
//        [--themes ops,projector] [--wait 2500]
// Uses playwright-core (already a devDependency) with the system Chrome: nothing is downloaded.
import { chromium } from 'playwright-core'

const args = process.argv.slice(2)
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d }
const base = opt('base')
const routes = (opt('routes', '') || '').split(',').map((s) => s.trim()).filter(Boolean)
const themes = (opt('themes', '') || '').split(',').filter(Boolean)
const wait = Number(opt('wait', 2500))
if (!base || !routes.length) {
  console.error('usage: node scripts/smoke.mjs --base http://127.0.0.1:8790 --routes /,/evidence [--themes ops,projector]')
  process.exit(2)
}

const CHROME = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const browser = await chromium.launch({ executablePath: CHROME, headless: true })
let failed = 0

for (const theme of themes.length ? themes : [null]) {
  for (const route of routes) {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
    const problems = []
    const pending = []
    let apiCount = 0
    page.on('console', (m) => { if (m.type() === 'error') problems.push(`console error: ${m.text()}`) })
    page.on('pageerror', (e) => problems.push(`page error: ${e}`))
    page.on('response', (res) => {
      const u = new URL(res.url())
      if (!u.pathname.startsWith('/api/')) return
      apiCount++
      pending.push((async () => {
        const where = `${res.request().method()} ${u.pathname}${u.search}`
        if (res.status() >= 500) problems.push(`${where}: HTTP ${res.status()}`)
        const type = res.headers()['content-type'] ?? ''
        if (!type.includes('application/json')) { problems.push(`${where}: content-type ${JSON.stringify(type)}, not JSON`); return }
        try { JSON.parse(await res.text()) } catch (e) { problems.push(`${where}: body is not JSON (${e})`) }
      })().catch((e) => problems.push(`response check failed: ${e}`)))
    })
    await page.addInitScript((theme) => {
      const ui = JSON.parse(localStorage.getItem('spectrum.ui') ?? '{}')
      ui.directorOpen = false
      if (theme) ui.theme = theme
      localStorage.setItem('spectrum.ui', JSON.stringify(ui))
    }, theme)
    try {
      await page.goto(base + route, { waitUntil: 'load', timeout: 20000 })
      await page.waitForTimeout(wait)
    } catch (e) {
      problems.push(`navigation failed: ${e.message}`)
    }
    await Promise.all(pending)
    const finalPath = new URL(page.url()).pathname
    if (finalPath !== route) problems.push(`final URL path ${finalPath}, expected ${route}`)
    const label = `${route}${theme ? ` [${theme}]` : ''}`
    if (problems.length) {
      failed++
      console.log(`FAIL ${label}\n  ` + problems.join('\n  '))
    } else {
      console.log(`ok   ${label}  (${apiCount} /api/ responses, all JSON)`)
    }
    await page.close()
  }
}
await browser.close()
const total = routes.length * Math.max(themes.length, 1)
console.log(`smoke: ${total - failed}/${total} passed`)
process.exit(failed ? 1 : 0)
