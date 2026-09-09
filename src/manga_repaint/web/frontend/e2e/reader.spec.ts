import { expect, test, type Page } from '@playwright/test'

async function fixture(page: Page) {
  const requests: string[] = []
  const job = { ...(await (await page.request.get('/api/jobs')).json())[0], id: 'reader-fixture', display_name: '阅读测试', page_count: 203 }
  job.progress = { ...job.progress, completed_pages: 203, total_pages: 203, page_states: [] }
  const pages = Array.from({ length: 203 }, (_, index) => ({
    page_index: index, width: 700, height: 1000, status: 'qa_passed', asset_revision: 'fixture-1',
    source_url: '/forbidden-original.png', final_url: '/forbidden-original.png',
    source_reading_url: `/__reader_fixture__/${index}-source.svg?size=reader`,
    final_reading_url: `/__reader_fixture__/${index}-final.svg?size=reader`,
    source_preview_url: `/__reader_fixture__/${index}-source.svg?size=preview`,
    final_preview_url: `/__reader_fixture__/${index}-final.svg?size=preview`,
    thumbnail_url: null,
  }))
  await page.route(url => url.pathname === '/api/jobs', route => route.fulfill({ json: [job] }))
  await page.route('**/api/events*', route => route.fulfill({ contentType: 'text/event-stream', body: 'retry: 60000\n\n' }))
  await page.route(url => url.pathname === '/api/library/tree', route => route.fulfill({ json: { folders: [], root_jobs: [job] } }))
  await page.route('**/api/jobs/reader-fixture/pages', route => route.fulfill({ json: pages }))
  const attempts = new Map<string, number>()
  await page.route('**/__reader_fixture__/**', async route => {
    const url = route.request().url()
    const name = new URL(url).pathname.split('/').pop()!
    if (url.includes('size=reader')) {
      requests.push(name)
      attempts.set(name, (attempts.get(name) || 0) + 1)
      await new Promise(resolve => setTimeout(resolve, 250))
      if (name === '8-final.svg' && attempts.get(name) === 1) return route.fulfill({ status: 503, body: 'Offline' })
      if (name === '29-final.svg' && attempts.get(name) === 1) return route.fulfill({ status: 202, json: { status: 'preparing' } })
    }
    return route.fulfill({ contentType: 'image/svg+xml', body: `<svg xmlns="http://www.w3.org/2000/svg" width="700" height="1000"><rect width="700" height="1000" fill="#efd1b5"/><text x="40" y="100" font-size="40">${name}</text></svg>`, headers: { 'Cache-Control': 'private, max-age=3600, immutable' } })
  })
  await page.goto('/')
  await expect(page.locator('.single-page-stage img')).toHaveAttribute('data-decoded', 'true')
  return requests
}

async function jump(page: Page, number: number) {
  await page.getByRole('spinbutton', { name: '跳转页码' }).fill(String(number))
  await page.getByRole('button', { name: '跳转', exact: true }).click()
}

test('light reader displays the right placeholder, reuses decoded neighbours and handles failures', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  const requests = await fixture(page)
  const image = page.locator('.single-page-stage img')
  await expect.poll(() => requests.includes('1-final.svg')).toBe(true)
  await page.waitForTimeout(350)
  await page.getByRole('button', { name: '下一页', exact: true }).click()
  await expect(image).toHaveAttribute('data-decoded', 'true', { timeout: 300 })
  expect(requests.filter(url => url === '1-final.svg')).toHaveLength(1)
  expect(requests.some(url => url.includes('source'))).toBe(false)
  await jump(page, 15)
  await expect(image).toHaveAttribute('data-reading-url', /14-final/, { timeout: 100 })
  await expect(image).toHaveAttribute('src', /14-final.*preview/)
  await jump(page, 20)
  await expect(image).toHaveAttribute('data-reading-url', /19-final/)
  await expect(image).toHaveAttribute('data-decoded', 'true')
  await page.waitForTimeout(400)
  await expect(image).toHaveAttribute('data-reading-url', /19-final/)
  await jump(page, 9)
  await expect(page.locator('.reader-image-status.error')).toContainText('离线')
  await page.getByRole('button', { name: '重试', exact: true }).click()
  await expect(image).toHaveAttribute('data-decoded', 'true')
  await jump(page, 30)
  await expect(image).toHaveAttribute('data-decoded', 'true')
  expect(requests.filter(url => url === '29-final.svg')).toHaveLength(2)
})

test('vertical reader is windowed, scrolls naturally, jumps and retains position between modes', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await fixture(page)
  await jump(page, 9)
  await page.getByRole('button', { name: '垂直连读', exact: true }).click()
  const reader = page.getByRole('region', { name: '垂直连续阅读' })
  await expect(reader).toBeVisible()
  expect(await page.locator('.continuous-page').count()).toBeLessThanOrEqual(3)
  await expect(page.locator('.single-page-stage')).toHaveCount(0)
  await reader.evaluate(element => { element.scrollTop += element.clientWidth * 1.5 })
  await expect(page.getByRole('spinbutton', { name: '跳转页码' })).not.toHaveValue('9')
  await jump(page, 199)
  await expect(page.locator('.continuous-page[data-page-index="198"]')).toBeVisible()
  await page.getByRole('button', { name: '单页翻阅', exact: true }).click()
  await expect(page.locator('.single-page-stage img')).toHaveAttribute('data-reading-url', /198-final/)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: '垂直连读', exact: true }).click()
  await expect(reader).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  expect(await page.locator('.continuous-page').count()).toBeLessThanOrEqual(3)
})

test('touch swipe turns a page, while a zoomed swipe pans instead', async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true })
  const page = await context.newPage()
  try {
    await page.goto('http://127.0.0.1:8765')
    // Reuse the same isolated routes; no real job is modified.
    await fixture(page)
    const cdp = await context.newCDPSession(page)
    const box = (await page.locator('.canvas').boundingBox())!
    const x = box.x + box.width * .8, y = box.y + box.height * .45
    async function swipe() {
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] })
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x - 110, y }] })
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] })
    }
    await swipe()
    await expect(page.getByRole('spinbutton', { name: '跳转页码' })).toHaveValue('2')
    await page.getByRole('button', { name: '200%', exact: true }).click()
    await swipe()
    await expect(page.getByRole('spinbutton', { name: '跳转页码' })).toHaveValue('2')
    await expect(page.locator('.single-page-stage img')).toHaveAttribute('style', /translate3d\(-110px/)
  } finally { await context.close() }
})
