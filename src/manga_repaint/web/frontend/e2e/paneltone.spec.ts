import { expect, test } from '@playwright/test'

const fixedViewports = [
  [1526, 986], [1440, 900], [1280, 900], [1150, 986], [1024, 768],
  [760, 900], [759, 900], [390, 844], [320, 720],
] as const

async function openWorkspace(page: Parameters<typeof test>[0]['page'], width: number, height: number) {
  await page.setViewportSize({ width, height })
  await page.goto('/')
  await page.waitForSelector('.app-shell')
  await page.waitForTimeout(700)
}

test('fixed viewports keep the workspace usable and geometry stable', async ({ page }) => {
  for (const [width, height] of fixedViewports) {
    await openWorkspace(page, width, height)
    const layout = await page.evaluate(() => ({
      overflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth,
      overview: Boolean(document.querySelector('.page-status-overview')),
      stateRows: document.querySelectorAll('.page-status-overview .status-list button').length,
    }))
    expect(layout.overflow, `${width}x${height} horizontal overflow`).toBeLessThanOrEqual(1)
    expect(layout.overview, `${width}x${height} status overview`).toBe(true)
    expect(layout.stateRows, `${width}x${height} status rows`).toBeLessThanOrEqual(24)

    const booksTab = page.getByRole('button', { name: /书籍/ })
    if (width < 760 && await booksTab.count()) {
      await booksTab.click()
      await expect(page.locator('.new-folder')).toBeVisible()
      await page.getByRole('button', { name: /预览/ }).click()
    } else {
      await expect(page.locator('.new-folder')).toBeVisible()
    }

    const buttons = page.locator('.canvas-page-nav')
    const svgs = page.locator('.canvas-page-nav .icon-chevron')
    expect(await buttons.count()).toBe(2)
    for (let index = 0; index < 2; index += 1) {
      const button = await buttons.nth(index).boundingBox()
      const svg = await svgs.nth(index).boundingBox()
      expect(button).not.toBeNull()
      expect(svg).not.toBeNull()
      expect(Math.abs((button!.x + button!.width / 2) - (svg!.x + svg!.width / 2))).toBeLessThanOrEqual(1)
      expect(Math.abs((button!.y + button!.height / 2) - (svg!.y + svg!.height / 2))).toBeLessThanOrEqual(1)
    }
  }
})

test('203-page status view stays windowed and keyboard addressable', async ({ page }) => {
  await openWorkspace(page, 1440, 900)
  await page.getByRole('button', { name: /203\/203/ }).click()
  await page.getByRole('button', { name: '查看逐页状态' }).click()
  await expect(page.locator('.status-list')).toBeVisible()
  await expect(page.locator('.status-list button')).toHaveCount(24)
  await page.locator('.status-list').focus()
  await page.keyboard.press('End')
  await expect(page.locator('.canvas-page-count')).toHaveText('203 / 203')
})

test('width sweep does not create horizontal overflow or hide mobile library actions', async ({ page }) => {
  test.setTimeout(120_000)
  for (let width = 320; width <= 1920; width += 40) {
    await openWorkspace(page, width, 900)
    const overflow = await page.evaluate(() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth)
    expect(overflow, `${width}px horizontal overflow`).toBeLessThanOrEqual(1)
    if (width < 760) {
      await page.getByRole('button', { name: /书籍/ }).click()
      await expect(page.locator('.new-folder')).toBeVisible()
      await page.getByRole('button', { name: /预览/ }).click()
      await page.getByRole('button', { name: /进度/ }).click()
    } else {
      await expect(page.locator('.new-folder')).toBeVisible()
    }
    await expect(page.locator('.page-status-overview')).toBeVisible()
    await expect(page.locator('.page-status-overview .status-list button')).toHaveCount(0)
  }
})
