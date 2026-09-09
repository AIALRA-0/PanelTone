import { expect, test } from '@playwright/test'

const fixedViewports = [
  [1526, 986], [1440, 900], [1280, 900], [1150, 986], [1024, 768],
  [760, 900], [759, 900], [390, 844], [320, 720],
] as const

async function openWorkspace(page: Parameters<typeof test>[0]['page'], width: number, height: number) {
  await page.setViewportSize({ width, height })
  await page.goto('/')
  await page.waitForSelector('.app-shell')
  await expect(page.locator('.canvas-page-jump')).toBeAttached()
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
    } else if (width < 1100) {
      await page.getByRole('button', { name: '展开书库', exact: true }).first().click()
      await expect(page.locator('.new-folder')).toBeVisible()
      await page.getByRole('button', { name: '关闭侧栏', exact: true }).click({ position: { x: 500, y: 100 } })
    } else {
      const expandLibrary = page.getByRole('button', { name: '展开书库', exact: true }).first()
      if (await expandLibrary.isVisible()) await expandLibrary.click()
      await expect(page.locator('.new-folder')).toBeVisible()
    }

    const buttons = page.locator('.canvas-page-nav')
    await page.getByRole('button', { name: '对比', exact: true }).click()
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
    const compareStage = await page.locator('.compare-stage').boundingBox()
    const previous = await buttons.nth(0).boundingBox()
    const next = await buttons.nth(1).boundingBox()
    if (compareStage && previous && next) {
      for (const nav of [previous, next]) {
        const separated = nav.x + nav.width <= compareStage.x + 1 || nav.x >= compareStage.x + compareStage.width - 1
          || nav.y >= compareStage.y + compareStage.height - 1 || nav.y + nav.height <= compareStage.y + 1
        expect(separated, 'comparison drag area must not overlap navigation').toBe(true)
      }
    }
  }
})

test('203-page status view stays windowed and keyboard addressable', async ({ page }) => {
  await openWorkspace(page, 1440, 900)
  await page.getByRole('button', { name: /203\/203/ }).click()
  await expect(page.getByRole('spinbutton', { name: '跳转页码' })).toHaveAttribute('max', '203')
  await page.getByRole('button', { name: '查看逐页状态' }).click()
  await expect(page.locator('.status-list')).toBeVisible()
  await expect(page.locator('.status-list button')).toHaveCount(24)
  await page.locator('.status-list').focus()
  await page.keyboard.press('End')
  await expect(page.getByRole('spinbutton', { name: '跳转页码' })).toHaveValue('203')
  await expect(page.locator('.canvas-page-total')).toHaveText('/ 203')
})

test('page jump goes directly to a valid page and rejects an invalid page', async ({ page }) => {
  await openWorkspace(page, 1440, 900)
  await page.getByRole('button', { name: /203\/203/ }).click()
  const input = page.getByRole('spinbutton', { name: '跳转页码' })
  await expect(input).toHaveAttribute('max', '203')
  await input.fill('109')
  await page.getByRole('button', { name: '跳转', exact: true }).click()
  await expect(input).toHaveValue('109')
  await input.fill('204')
  await page.getByRole('button', { name: '跳转', exact: true }).click()
  await expect(input).toHaveValue('109')
  await expect(page.getByText('请输入 1 到 203 之间的页码')).toBeVisible()
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
      const expand = page.getByRole('button', { name: '展开书库', exact: true }).first()
      if (await expand.isVisible()) await expand.click()
      await expect(page.locator('.new-folder')).toBeVisible()
    }
    await expect(page.locator('.page-status-overview')).toBeVisible()
    await expect(page.locator('.page-status-overview .status-list button')).toHaveCount(0)
  }
})
