import { useEffect, useMemo, useRef, useState } from 'react'
import { ReaderImage } from './readerAssets'

export type ReadingPage = {
  page_index: number; width?: number; height?: number; source_url: string
  source_reading_url?: string | null; final_reading_url?: string | null
  source_preview_url?: string | null; final_preview_url?: string | null
  source_display_url?: string | null; final_display_url?: string | null
  final_url: string | null; preview_url?: string | null; thumbnail_url: string | null
}
export type PreviewMode = 'compare' | 'source' | 'final' | 'mask'
export function readingUrl(page: ReadingPage, variant: 'source' | 'final', detail = false) {
  if (variant === 'source') return (detail ? page.source_display_url : page.source_reading_url) || page.source_display_url || page.source_url
  return (detail ? page.final_display_url : page.final_reading_url) || page.final_display_url || page.preview_url || page.final_url
}
export function placeholderUrl(page: ReadingPage, variant: 'source' | 'final') {
  return variant === 'source' ? page.source_preview_url : page.final_preview_url || page.thumbnail_url
}

export function ContinuousReader({ pages, pageIndex, jumpToken, mode, jobId, detail, onPageChange }: {
  pages: ReadingPage[]; pageIndex: number; jumpToken: number; mode: PreviewMode
  jobId: string; detail: boolean; onPageChange: (index: number) => void
}) {
  const viewport = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 600, height: 700 })
  const [scrollTop, setScrollTop] = useState(0)
  const [compare, setCompare] = useState(50)
  const frame = useRef<number | null>(null)
  const currentIndex = useRef(pageIndex)
  currentIndex.current = pageIndex
  useEffect(() => {
    const element = viewport.current!
    const observer = new ResizeObserver(() => setSize({ width: element.clientWidth, height: element.clientHeight }))
    observer.observe(element)
    return () => { observer.disconnect(); if (frame.current != null) cancelAnimationFrame(frame.current) }
  }, [])
  const layout = useMemo(() => {
    let top = 0
    const width = Math.max(1, Math.min(1100, size.width - 16))
    const rows = pages.map(page => {
      const height = width * (page.height || 1400) / (page.width || 1000)
      const row = { page, top, height, width }
      top += height + 36
      return row
    })
    return { rows, height: top }
  }, [pages, size.width])
  useEffect(() => {
    const row = layout.rows.find(item => item.page.page_index === currentIndex.current)
    if (row && viewport.current) { viewport.current.scrollTop = row.top; setScrollTop(row.top) }
  }, [jumpToken, size.width, pages.length])
  const firstVisible = Math.max(0, layout.rows.findIndex(row => row.top + row.height > scrollTop))
  // Three page nodes (six decoded images in comparison mode), not 244 full images.
  const first = Math.max(0, firstVisible - 1)
  const rows = layout.rows.slice(first, first + 3)
  return <div ref={viewport} className="continuous-reader" role="region" aria-label="垂直连续阅读" tabIndex={0}
    onScroll={() => {
      if (frame.current != null) return
      frame.current = requestAnimationFrame(() => {
        frame.current = null
        const top = viewport.current?.scrollTop || 0
        setScrollTop(top)
        const row = layout.rows.find(item => item.top + item.height > top + Math.min(160, size.height / 3))
        if (row && row.page.page_index !== currentIndex.current) onPageChange(row.page.page_index)
      })
    }}>
    <div className="continuous-track" style={{ height: layout.height }}>
      {rows.map(({ page, top, height, width }) => {
        const source = readingUrl(page, 'source', detail)
        const final = readingUrl(page, 'final', detail)
        const variant = mode === 'source' || !final ? 'source' : 'final'
        return <article key={page.page_index} className="continuous-page" data-page-index={page.page_index}
          style={{ top, height, width }}>
          {mode === 'compare' && final ? <div className="continuous-compare">
            <ReaderImage url={source!} placeholder={placeholderUrl(page, 'source')} alt={`第 ${page.page_index + 1} 页原图`} />
            <div className="continuous-result" style={{ clipPath: `inset(0 0 0 ${compare}%)` }}><ReaderImage url={final} placeholder={placeholderUrl(page, 'final')} alt={`第 ${page.page_index + 1} 页结果`} /></div>
            <input type="range" aria-label={`第 ${page.page_index + 1} 页对比位置`} value={compare} onChange={event => setCompare(Number(event.target.value))} />
          </div> : <ReaderImage url={mode === 'mask' ? `/api/jobs/${jobId}/pages/${page.page_index}/mask` : readingUrl(page, variant, detail)!}
            placeholder={placeholderUrl(page, variant)} alt={`第 ${page.page_index + 1} 页${variant === 'source' ? '原图' : '结果'}`} />}
          <footer>第 {page.page_index + 1} / {pages.length} 页{!final && mode === 'final' ? ' · 上色结果尚未就绪，当前为原图' : ''}</footer>
        </article>
      })}
    </div>
  </div>
}
