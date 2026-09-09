import { type CSSProperties, useEffect, useState } from 'react'

type Entry = { image: HTMLImageElement; objectUrl: string }
type Work = { url: string; controller: AbortController }
// This cache holds small display images only. Downloads always use native streaming.
class ReaderCache {
  entries = new Map<string, Entry>()
  errors = new Map<string, string>()
  private listeners = new Map<string, Set<() => void>>()
  private active = new Map<string, Work>()
  private queue: string[] = []
  private wanted = new Set<string>()

  subscribe(url: string, notify: () => void) {
    const listeners = this.listeners.get(url) || new Set()
    listeners.add(notify)
    this.listeners.set(url, listeners)
    this.request(url, true)
    return () => { listeners.delete(notify); this.trim() }
  }
  private notify(url: string) { this.listeners.get(url)?.forEach(callback => callback()) }
  private trim() {
    for (const [url, entry] of this.entries) {
      if (this.entries.size <= 8) break
      if (this.listeners.get(url)?.size) continue
      URL.revokeObjectURL(entry.objectUrl)
      this.entries.delete(url)
    }
    for (const [url, listeners] of this.listeners) if (!listeners.size) this.listeners.delete(url)
  }
  request(url: string, priority = false) {
    const entry = this.entries.get(url)
    if (entry) {
      this.entries.delete(url); this.entries.set(url, entry)
      return
    }
    if (this.active.has(url) || this.errors.has(url)) return
    this.queue = this.queue.filter(item => item !== url)
    if (priority) this.queue.unshift(url); else this.queue.push(url)
    this.pump()
  }
  prefetch(urls: string[]) {
    this.wanted = new Set(urls)
    this.queue = this.queue.filter(url => this.wanted.has(url) || this.listeners.get(url)?.size)
    for (const [url, work] of this.active) {
      if (!this.wanted.has(url) && !this.listeners.get(url)?.size) work.controller.abort()
    }
    this.queue = [...new Set([...urls.filter(url => !this.entries.has(url) && !this.active.has(url) && !this.errors.has(url)), ...this.queue])]
    this.pump()
  }
  retry(url: string) { this.errors.delete(url); this.notify(url); this.request(url, true) }
  private pump() {
    while (this.active.size < 2 && this.queue.length) {
      const url = this.queue.shift()!
      if (this.entries.has(url) || this.active.has(url)) continue
      const work = { url, controller: new AbortController() }
      this.active.set(url, work)
      void this.load(work)
    }
  }
  private async load(work: Work) {
    let objectUrl: string | undefined
    const deadline = window.setTimeout(() => work.controller.abort('timeout'), 30000)
    try {
      for (let attempt = 0; attempt < 12; attempt++) {
        const response = await fetch(work.url, { signal: work.controller.signal, cache: 'default',
          priority: this.listeners.get(work.url)?.size ? 'high' : 'low' })
        if (response.status === 202) {
          await new Promise<void>((resolve, reject) => {
            const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')) }
            const timer = window.setTimeout(() => { work.controller.signal.removeEventListener('abort', abort); resolve() }, 1000)
            work.controller.signal.addEventListener('abort', abort, { once: true })
            if (work.controller.signal.aborted) abort()
          })
          continue
        }
        if (!response.ok) throw new Error(response.status === 503 ? '本地服务暂时离线' : `图片读取失败（${response.status}）`)
        if (!response.headers.get('content-type')?.startsWith('image/')) throw new Error('登录可能已过期，请重新登录后重试')
        objectUrl = URL.createObjectURL(await response.blob())
        const image = new Image()
        image.decoding = 'async'
        image.src = objectUrl
        await image.decode()
        if (work.controller.signal.aborted) throw new DOMException('Aborted', 'AbortError')
        this.entries.set(work.url, { image, objectUrl })
        objectUrl = undefined
        this.trim()
        this.notify(work.url)
        return
      }
      throw new Error('阅读图片仍在准备，请稍后重试')
    } catch (error) {
      if (!work.controller.signal.aborted || work.controller.signal.reason === 'timeout') {
        this.errors.set(work.url, work.controller.signal.reason === 'timeout' ? '图片加载超时，请重试' : error instanceof Error ? error.message : '图片加载失败')
        this.notify(work.url)
      }
    } finally {
      clearTimeout(deadline)
      if (objectUrl) URL.revokeObjectURL(objectUrl)
      this.active.delete(work.url)
      this.pump()
    }
  }
}
export const readerCache = new ReaderCache()

export function ReaderImage({ url, placeholder, alt, style, onReady }: {
  url: string; placeholder?: string | null; alt: string; style?: CSSProperties; onReady?: () => void
}) {
  const [, update] = useState(0)
  useEffect(() => readerCache.subscribe(url, () => update(value => value + 1)), [url])
  const entry = readerCache.entries.get(url)
  const error = readerCache.errors.get(url)
  useEffect(() => { if (entry || error) onReady?.() }, [url, entry, error]) // onReady is notification, not a load dependency.
  return <>
    <img src={entry?.objectUrl || placeholder || undefined} alt={alt} style={style}
      className={entry ? 'reader-image decoded' : 'reader-image placeholder'} draggable={false}
      data-reading-url={url} data-decoded={!!entry} />
    {!entry && <span className={`reader-image-status ${error ? 'error' : ''}`} role="status">
      {error || '正在加载清晰图片…'}
      {error && <button type="button" onClick={() => readerCache.retry(url)}>重试</button>}
    </span>}
  </>
}

export function ReaderThumbnail({ url, alt }: { url?: string | null; alt: string }) {
  const [attempt, setAttempt] = useState(0)
  const [failed, setFailed] = useState(false)
  useEffect(() => { setAttempt(0); setFailed(false) }, [url])
  useEffect(() => {
    if (!failed || attempt >= 4) return
    const timer = window.setTimeout(() => { setFailed(false); setAttempt(value => value + 1) }, 1000)
    return () => clearTimeout(timer)
  }, [failed, attempt])
  return <img key={`${url}:${attempt}`} loading="lazy" decoding="async" fetchPriority="low"
    src={url || undefined} alt={alt} onError={() => setFailed(true)} />
}
