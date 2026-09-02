import { ChangeEvent, DragEvent, type KeyboardEvent as ReactKeyboardEvent, type MouseEvent, type PointerEvent as ReactPointerEvent, type ReactElement, type WheelEvent as ReactWheelEvent, useEffect, useMemo, useRef, useState } from 'react'

type Descriptor = {
  id: string
  name: string
  description: string
  best_for?: string
  changes?: string
  tradeoff?: string
  speed?: string
  memory?: string
}

type Presets = {
  colors: Descriptor[]
  styles: Descriptor[]
  modes: Descriptor[]
  details: Descriptor[]
  panels: Descriptor[]
  outputs: Descriptor[]
}

type Progress = {
  stage: string
  stage_percent: number
  completed_units: number
  total_units: number
  failed_units: number
  completed_pages: number
  total_pages: number
  current_page: number | null
  percent: number
  eta_seconds: number | null
  elapsed_seconds: number
  seconds_per_megapixel: number | null
  uploaded_bytes: number
  total_upload_bytes: number
  ready_page_indices: number[]
  bytes_processed?: number
  bytes_total?: number
  current_unit?: number | null
  latest_message?: string | null
  control_state?: {
    action: string
    requested_at?: string | null
    deadline_at?: string | null
    active_request?: boolean
    message?: string | null
  } | null
  page_states?: Array<{
    page_index: number
    status: string
    completed_units: number
    total_units: number
    error?: string | null
  }>
}

type Job = {
  id: string
  display_name: string
  status: string
  page_count: number
  queue_position: number | null
  error?: string
  spec: Record<string, string | boolean | number>
  progress: Progress
  folder_id?: string | null
  library_order?: number
}

type FolderNode = {
  id: string
  parent_id: string | null
  name: string
  sort_order: number
  archived_at?: string | null
  job_count: number
  children: FolderNode[]
  jobs: Job[]
}

type LibraryTree = {
  folders: FolderNode[]
  root_jobs: Job[]
}

type Page = {
  page_index: number
  status: string
  completed_units?: number
  total_units?: number
  error?: string | null
  source_url: string
  source_display_url?: string | null
  final_url: string | null
  final_display_url?: string | null
  preview_url?: string | null
  preview_only?: boolean
  thumbnail_url: string | null
  asset_revision?: string | null
  mask_status?: string
  semantic_mask?: Record<string, unknown> | null
}

type Uploaded = {
  source_id: string
  name: string
  kind: 'image' | 'book'
  size: number
  duplicate: boolean
  progress: number
}

type ActivityLog = {
  id: number
  job_id: string | null
  kind: string
  created_at: string | number | null
  message: string
}

type RawLog = {
  id?: number
  timestamp: string | number | null
  level: string
  component: string
  job_id?: string | null
  page_index?: number | null
  unit_index?: number | null
  event: string
  message: string
  metrics?: Record<string, unknown>
}

type GpuMetrics = {
  timestamp: string
  available: boolean
  reason?: string | null
  name?: string | null
  utilization_percent?: number | null
  memory_used_mib?: number | null
  memory_total_mib?: number | null
  temperature_c?: number | null
  power_w?: number | null
  cpu_percent?: number | null
  memory_percent?: number | null
  disk_free_gib?: number | null
}

type EngineHealth = {
  ok: boolean
  detail?: string
  error?: string
  state?: 'idle' | 'loading' | 'ready' | 'generating' | 'failed'
  loaded?: boolean
  progress?: number | null
  stage?: string | null
  message?: string | null
}

const statusLabel: Record<string, string> = {
  created: '正在建立', ingesting: '正在展开', ready: '可以开始', queued: '队列等待',
  running: '正在处理', paused: '已暂停', waiting_model: '等待模型',
  needs_attention: '需要检查', completed: '已完成', failed: '处理失败',
  cancelled: '已取消', archived: '回收站',
}

const stageLabel: Record<string, string> = {
  accepted: '已提交', ingesting: '正在展开', indexing: '正在建立页面', masking: '正在准备遮罩',
  reading_source: '正在读取来源', expanding_archive: '正在展开压缩包',
  validating_members: '正在校验页面', writing_pages: '正在写入页面', metadata: '正在生成页面元数据',
  units: '正在建立处理单元', extracting: '正在提取页面', copying: '正在复制页面',
  normalizing: '正在规范页面', rendering: '正在渲染页面',
  ready: '准备完成', queued: '队列等待', loading_model: '正在加载模型', generating: '正在生成',
  repairing: '正在重组结果', completed: '已完成', failed: '处理失败',
}

function progressStageLabel(job: Job): string {
  // The persisted stage describes the last worker phase, while status is the
  // authoritative user-facing state when a task is paused or waiting
  // temporarily. This prevents a paused task from still reading "generating".
  if (['paused', 'waiting_model', 'needs_attention', 'failed', 'cancelled', 'archived'].includes(job.status)) return statusLabel[job.status]
  return stageLabel[job.progress.stage] || statusLabel[job.status] || job.progress.stage || '准备中'
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    cache: options.cache || (!options.method || options.method.toUpperCase() === 'GET' ? 'no-store' : undefined),
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.detail || `请求失败 ${response.status}`)
  }
  return response.json()
}

async function uploadFile(file: File, clientUploadId: string, onProgress: (value: number) => void): Promise<Uploaded> {
  const relativePath = file.webkitRelativePath || file.name
  const chunkSize = 8 * 1024 * 1024
  let offset = 0
  let last: Record<string, unknown> | null = null
  while (offset < file.size || (file.size === 0 && !last)) {
    const end = Math.min(file.size, offset + chunkSize)
    const chunk = file.slice(offset, end)
    const result = await new Promise<{ status: number; data: Record<string, unknown> }>((resolve, reject) => {
      const request = new XMLHttpRequest()
      const form = new FormData()
      form.append('file', chunk, file.name)
      form.append('client_upload_id', clientUploadId)
      form.append('relative_path', relativePath)
      request.open('POST', '/api/import')
      request.timeout = 30 * 60 * 1000
      request.setRequestHeader('X-Upload-ID', clientUploadId)
      request.setRequestHeader('X-Upload-Offset', String(offset))
      request.setRequestHeader('X-Upload-Total', String(file.size))
      request.upload.onprogress = event => {
        if (event.lengthComputable) onProgress(Math.round((offset + event.loaded) / Math.max(1, file.size) * 100))
      }
      request.onerror = () => reject(new Error(`无法上传 ${file.name}`))
      request.ontimeout = () => reject(new Error(`上传 ${file.name} 超时，可以继续重试`))
      request.onload = () => {
        let data: Record<string, unknown> = {}
        try { data = JSON.parse(request.responseText || '{}') as Record<string, unknown> } catch { /* handled below */ }
        resolve({ status: request.status, data })
      }
      request.send(form)
    })
    if (result.status === 409) {
      const expected = Number(result.data.expected_offset)
      if (Number.isFinite(expected) && expected >= 0 && expected <= file.size && expected !== offset) {
        offset = expected
        continue
      }
    }
    if (result.status < 200 || result.status >= 300) throw new Error(String(result.data.detail || `无法上传 ${file.name}`))
    last = result.data
    const uploaded = Number(result.data.uploaded_bytes)
    offset = Number.isFinite(uploaded) ? uploaded : end
    onProgress(Math.round(offset / Math.max(1, file.size) * 100))
    if (result.data.complete || result.status === 201) break
  }
  if (!last || !last.source_id) throw new Error(`上传 ${file.name} 未返回来源编号`)
  return { ...last, progress: 100 } as Uploaded
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatEta(seconds: number | null) {
  if (seconds == null) return '正在估算'
  if (seconds < 60) return `约 ${seconds} 秒`
  return `约 ${Math.ceil(seconds / 60)} 分钟`
}

function formatLogTime(value: unknown): string {
  let date: Date
  if (typeof value === 'number' && Number.isFinite(value)) {
    date = new Date(Math.abs(value) < 1e12 ? value * 1000 : value)
  } else if (typeof value === 'string' && value.trim()) {
    const numeric = Number(value)
    date = Number.isFinite(numeric)
      ? new Date(Math.abs(numeric) < 1e12 ? numeric * 1000 : numeric)
      : new Date(value)
  } else {
    return '时间未知'
  }
  if (Number.isNaN(date.getTime())) return '时间未知'
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

function latestReadyPageIndex(job: Job | null, pages: Page[] = []) {
  const readyFromPages = pages
    .filter(page => Boolean(page.final_url || page.preview_url))
    .map(page => page.page_index)
  if (readyFromPages.length) return readyFromPages[readyFromPages.length - 1]
  const readyFromProgress = job?.progress.ready_page_indices || []
  if (readyFromProgress.length) return readyFromProgress[readyFromProgress.length - 1]
  return job?.progress.current_page ?? 0
}

function freshAssetUrl(value: string, revision: string | number = Date.now()) {
  const separator = value.includes('?') ? '&' : '?'
  return `${value}${separator}pt_revision=${encodeURIComponent(String(revision))}`
}

function Icon({ name }: { name: string }) {
  const icons: Record<string, string> = {
    add: '+', settings: '⚙', books: '▤', preview: '◫', progress: '◒',
    play: '▶', pause: 'Ⅱ', more: '•••', close: '×', upload: '↑', check: '✓',
    'chevron-left': '‹', 'chevron-right': '›',
  }
  return <span aria-hidden="true" className="icon">{icons[name] || '•'}</span>
}

function OptionInfo({ item }: { item?: Descriptor }) {
  if (!item) return null
  return (
    <div className="option-info">
      <p>{item.description}</p>
      <dl>
        {item.best_for && <><dt>适合</dt><dd>{item.best_for}</dd></>}
        {item.changes && <><dt>改变</dt><dd>{item.changes}</dd></>}
        {item.tradeoff && <><dt>代价</dt><dd>{item.tradeoff}</dd></>}
        {item.memory && <><dt>显存</dt><dd>{item.memory}</dd></>}
      </dl>
    </div>
  )
}

function App() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [libraryTree, setLibraryTree] = useState<LibraryTree | null>(null)
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [libraryContext, setLibraryContext] = useState<{ x: number; y: number; type: 'folder' | 'job'; id: string } | null>(null)
  const [draggedLibraryId, setDraggedLibraryId] = useState<string | null>(null)
  const [draggedFolderId, setDraggedFolderId] = useState<string | null>(null)
  const [presets, setPresets] = useState<Presets | null>(null)
  const [health, setHealth] = useState<Record<string, EngineHealth>>({})
  const [logs, setLogs] = useState<ActivityLog[]>([])
  const [rawLogs, setRawLogs] = useState<RawLog[]>([])
  const [gpuMetrics, setGpuMetrics] = useState<GpuMetrics | null>(null)
  const [logKind, setLogKind] = useState<'activity' | 'raw' | 'gpu'>('activity')
  const [logLevel, setLogLevel] = useState('all')
  const [logComponent, setLogComponent] = useState('all')
  const [autoScrollLogs, setAutoScrollLogs] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [pages, setPages] = useState<Page[]>([])
  const [pageIndex, setPageIndex] = useState(0)
  const [importOpen, setImportOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [progressOpen, setProgressOpen] = useState(true)
  const [previewDark, setPreviewDark] = useState(true)
  const [previewMode, setPreviewMode] = useState<'compare' | 'source' | 'final' | 'mask'>('compare')
  const [compare, setCompare] = useState(50)
  const [zoom, setZoom] = useState<'fit' | '50' | '100' | '200'>('fit')
  const [canvasScale, setCanvasScale] = useState(1)
  const [canvasPan, setCanvasPan] = useState({ x: 0, y: 0 })
  const [pageLoading, setPageLoading] = useState(false)
  const [readyNotice, setReadyNotice] = useState<number | null>(null)
  const [leftCollapsed, setLeftCollapsed] = useState(() => localStorage.getItem('paneltone.leftCollapsed') === '1')
  const [rightCollapsed, setRightCollapsed] = useState(() => localStorage.getItem('paneltone.rightCollapsed') === '1')
  const [taskMenuOpen, setTaskMenuOpen] = useState(false)
  const [mobileTab, setMobileTab] = useState<'books' | 'preview' | 'settings' | 'progress'>('preview')
  const [largeText, setLargeText] = useState(() => localStorage.getItem('paneltone.largeText') !== '0')
  const [filter, setFilter] = useState('all')
  const [message, setMessage] = useState('')
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set())
  const [renameTarget, setRenameTarget] = useState<Job | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Job | null>(null)
  const eventSource = useRef<EventSource | null>(null)
  const taskMenuRef = useRef<HTMLDivElement | null>(null)
  const libraryContextRef = useRef<HTMLDivElement | null>(null)
  const logKindRef = useRef(logKind)
  const selectedRef = useRef<string | null>(null)
  const jobsRequestRef = useRef(0)
  const jobsAbortRef = useRef<AbortController | null>(null)
  const libraryAbortRef = useRef<AbortController | null>(null)
  const healthRequestRef = useRef(0)
  const healthAbortRef = useRef<AbortController | null>(null)
  const gpuAbortRef = useRef<AbortController | null>(null)
  const pagesRequestRef = useRef(0)
  const pagesAbortRef = useRef<AbortController | null>(null)
  const logsRequestRef = useRef(0)
  const rawLogsRequestRef = useRef(0)
  const pagesRef = useRef<Page[]>([])
  const pageIndexRef = useRef(0)
  const pageSelectionRef = useRef(new Map<string, number>())
  const manualPageSelectionRef = useRef(new Set<string>())
  const logListRef = useRef<HTMLDivElement | null>(null)
  const activeFilmstripPageRef = useRef<HTMLButtonElement | null>(null)
  const eventRefreshTimerRef = useRef<number | null>(null)
  const pendingEventRefreshRef = useRef({ jobs: false, library: false, pages: false, logs: false, health: false })
  const pageLoadCountRef = useRef(0)
  const pageLoadTargetRef = useRef(1)
  const prefetchQueueRef = useRef<string[]>([])
  const prefetchActiveRef = useRef(0)
  const prefetchedUrlsRef = useRef(new Set<string>())
  const gestureRef = useRef({
    pointers: new Map<number, { x: number; y: number }>(),
    startPan: { x: 0, y: 0 },
    startScale: 1,
    startDistance: 0,
    swipeStart: { x: 0, y: 0 },
  })

  const selected = jobs.find(job => job.id === selectedId) || jobs[0] || null
  const currentPage = pages.find(page => page.page_index === pageIndex) || pages[0]
  const currentSourceUrl = currentPage?.source_display_url || currentPage?.source_url || null
  const currentResultUrl = currentPage?.final_display_url || currentPage?.final_url || currentPage?.preview_url || null
  const selectedEngine = typeof selected?.spec.engine === 'string' ? selected.spec.engine : null
  const activeEngine = selectedEngine || 'palette'
  const modelHealth = health[activeEngine]
  const modelReady = activeEngine === 'palette' || Boolean(modelHealth?.ok)
  const modelLoading = activeEngine !== 'palette' && modelHealth?.state === 'loading'
  const modelStatusText = activeEngine === 'palette' ? '内置引擎已就绪' : modelHealth?.state === 'loading' ? '正在加载模型权重' : modelHealth?.state === 'generating' ? '模型正在生成' : modelReady ? '本地模型已连接' : jobs.some(job => job.status === 'running') ? '模型正在启动' : '模型服务未连接'
  const waitingForModel = selected?.status === 'waiting_model'
  const waitingPageTitle = selected?.status === 'paused'
    ? '任务已暂停'
    : selected?.status === 'cancelled'
      ? '任务已取消'
      : selected?.status === 'needs_attention'
        ? '这一页需要检查'
        : selected?.status === 'queued'
          ? '正在等待前面的任务'
          : selected?.status === 'ingesting' || selected?.status === 'created'
            ? '正在准备页面'
            : '这一页正在处理'
  const waitingPageHint = selected?.status === 'paused'
    ? '继续任务后会从当前页接着处理'
    : selected?.status === 'cancelled'
      ? '重新开始任务后才会生成页面结果'
      : selected?.status === 'needs_attention'
        ? '请在进度详情中查看错误并重试失败页'
        : selected?.status === 'queued'
          ? '前面的任务完成后会自动开始'
          : selected?.status === 'ingesting' || selected?.status === 'created'
            ? '漫画展开完成后会显示页面'
            : '完成后会自动出现在这里'

  useEffect(() => { selectedRef.current = selected?.id || null }, [selected?.id])
  // A page-ready notice belongs to the book that produced it.  Clear it when
  // the user switches books so a stale toast cannot claim that another book's
  // page is ready.
  useEffect(() => { setReadyNotice(null) }, [selected?.id])
  useEffect(() => { pagesRef.current = pages }, [pages])
  useEffect(() => { pageIndexRef.current = pageIndex }, [pageIndex])
  useEffect(() => { logKindRef.current = logKind }, [logKind])
  useEffect(() => { localStorage.setItem('paneltone.leftCollapsed', leftCollapsed ? '1' : '0') }, [leftCollapsed])
  useEffect(() => { localStorage.setItem('paneltone.rightCollapsed', rightCollapsed ? '1' : '0') }, [rightCollapsed])
  useEffect(() => { localStorage.setItem('paneltone.largeText', largeText ? '1' : '0') }, [largeText])
  useEffect(() => {
    const handleKeyboard = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') { setTaskMenuOpen(false); setLibraryContext(null) }
      if (!event.ctrlKey || !event.shiftKey) return
      const target = event.target as HTMLElement | null
      if (target?.matches('input, textarea, select')) return
      if (event.key.toLowerCase() === 'b') { event.preventDefault(); setLeftCollapsed(value => !value) }
      if (event.key.toLowerCase() === 'i') { event.preventDefault(); setRightCollapsed(value => !value) }
    }
    window.addEventListener('keydown', handleKeyboard)
    return () => window.removeEventListener('keydown', handleKeyboard)
  }, [])
  useEffect(() => {
    if (!taskMenuOpen) return
    const closeOutside = (event: PointerEvent) => {
      if (taskMenuRef.current && !taskMenuRef.current.contains(event.target as Node)) setTaskMenuOpen(false)
    }
    document.addEventListener('pointerdown', closeOutside)
    return () => document.removeEventListener('pointerdown', closeOutside)
  }, [taskMenuOpen])
  useEffect(() => {
    if (!libraryContext) return
    const closeContext = (event: PointerEvent) => {
      if (libraryContextRef.current && libraryContextRef.current.contains(event.target as Node)) return
      setLibraryContext(null)
    }
    document.addEventListener('pointerdown', closeContext)
    return () => document.removeEventListener('pointerdown', closeContext)
  }, [libraryContext])

  async function refreshJobs() {
    const requestId = ++jobsRequestRef.current
    jobsAbortRef.current?.abort()
    const controller = new AbortController()
    jobsAbortRef.current = controller
    try {
      const result = await api<Job[]>('/api/jobs?include_archived=true', { signal: controller.signal })
      if (requestId !== jobsRequestRef.current) return
      setJobs(result)
      if (!result.some(job => job.id === selectedRef.current)) setSelectedId(result[0]?.id || null)
    } catch (reason) {
      if (!controller.signal.aborted) throw reason
    }
  }

  async function refreshLibrary() {
    libraryAbortRef.current?.abort()
    const controller = new AbortController()
    try {
      libraryAbortRef.current = controller
      const tree = await api<LibraryTree>('/api/library/tree?include_archived=true', { signal: controller.signal })
      if (controller.signal.aborted) return
      setLibraryTree(tree)
      setExpandedFolders(current => {
        if (current.size) return current
        return new Set(tree.folders.map(folder => folder.id))
      })
    } catch (reason) {
      if (controller.signal.aborted) return
      setLibraryTree(null)
      throw reason
    }
  }

  async function refreshHealth() {
    const requestId = ++healthRequestRef.current
    healthAbortRef.current?.abort()
    const controller = new AbortController()
    healthAbortRef.current = controller
    try {
      const data = await api<{ engines: Record<string, EngineHealth> }>('/api/health', { signal: controller.signal })
      if (requestId !== healthRequestRef.current) return
      setHealth(data.engines)
    } catch (reason) {
      if (!controller.signal.aborted) throw reason
    }
  }

  async function refreshLogs(jobId = selectedRef.current) {
    const requestId = ++logsRequestRef.current
    if (!jobId) return setLogs([])
    const result = await api<ActivityLog[]>(`/api/logs?job_id=${encodeURIComponent(jobId)}&limit=60`)
    if (requestId === logsRequestRef.current && jobId === selectedRef.current) setLogs(result)
  }

  async function refreshRawLogs(jobId = selectedRef.current) {
    const requestId = ++rawLogsRequestRef.current
    const query = new URLSearchParams({ kind: logKind, limit: '80' })
    if (jobId && logKind !== 'gpu') query.set('job_id', jobId)
    const result = await api<RawLog[]>(`/api/logs?${query.toString()}`)
    if (requestId === rawLogsRequestRef.current) setRawLogs(result)
  }

  async function refreshGpu() {
    gpuAbortRef.current?.abort()
    const controller = new AbortController()
    gpuAbortRef.current = controller
    try {
      const result = await api<GpuMetrics>('/api/gpu', { signal: controller.signal })
      if (!controller.signal.aborted) setGpuMetrics(result)
    } catch (reason) {
      if (!controller.signal.aborted) throw reason
    }
  }

  async function refreshPages(jobId = selected?.id) {
    const requestId = ++pagesRequestRef.current
    pagesAbortRef.current?.abort()
    const controller = new AbortController()
    pagesAbortRef.current = controller
    if (!jobId) {
      setPages([])
      return
    }
    try {
      const result = await api<Page[]>(`/api/jobs/${jobId}/pages`, { signal: controller.signal })
      if (requestId !== pagesRequestRef.current || jobId !== selectedRef.current) return
      pagesRef.current = result
      setPages(result)
      const manuallySelected = manualPageSelectionRef.current.has(jobId)
      const selectedPage = pageSelectionRef.current.get(jobId)
      if (manuallySelected && selectedPage != null && result.some(page => page.page_index === selectedPage)) {
        setPageIndex(selectedPage)
        return
      }
      if (!manuallySelected) {
        const readyPage = latestReadyPageIndex(jobs.find(job => job.id === jobId) || null, result)
        setPageIndex(readyPage)
        pageSelectionRef.current.set(jobId, readyPage)
        return
      }
      const current = result.find(page => page.page_index === pageIndexRef.current)
      if (!current || !(current.final_url || current.preview_url)) {
        const readyPage = latestReadyPageIndex(jobs.find(job => job.id === jobId) || null, result)
        setPageIndex(readyPage)
        pageSelectionRef.current.set(jobId, readyPage)
      }
    } catch (reason) {
      if (!controller.signal.aborted) throw reason
    }
  }

  function scheduleEventRefresh(flags: Partial<typeof pendingEventRefreshRef.current>) {
    const pending = pendingEventRefreshRef.current
    pendingEventRefreshRef.current = {
      jobs: pending.jobs || Boolean(flags.jobs),
      library: pending.library || Boolean(flags.library),
      pages: pending.pages || Boolean(flags.pages),
      logs: pending.logs || Boolean(flags.logs),
      health: pending.health || Boolean(flags.health),
    }
    if (eventRefreshTimerRef.current != null) return
    eventRefreshTimerRef.current = window.setTimeout(() => {
      eventRefreshTimerRef.current = null
      const pending = pendingEventRefreshRef.current
      pendingEventRefreshRef.current = { jobs: false, library: false, pages: false, logs: false, health: false }
      if (pending.jobs) refreshJobs().catch(() => undefined)
      if (pending.library) refreshLibrary().catch(() => undefined)
      if (pending.pages) refreshPages(selectedRef.current || undefined).catch(() => undefined)
      if (pending.logs) {
        refreshLogs(selectedRef.current).catch(() => undefined)
        if (logKindRef.current !== 'activity') refreshRawLogs(selectedRef.current).catch(() => undefined)
      }
      if (pending.health) refreshHealth().catch(() => undefined)
    }, 160)
  }

  function pumpPrefetchQueue() {
    while (prefetchActiveRef.current < 2 && prefetchQueueRef.current.length) {
      const url = prefetchQueueRef.current.shift()
      if (!url) continue
      prefetchActiveRef.current += 1
      const image = new Image()
      image.decoding = 'async'
      const finish = () => {
        prefetchActiveRef.current = Math.max(0, prefetchActiveRef.current - 1)
        pumpPrefetchQueue()
      }
      image.onload = finish
      image.onerror = finish
      image.src = url
    }
  }

  function queueImagePrefetch(url: string | null | undefined) {
    if (!url || prefetchedUrlsRef.current.has(url)) return
    prefetchedUrlsRef.current.add(url)
    prefetchQueueRef.current.push(url)
    pumpPrefetchQueue()
  }

  function prefetchPageAssets(jobId: string | null, index: number, sourcePages = pagesRef.current) {
    if (!jobId) return
    const targets = sourcePages
      .filter(page => Math.abs(page.page_index - index) <= 2)
      .sort((left, right) => Math.abs(left.page_index - index) - Math.abs(right.page_index - index))
    for (const page of targets) {
      queueImagePrefetch(page.source_display_url || page.source_url)
      queueImagePrefetch(page.final_display_url || page.final_url || page.preview_url)
    }
  }

  function selectPage(index: number) {
    if (!selected || !pagesRef.current.some(page => page.page_index === index)) return
    manualPageSelectionRef.current.add(selected.id)
    pageSelectionRef.current.set(selected.id, index)
    setPageLoading(true)
    setCanvasPan({ x: 0, y: 0 })
    setPageIndex(index)
    setReadyNotice(current => current === index ? null : current)
    prefetchPageAssets(selected.id, index)
  }

  function movePage(offset: number) {
    const currentPosition = pagesRef.current.findIndex(page => page.page_index === pageIndexRef.current)
    const target = pagesRef.current[currentPosition + offset]
    if (target) selectPage(target.page_index)
  }

  function setZoomPreset(value: 'fit' | '50' | '100' | '200') {
    setZoom(value)
    setCanvasScale(value === 'fit' ? 1 : Number(value) / 100)
    setCanvasPan({ x: 0, y: 0 })
  }

  function updateScale(value: number) {
    const next = Math.min(4, Math.max(0.25, value))
    setCanvasScale(next)
    if (Math.abs(next - 0.5) < 0.02) setZoom('50')
    else if (Math.abs(next - 1) < 0.02) setZoom('100')
    else if (Math.abs(next - 2) < 0.04) setZoom('200')
    else setZoom('fit')
  }

  function markPageImageLoaded() {
    pageLoadCountRef.current += 1
    if (pageLoadCountRef.current >= pageLoadTargetRef.current) setPageLoading(false)
  }

  function handleCanvasWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (!currentPage) return
    event.preventDefault()
    updateScale(canvasScale * Math.exp(-event.deltaY * 0.001))
  }

  function handleCanvasPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest('button, input, a')) return
    const gesture = gestureRef.current
    gesture.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
    gesture.startPan = canvasPan
    gesture.startScale = canvasScale
    gesture.swipeStart = { x: event.clientX, y: event.clientY }
    if (gesture.pointers.size === 2) {
      const points = Array.from(gesture.pointers.values())
      gesture.startDistance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y)
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function handleCanvasPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current
    if (!gesture.pointers.has(event.pointerId)) return
    gesture.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (gesture.pointers.size >= 2) {
      const points = Array.from(gesture.pointers.values())
      const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y)
      if (gesture.startDistance > 0) updateScale(gesture.startScale * distance / gesture.startDistance)
      return
    }
    if (canvasScale <= 1.05) return
    const dx = event.clientX - gesture.swipeStart.x
    const dy = event.clientY - gesture.swipeStart.y
    setCanvasPan({ x: gesture.startPan.x + dx, y: gesture.startPan.y + dy })
  }

  function handleCanvasPointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current
    const point = gesture.pointers.get(event.pointerId)
    const wasSingle = gesture.pointers.size === 1
    gesture.pointers.delete(event.pointerId)
    if (wasSingle && point && canvasScale <= 1.05) {
      const dx = point.x - gesture.swipeStart.x
      const dy = point.y - gesture.swipeStart.y
      if (Math.abs(dx) >= 70 && Math.abs(dx) > Math.abs(dy) * 1.25) movePage(dx < 0 ? 1 : -1)
    }
    if (gesture.pointers.size < 2) gesture.startDistance = 0
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  useEffect(() => {
    if (!currentPage) return
    pageLoadCountRef.current = 0
    const imageWillLoad = previewMode === 'source' || previewMode === 'mask' || Boolean(currentResultUrl)
    pageLoadTargetRef.current = previewMode === 'compare' && Boolean(currentResultUrl) ? 2 : 1
    setPageLoading(imageWillLoad)
    prefetchPageAssets(selected?.id || null, pageIndex)
  }, [selected?.id, pageIndex, currentPage?.asset_revision, previewMode, currentResultUrl])

  useEffect(() => {
    activeFilmstripPageRef.current?.scrollIntoView({
      behavior: 'auto',
      block: 'nearest',
      inline: 'center',
    })
  }, [selected?.id, pageIndex, pages.length])

  function handlePageReady(data: Record<string, unknown>) {
    const jobId = String(data.job_id || '')
    const index = Number(data.page_index)
    if (!jobId || jobId !== selectedRef.current || !Number.isInteger(index) || index < 0) return
    const hasPreview = typeof data.preview_url === 'string'
    // All variants emitted for one page share the persisted asset revision.
    // Use one cache key for the complete event so source, preview, final and
    // thumbnail cannot briefly show different generations after a repair.
    const assetRevision = String(data.asset_revision || Date.now())
    const readyPage: Page = {
      page_index: index,
      status: typeof data.status === 'string' ? data.status : hasPreview ? 'processing' : 'qa_passed',
      completed_units: Number(data.completed_units || 0),
      total_units: Number(data.total_units || data.completed_units || 0),
      error: null,
      source_url: typeof data.source_url === 'string'
        ? freshAssetUrl(data.source_url, assetRevision)
        : `/api/jobs/${jobId}/pages/${index}/source`,
      source_display_url: `/api/assets/jobs/${jobId}/pages/${index}/source.webp?v=${encodeURIComponent(assetRevision)}`,
      final_url: typeof data.final_url === 'string' ? freshAssetUrl(data.final_url, assetRevision) : null,
      final_display_url: typeof data.final_url === 'string' ? `/api/assets/jobs/${jobId}/pages/${index}/final.webp?v=${encodeURIComponent(assetRevision)}` : null,
      preview_url: hasPreview ? freshAssetUrl(String(data.preview_url), assetRevision) : null,
      preview_only: hasPreview && typeof data.final_url !== 'string',
      thumbnail_url: typeof data.thumbnail_url === 'string'
        ? freshAssetUrl(data.thumbnail_url, assetRevision)
        : `/api/jobs/${jobId}/pages/${index}/thumbnail`,
      asset_revision: assetRevision,
    }
    pagesRef.current = (() => {
      const existing = pagesRef.current.some(page => page.page_index === index)
      return existing
        ? pagesRef.current.map(page => page.page_index === index ? { ...page, ...readyPage } : page)
        : [...pagesRef.current, readyPage].sort((left, right) => left.page_index - right.page_index)
    })()
    setPages(pagesRef.current)
    const manuallySelected = manualPageSelectionRef.current.has(jobId)
    if (!manuallySelected) {
      const latest = latestReadyPageIndex(null, pagesRef.current)
      setPageIndex(latest)
      pageSelectionRef.current.set(jobId, latest)
      setReadyNotice(null)
    } else if (pageIndexRef.current !== index) {
      setReadyNotice(index)
    }
    scheduleEventRefresh({ pages: true })
  }

  useEffect(() => {
    Promise.all([
      api<Presets>('/api/presets').then(setPresets),
      refreshHealth(),
      refreshJobs(),
      refreshLibrary(),
      refreshGpu().catch(() => setGpuMetrics(null)),
    ]).catch(error => setMessage(error.message))
    const timer = window.setInterval(() => {
      refreshHealth().catch(() => undefined)
      refreshGpu().catch(() => undefined)
    }, 10000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    eventSource.current?.close()
    const stream = new EventSource('/api/events')
    eventSource.current = stream
    const update = (event: MessageEvent) => {
      const data = JSON.parse(event.data || '{}') as Record<string, unknown>
      if (event.type === 'gpu_metrics') {
        setGpuMetrics(data as unknown as GpuMetrics)
        return
      }
      const jobEvents = ['snapshot', 'job_status', 'job_queued', 'job_ready', 'job_error']
      const pageEvents = ['snapshot', 'page_ready', 'page_preview_ready', 'job_ready', 'results_repaired']
      scheduleEventRefresh({
        jobs: jobEvents.includes(event.type),
        library: jobEvents.includes(event.type),
        pages: pageEvents.includes(event.type),
        logs: !data.job_id || data.job_id === selectedRef.current,
        health: ['snapshot', 'job_status', 'model_progress', 'model_reconnected'].includes(event.type),
      })
      if (event.type === 'page_ready' || event.type === 'page_preview_ready') handlePageReady(data)
      if (event.type === 'results_repaired' && data.job_id === selectedRef.current && selectedRef.current) {
        setReadyNotice(null)
      }
    }
    ;['snapshot', 'job_status', 'job_queued', 'job_progress', 'page_ready', 'page_preview_ready', 'job_ready', 'job_error', 'model_progress', 'model_reconnected', 'unit_started', 'unit_finished', 'ingest_started', 'ingest_progress', 'mask_correction_saved', 'results_repaired', 'gpu_metrics'].forEach(kind => stream.addEventListener(kind, update))
    stream.onerror = () => setMessage('实时连接正在自动恢复')
    stream.onopen = () => setMessage('')
    return () => stream.close()
  }, [])

  useEffect(() => {
    refreshPages(selected?.id).catch(() => setPages([]))
    refreshLogs(selected?.id).catch(() => setLogs([]))
  }, [selected?.id])

  useEffect(() => {
    if (logKind === 'activity') return
    refreshRawLogs(selected?.id).catch(() => setRawLogs([]))
  }, [selected?.id, logKind])

  const visibleJobs = useMemo(() => jobs.filter(job => {
    if (filter === 'active') return ['queued', 'running', 'waiting_model'].includes(job.status)
    if (filter === 'attention') return ['failed', 'needs_attention'].includes(job.status)
    if (filter === 'done') return job.status === 'completed'
    if (filter === 'trash') return job.status === 'archived'
    return job.status !== 'archived'
  }), [jobs, filter])
  const effectiveProgress = selected
    ? (selected.progress.total_units > 0 ? selected.progress.percent : selected.progress.stage_percent || 0)
    : 0
  const visibleRawLogs = useMemo(() => rawLogs.filter(item => {
    if (logLevel !== 'all' && item.level !== logLevel) return false
    if (logComponent !== 'all' && item.component !== logComponent) return false
    return true
  }), [rawLogs, logLevel, logComponent])
  const logComponents = useMemo(
    () => Array.from(new Set(rawLogs.map(item => item.component).filter(Boolean))).sort(),
    [rawLogs],
  )

  useEffect(() => {
    if (!autoScrollLogs || !logListRef.current) return
    logListRef.current.scrollTop = 0
  }, [logs, visibleRawLogs, autoScrollLogs])

  async function copyLogs() {
    const entries = logKind === 'activity'
      ? logs.map(item => `${item.created_at} ${item.message}`)
      : visibleRawLogs.map(item => JSON.stringify(item))
    try {
      await navigator.clipboard.writeText(entries.join('\n'))
      setMessage('当前日志已复制')
    } catch {
      setMessage('浏览器未允许复制，请使用下载日志')
    }
  }

  async function action(jobId: string, name: string) {
    const labels: Record<string, string> = {
      start: '开始',
      pause: '暂停',
      cancel: '取消',
      archive: '移到回收站',
      restore: '恢复任务',
      retry: '重试失败页',
      'repair-results': '修复已生成结果',
      duplicate: '复制任务',
      rename: '修改书名',
    }
    if (name !== 'download') setMessage(name === 'pause' ? '暂停中，正在中断当前模型请求' : name === 'cancel' ? '取消中，正在清理当前请求' : `正在${labels[name] || '提交操作'}`)
    try {
      if (name === 'download') return void (window.location.href = `/api/jobs/${jobId}/download`)
      const result = await api<{ message?: string }>(`/api/jobs/${jobId}/${name}`, { method: 'POST' })
      if (result.message) setMessage(result.message)
      await refreshJobs()
      await refreshLibrary()
    } catch (error) {
      setMessage((error as Error).message)
    }
  }

  async function libraryFolderAction(actionName: 'create' | 'rename' | 'archive' | 'restore' | 'delete', folder?: FolderNode) {
    try {
      if (actionName === 'create') {
        const name = window.prompt('新建目录名称')?.trim()
        if (!name) return
        await api('/api/folders', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, parent_id: folder?.id || null }) })
      } else if (actionName === 'rename' && folder) {
        const name = window.prompt('修改目录名称', folder.name)?.trim()
        if (!name || name === folder.name) return
        await api(`/api/folders/${folder.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })
      } else if (actionName === 'archive' && folder) {
        await api(`/api/folders/${folder.id}/archive`, { method: 'POST' })
      } else if (actionName === 'restore' && folder) {
        await api(`/api/folders/${folder.id}/restore`, { method: 'POST' })
      } else if (actionName === 'delete' && folder) {
        if (!window.confirm(`永久删除空目录“${folder.name}”？`)) return
        await api(`/api/folders/${folder.id}`, { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirmation: '永久删除' }) })
      }
      setLibraryContext(null)
      await refreshLibrary()
    } catch (reason) { setMessage((reason as Error).message) }
  }

  async function moveLibraryJob(jobId: string, folderId: string | null, beforeJobId: string | null = null) {
    try {
      await api(`/api/library/jobs/${jobId}/move`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder_id: folderId, before_job_id: beforeJobId }) })
      await refreshJobs()
      await refreshLibrary()
    } catch (reason) { setMessage((reason as Error).message) }
  }

  async function moveLibraryFolder(folderId: string, parentId: string | null) {
    try {
      await api(`/api/folders/${folderId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parent_id: parentId }),
      })
      setMessage('目录已移动')
      await refreshLibrary()
    } catch (reason) { setMessage((reason as Error).message) }
  }

  async function reorderLibraryFolders(parentId: string | null, folderId: string, beforeFolderId: string | null) {
    if (!libraryTree) return
    const siblings = parentId
      ? (findFolder(libraryTree.folders, parentId)?.children || [])
      : libraryTree.folders
    const ordered = siblings
      .filter(folder => !folder.archived_at)
      .map(folder => folder.id)
      .filter(id => id !== folderId)
    const targetIndex = beforeFolderId ? ordered.indexOf(beforeFolderId) : -1
    ordered.splice(targetIndex >= 0 ? targetIndex : ordered.length, 0, folderId)
    try {
      await api('/api/library/reorder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parent_id: parentId, folder_ids: ordered }),
      })
      setMessage('目录顺序已更新')
      await refreshLibrary()
    } catch (reason) { setMessage((reason as Error).message) }
  }

  function libraryJobAction(jobId: string, name: string) {
    const job = jobs.find(item => item.id === jobId)
    setLibraryContext(null)
    if (!job) return
    if (name === 'rename') return void setRenameTarget(job)
    if (name === 'delete') return void setDeleteTarget(job)
    action(jobId, name)
  }

  function openLibraryContext(event: MouseEvent, type: 'folder' | 'job', id: string) {
    event.preventDefault()
    setLibraryContext({ x: event.clientX, y: event.clientY, type, id })
  }

  function openLibraryContextFromKeyboard(event: ReactKeyboardEvent<HTMLElement>, type: 'folder' | 'job', id: string) {
    if (event.altKey && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
      event.preventDefault()
      nudgeLibraryNode(type, id, event.key === 'ArrowUp' ? -1 : 1).catch(reason => setMessage((reason as Error).message))
      return
    }
    if (event.key !== 'ContextMenu' && !(event.shiftKey && event.key === 'F10')) return
    event.preventDefault()
    const target = event.currentTarget.getBoundingClientRect()
    setLibraryContext({ x: target.left + Math.min(target.width, 24), y: target.bottom + 4, type, id })
  }

  function findFolderContainingJob(nodes: FolderNode[], jobId: string): FolderNode | null {
    for (const node of nodes) {
      if (node.jobs.some(job => job.id === jobId)) return node
      const nested = findFolderContainingJob(node.children, jobId)
      if (nested) return nested
    }
    return null
  }

  async function nudgeLibraryNode(type: 'folder' | 'job', id: string, delta: -1 | 1) {
    if (!libraryTree) return
    if (type === 'folder') {
      const folder = findFolder(libraryTree.folders, id)
      if (!folder) return
      const siblings = folder.parent_id
        ? (findFolder(libraryTree.folders, folder.parent_id)?.children || [])
        : libraryTree.folders
      const ordered = siblings.filter(item => !item.archived_at).map(item => item.id)
      const index = ordered.indexOf(id)
      const target = index + delta
      if (index < 0 || target < 0 || target >= ordered.length) return
      ;[ordered[index], ordered[target]] = [ordered[target], ordered[index]]
      await api('/api/library/reorder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parent_id: folder.parent_id, folder_ids: ordered }),
      })
    } else {
      const parent = findFolderContainingJob(libraryTree.folders, id)
      const siblings = parent ? parent.jobs : libraryTree.root_jobs
      const ordered = siblings.filter(job => job.status !== 'archived').map(job => job.id)
      const index = ordered.indexOf(id)
      const target = index + delta
      if (index < 0 || target < 0 || target >= ordered.length) return
      ;[ordered[index], ordered[target]] = [ordered[target], ordered[index]]
      await api('/api/library/reorder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_id: parent?.id || null, job_ids: ordered }),
      })
    }
    setMessage('书库顺序已更新')
    await refreshJobs()
    await refreshLibrary()
  }

  async function rename(jobId: string, displayName: string) {
    await api(`/api/jobs/${jobId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: displayName }),
    })
    setRenameTarget(null)
    await refreshJobs()
  }

  async function permanentlyDelete(jobId: string) {
    await api(`/api/jobs/${jobId}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmation: '永久删除' }),
    })
    setDeleteTarget(null)
    setSelectedId(null)
    await refreshJobs()
  }

  async function retryPage(jobId: string, index: number) {
    await api(`/api/jobs/${jobId}/pages/${index}/retry`, { method: 'POST' })
    setMessage(`第 ${index + 1} 页已重新加入队列`)
    await refreshJobs()
  }

  async function batchAction(name: 'start' | 'pause' | 'cancel' | 'archive') {
    const targets = jobs.filter(job => selectedJobs.has(job.id))
    const allowed = targets.filter(job => {
      if (name === 'start') return !['running', 'queued', 'completed', 'archived'].includes(job.status)
      if (name === 'archive') return !['created', 'ingesting', 'running', 'queued', 'waiting_model', 'archived'].includes(job.status)
      return ['running', 'queued'].includes(job.status)
    })
    try {
      await Promise.all(allowed.map(job => api(`/api/jobs/${job.id}/${name}`, { method: 'POST' })))
      setSelectedJobs(new Set())
      setMessage(`${allowed.length} 个任务已执行“${name === 'start' ? '开始' : name === 'pause' ? '暂停' : name === 'cancel' ? '取消' : '归档'}”`)
      await refreshJobs()
    } catch (reason) { setMessage((reason as Error).message) }
  }

  const visibleJobIds = useMemo(() => new Set(visibleJobs.map(job => job.id)), [visibleJobs])

  function renderJobRow(job: Job, depth = 0) {
    return <div className="job-item library-job-item" key={job.id} style={{ paddingLeft: `${depth * 10}px` }} draggable onDragStart={() => { setDraggedLibraryId(job.id); setDraggedFolderId(null) }} onDragEnd={() => setDraggedLibraryId(null)} onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }} onDrop={event => { event.preventDefault(); if (draggedLibraryId && draggedLibraryId !== job.id) moveLibraryJob(draggedLibraryId, job.folder_id || null, job.id) }} onContextMenu={event => openLibraryContext(event, 'job', job.id)}>
      <input type="checkbox" aria-label={`选择 ${job.display_name}`} checked={selectedJobs.has(job.id)} onChange={event => setSelectedJobs(current => { const next = new Set(current); if (event.target.checked) next.add(job.id); else next.delete(job.id); return next })} />
      <button className={`job-row ${selected?.id === job.id ? 'selected' : ''}`} onClick={() => { pageSelectionRef.current.set(job.id, latestReadyPageIndex(job)); setPageLoading(true); setCanvasPan({ x: 0, y: 0 }); setPageIndex(latestReadyPageIndex(job)); setSelectedId(job.id); setMobileTab('preview') }} onKeyDown={event => openLibraryContextFromKeyboard(event, 'job', job.id)}>
        <span className="job-cover"><span className="cover-frame"><img key={`${job.id}-${job.status}-${job.page_count}-${job.progress.completed_pages}`} src={`/api/jobs/${job.id}/pages/0/thumbnail?v=${job.status}-${job.page_count}-${job.progress.completed_pages}`} alt="" onError={event => { const image = event.currentTarget; if (image.dataset.fallback !== '1') { image.dataset.fallback = '1'; image.src = `/api/jobs/${job.id}/pages/0/source?v=${job.status}-${job.page_count}` } else image.style.display = 'none' }} /></span><small>{job.progress.completed_pages || 0}/{job.page_count}</small></span>
        <span className="job-copy"><strong>{job.display_name}</strong><small>{statusLabel[job.status] || job.status}{job.progress.current_page != null ? ` · 第 ${job.progress.current_page + 1} 页` : ''}</small></span>
        <span className={`status-shape ${job.status}`} aria-label={statusLabel[job.status]} />
      </button>
    </div>
  }

  function renderFolderNode(folder: FolderNode, depth = 0): ReactElement | null {
    const folderJobs = folder.jobs.filter(job => visibleJobIds.has(job.id))
    const childNodes = folder.children.map(child => renderFolderNode(child, depth + 1)).filter(Boolean) as ReactElement[]
    const visible = filter === 'trash'
      ? Boolean(folder.archived_at) || folderJobs.length > 0 || childNodes.length > 0
      : !folder.archived_at
    if (!visible && !folderJobs.length && !childNodes.length) return null
    const expanded = expandedFolders.has(folder.id)
    return <div className={`folder-node ${folder.archived_at ? 'archived' : ''}`} key={folder.id}>
      <button className="folder-row" style={{ paddingLeft: `${10 + depth * 12}px` }} aria-expanded={expanded} draggable onDragStart={() => { setDraggedFolderId(folder.id); setDraggedLibraryId(null) }} onDragEnd={() => setDraggedFolderId(null)} onClick={() => setExpandedFolders(current => { const next = new Set(current); if (next.has(folder.id)) next.delete(folder.id); else next.add(folder.id); return next })} onKeyDown={event => openLibraryContextFromKeyboard(event, 'folder', folder.id)} onContextMenu={event => openLibraryContext(event, 'folder', folder.id)} onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }} onDrop={event => { event.preventDefault(); if (draggedFolderId && draggedFolderId !== folder.id) { const dragged = libraryTree ? findFolder(libraryTree.folders, draggedFolderId) : null; if (dragged?.parent_id === folder.parent_id) reorderLibraryFolders(folder.parent_id, draggedFolderId, folder.id); else moveLibraryFolder(draggedFolderId, folder.id) } else if (draggedLibraryId) moveLibraryJob(draggedLibraryId, folder.id) }}>
        <span className="folder-chevron" aria-hidden="true">{expanded ? '▾' : '›'}</span><span aria-hidden="true">▱</span><strong>{folder.name}</strong><small>{folderJobs.length}</small>
      </button>
      {expanded && <div className="folder-children">{folderJobs.map(job => renderJobRow(job, depth + 1))}{childNodes}</div>}
    </div>
  }

  function renderLibraryItems() {
    if (!libraryTree) return visibleJobs.map(job => renderJobRow(job))
    const rootJobs = libraryTree.root_jobs.filter(job => visibleJobIds.has(job.id))
    const folders = libraryTree.folders.map(folder => renderFolderNode(folder)).filter(Boolean)
    return <>{rootJobs.map(job => renderJobRow(job))}{folders}</>
  }

  function findFolder(nodes: FolderNode[], id: string): FolderNode | undefined {
    for (const node of nodes) {
      if (node.id === id) return node
      const nested = findFolder(node.children, id)
      if (nested) return nested
    }
    return undefined
  }

  function flattenFolders(nodes: FolderNode[], depth = 0): Array<{ folder: FolderNode; depth: number }> {
    return nodes.flatMap(folder => [
      { folder, depth },
      ...flattenFolders(folder.children, depth + 1),
    ])
  }

  return (
    <div className={`app-shell ${largeText ? 'large-text' : 'standard-text'}`}>
      <header className="topbar">
        <button className="brand" onClick={() => setMobileTab('preview')} aria-label="PanelTone 首页">
          <span className="brand-mark">PT</span>
          <span><strong>PanelTone</strong><small>本地漫画工作台</small></span>
        </button>
        <div className="topbar-left-tools">
          <button className="icon-button panel-toggle" onClick={() => setLeftCollapsed(value => !value)} aria-label={leftCollapsed ? '展开书库' : '折叠书库'} title={`${leftCollapsed ? '展开' : '折叠'}书库 · Ctrl+Shift+B`}><Icon name={leftCollapsed ? 'chevron-right' : 'chevron-left'} /></button>
        </div>
        <div className="top-status" aria-live="polite">
          <span className={`health-dot ${modelReady ? 'ready' : ''}`} />
          <span>{modelStatusText}</span>
          <span className="divider" />
          <span>{jobs.filter(job => job.status === 'running').length} 个处理中</span>
          <span>{jobs.filter(job => ['queued', 'waiting_model'].includes(job.status)).length} 个等待</span>
        </div>
        <div className="top-actions">
          <button className="button primary" onClick={() => setImportOpen(true)}><Icon name="add" />导入漫画</button>
          <button className="plain-button text-size-button" onClick={() => setLargeText(value => !value)} aria-label={largeText ? '切换标准字' : '切换大字'} title={largeText ? '切换标准字' : '切换大字'}>{largeText ? '大字' : '标准字'}</button>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="模型与设置"><Icon name="settings" /></button>
          <button className="icon-button panel-toggle" onClick={() => setRightCollapsed(value => !value)} aria-label={rightCollapsed ? '展开设置' : '折叠设置'} title={`${rightCollapsed ? '展开' : '折叠'}设置 · Ctrl+Shift+I`}><Icon name={rightCollapsed ? 'chevron-left' : 'chevron-right'} /></button>
        </div>
      </header>

      <main className={`workspace mobile-${mobileTab} ${leftCollapsed ? 'left-collapsed' : ''} ${rightCollapsed ? 'right-collapsed' : ''}`}>
        <aside className="library-panel">
          <button className="rail-toggle left-toggle" onClick={() => setLeftCollapsed(value => !value)} aria-label={leftCollapsed ? '展开书库' : '折叠书库'} title={`${leftCollapsed ? '展开' : '折叠'}书库 · Ctrl+Shift+B`} aria-expanded={!leftCollapsed}><Icon name={leftCollapsed ? 'chevron-right' : 'chevron-left'} /></button>
          <button className="rail-action trash-rail" onClick={() => { setFilter('trash'); setLeftCollapsed(false) }} aria-label="打开回收站" title="回收站"><span aria-hidden="true">♻</span><small>{jobs.filter(job => job.status === 'archived').length}</small></button>
          <div className="panel-heading">
            <div><span className="kicker">书库</span><h1>处理任务</h1></div>
            <span className="count">{jobs.length}</span>
          </div>
          <div className="filter-row" role="tablist" aria-label="筛选任务">
            {[['all', '全部'], ['active', '进行中'], ['attention', '需处理'], ['done', '完成'], ['trash', '回收站']].map(([id, label]) =>
              <button key={id} className={filter === id ? 'active' : ''} onClick={() => setFilter(id)}>{label}</button>)}
          </div>
          {selectedJobs.size > 0 && <div className="batch-bar" aria-label="批量任务操作"><strong>{selectedJobs.size} 项</strong><button onClick={() => batchAction('start')}>开始</button><button onClick={() => batchAction('pause')}>暂停</button><button onClick={() => batchAction('cancel')}>取消</button><button onClick={() => batchAction('archive')}>归档</button></div>}
          <div className="job-list" onClick={() => libraryContext && setLibraryContext(null)} onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }} onDrop={event => { event.preventDefault(); if (draggedFolderId) moveLibraryFolder(draggedFolderId, null); else if (draggedLibraryId) moveLibraryJob(draggedLibraryId, null) }}>
            {renderLibraryItems()}
            {!visibleJobs.length && <div className="empty-list">当前筛选下没有任务</div>}
          </div>
          <button className="new-folder" onClick={() => libraryFolderAction('create')}>＋新建目录</button>
          <button className="new-book" onClick={() => setImportOpen(true)}><Icon name="add" />导入新漫画</button>
        </aside>

        <section className="preview-panel">
          {selected ? <>
            <div className="preview-toolbar">
              <div className="title-block"><span className="kicker">当前书籍</span><h2>{selected.display_name}</h2></div>
              <div className="toolbar-actions" ref={taskMenuRef} role="toolbar" aria-label="预览工具">
                <div className="segmented">
                  {([['compare', '对比'], ['source', '原图'], ['final', '结果'], ['mask', '遮罩']] as const).map(([id, label]) =>
                    <button key={id} className={previewMode === id ? 'active' : ''} onClick={() => setPreviewMode(id)}>{label}</button>)}
                </div>
                <button className="plain-button" onClick={() => setPreviewDark(value => !value)}>{previewDark ? '白底' : '深色底'}</button>
                <div className="segmented zoom-controls" aria-label="缩放">
                  {(['fit', '50', '100', '200'] as const).map(value => <button key={value} className={zoom === value && ((value === 'fit' && canvasScale === 1) || value !== 'fit') ? 'active' : ''} onClick={() => setZoomPreset(value)}>{value === 'fit' ? '适配' : `${value}%`}</button>)}
                  <span className="zoom-readout" aria-live="polite">{Math.round(canvasScale * 100)}%</span>
                </div>
                <button className="more-button" onClick={() => setTaskMenuOpen(value => !value)} aria-label="任务操作" title="任务操作" aria-expanded={taskMenuOpen}><Icon name="more" /></button>
                {taskMenuOpen && <div className="task-menu" role="menu">
                  {([
                    ['运行', ['start', 'pause', 'cancel']],
                    ['结果', ['retry', 'repair-results', 'download']],
                    ['书籍', ['duplicate', 'rename']],
                    ['回收站', ['archive', 'restore']],
                  ] as const).map(([group, names]) => <div className="task-menu-section" key={group}>
                    <h4>{group}</h4>
                    {names.map(name => {
                      const label = name === 'start' ? '开始或继续' : name === 'pause' ? '暂停' : name === 'cancel' ? '取消' : name === 'retry' ? '重试失败页' : name === 'repair-results' ? '修复已生成结果' : name === 'duplicate' ? '复制并调整' : name === 'rename' ? '修改书名' : name === 'archive' ? '移到回收站' : name === 'restore' ? '从回收站恢复' : '下载成品'
                      const disabled = name === 'start' ? ['running', 'queued', 'completed', 'archived', 'ingesting', 'created'].includes(selected.status) : name === 'pause' || name === 'cancel' ? !['running', 'queued', 'waiting_model'].includes(selected.status) : name === 'retry' ? !['failed', 'needs_attention'].includes(selected.status) : name === 'repair-results' ? ['running', 'queued', 'ingesting', 'created', 'archived'].includes(selected.status) || selected.progress.completed_units === 0 : name === 'archive' ? ['created', 'ingesting', 'running', 'queued', 'waiting_model', 'archived'].includes(selected.status) : name === 'restore' ? selected.status !== 'archived' : name === 'download' ? selected.status !== 'completed' : false
                      const reason = disabled ? name === 'download' ? '整本完成后可下载' : name === 'retry' ? '没有可重试的失败页' : name === 'repair-results' ? '任务完成或暂停后才可修复' : name === 'restore' ? '任务不在回收站' : name === 'archive' ? '准备或运行中的任务不能归档' : name === 'start' && selected.status === 'ingesting' ? '建书完成后会自动开始处理' : '当前阶段不可用' : undefined
                      return <button key={name} role="menuitem" title={reason} disabled={disabled} onClick={() => { setTaskMenuOpen(false); if (name === 'retry') { const failed = selected.progress.page_states?.find(page => ['failed', 'qa_failed'].includes(page.status)); if (failed) retryPage(selected.id, failed.page_index).catch(error => setMessage(error.message)) } else if (name === 'duplicate') action(selected.id, 'duplicate'); else if (name === 'rename') setRenameTarget(selected); else action(selected.id, name) }}>{label}{reason && <small>{reason}</small>}</button>
                    })}
                  </div>)}
                </div>}
              </div>
            </div>
            <div className={`canvas canvas-gesture-zone ${previewDark ? 'dark' : ''} zoom-${zoom}`} onWheel={handleCanvasWheel} onPointerDown={handleCanvasPointerDown} onPointerMove={handleCanvasPointerMove} onPointerUp={handleCanvasPointerUp} onPointerCancel={handleCanvasPointerUp}>
              {currentPage ? <>
                {previewMode === 'compare' && currentResultUrl ? <div className="compare-stage">
                  <img style={{ transform: `translate3d(${canvasPan.x}px, ${canvasPan.y}px, 0) scale(${canvasScale})` }} src={currentSourceUrl || currentPage.source_url} alt={`第 ${currentPage.page_index + 1} 页原图`} onLoad={markPageImageLoaded} onError={() => setPageLoading(false)} />
                  <div className="compare-result" style={{ clipPath: `inset(0 0 0 ${compare}%)`, transform: `translate3d(${canvasPan.x}px, ${canvasPan.y}px, 0) scale(${canvasScale})` }}><img src={currentResultUrl} alt={`第 ${currentPage.page_index + 1} 页结果`} onLoad={markPageImageLoaded} onError={() => setPageLoading(false)} /></div>
                  <div className="compare-line" style={{ left: `${compare}%` }}><span /></div>
                  <input className="compare-range" aria-label="拖动比较原图和结果" type="range" min="0" max="100" value={compare} onChange={event => setCompare(Number(event.target.value))} />
                </div> : previewMode === 'source' ? <img className="single-page" style={{ transform: `translate3d(${canvasPan.x}px, ${canvasPan.y}px, 0) scale(${canvasScale})` }} src={currentSourceUrl || currentPage.source_url} alt={`第 ${currentPage.page_index + 1} 页原图`} onLoad={markPageImageLoaded} onError={() => setPageLoading(false)} /> : previewMode === 'mask' ? <img className="single-page mask-page" style={{ transform: `translate3d(${canvasPan.x}px, ${canvasPan.y}px, 0) scale(${canvasScale})` }} src={`/api/jobs/${selected.id}/pages/${currentPage.page_index}/mask`} alt={`第 ${currentPage.page_index + 1} 页保护遮罩`} onLoad={markPageImageLoaded} onError={() => setPageLoading(false)} /> : currentResultUrl ? <img className="single-page" style={{ transform: `translate3d(${canvasPan.x}px, ${canvasPan.y}px, 0) scale(${canvasScale})` }} src={currentResultUrl} alt={`第 ${currentPage.page_index + 1} 页结果`} onLoad={markPageImageLoaded} onError={() => setPageLoading(false)} /> : waitingForModel ? <div className="waiting-page model-waiting"><span className="waiting-symbol">!</span><strong>模型服务未连接</strong><span>当前没有页面在处理，服务恢复后会自动继续</span>{selected.error && <small className="waiting-detail">{selected.error}</small>}<button onClick={() => refreshHealth()}>重新检测模型</button></div> : <div className="waiting-page"><div className="spinner" /><strong>{modelLoading ? '正在加载模型权重' : waitingPageTitle}</strong><span>{modelLoading ? '首次启动需要准备模型，完成后才会生成第 1 页' : waitingPageHint}</span>{modelLoading && <div className="model-stage-progress">{typeof modelHealth?.progress === 'number' ? <><progress max="100" value={modelHealth.progress} /><strong>{modelHealth.progress.toFixed(0)}%</strong></> : <small>{modelHealth?.stage || modelHealth?.message || '当前服务只提供阶段状态，未提供可靠总量'}</small>}</div>}</div>}
                <button className="canvas-page-nav previous" onClick={() => movePage(-1)} disabled={pages.findIndex(page => page.page_index === pageIndex) <= 0} aria-label="上一页" title="上一页"><Icon name="chevron-left" /></button>
                <button className="canvas-page-nav next" onClick={() => movePage(1)} disabled={pages.findIndex(page => page.page_index === pageIndex) < 0 || pages.findIndex(page => page.page_index === pageIndex) >= pages.length - 1} aria-label="下一页" title="下一页"><Icon name="chevron-right" /></button>
                <span className="canvas-page-count" aria-label="当前页码">{currentPage.page_index + 1} / {pages.length}</span>
                {pageLoading && <div className="page-loading" role="status"><span className="spinner" />正在载入第 {currentPage.page_index + 1} 页</div>}
              </> : <div className="waiting-page"><strong>正在准备页面</strong><span>页面展开后会显示缩略图</span></div>}
            </div>
            {readyNotice != null && <div className="page-ready-notice" role="status" aria-live="polite"><span>第 {readyNotice + 1} 页已完成，可以预览</span><button onClick={() => { manualPageSelectionRef.current.add(selected.id); pageSelectionRef.current.set(selected.id, readyNotice); setPageIndex(readyNotice); setReadyNotice(null) }}>查看这一页</button><button className="notice-dismiss" aria-label="关闭新页面提示" onClick={() => setReadyNotice(null)}>×</button></div>}
            <div className="filmstrip filmstrip-shell" aria-label="漫画页面">
              <div className="filmstrip-viewport">
                <div className="filmstrip-pages">
                  {pages.map(page => <button key={page.page_index} ref={page.page_index === pageIndex ? activeFilmstripPageRef : undefined} className={page.page_index === pageIndex ? 'active' : ''} onClick={() => selectPage(page.page_index)}>
                    <img loading="lazy" decoding="async" src={page.thumbnail_url || page.source_url} alt={`第 ${page.page_index + 1} 页`} /><span>{page.page_index + 1}</span>{page.final_url ? <i className="page-state-mark completed" aria-label="已完成"><Icon name="check" /></i> : page.preview_url ? <i className="page-state-mark processing" title="已有预览，尚未通过整页检查" aria-label="处理中">◌</i> : page.status === 'failed' ? <i className="page-state-mark failed" aria-label="失败" /> : null}
                  </button>)}
                </div>
              </div>
            </div>
          </> : <div className="empty-workspace"><span className="empty-mark">PT</span><h2>从一本漫画开始</h2><p>拖入图片、PDF、ZIP、CBZ、RAR 或 CBR</p><button className="button primary" onClick={() => setImportOpen(true)}><Icon name="upload" />选择漫画</button></div>}
        </section>

        <Inspector selected={selected} presets={presets} health={health} currentPage={currentPage} collapsed={rightCollapsed} onToggle={() => setRightCollapsed(value => !value)} onAction={action} onRename={() => selected && setRenameTarget(selected)} onDelete={() => selected && setDeleteTarget(selected)} onRetry={retryPage} />
      </main>

      {selected && <section className={`progress-drawer ${progressOpen ? 'open' : ''}`}>
        <button className="progress-summary" onClick={() => setProgressOpen(value => !value)} aria-expanded={progressOpen}>
          <span className={`status-shape ${selected.status}`} />
          <strong>{statusLabel[selected.status] || selected.status}</strong>
          <span>{selected.progress.completed_pages}/{selected.progress.total_pages || selected.page_count} 页</span>
          <progress max="100" value={effectiveProgress}>{effectiveProgress}%</progress>
          <b>{effectiveProgress.toFixed(0)}%</b>
          <span>{selected.progress.latest_message || (waitingForModel ? '模型就绪后开始' : modelLoading ? '正在加载模型' : formatEta(selected.progress.eta_seconds))}</span>
          <span className="drawer-toggle">{progressOpen ? '收起' : '详情'}</span>
        </button>
        {progressOpen && <div className="progress-content">
          <div className="progress-main">
            <div className="progress-details" aria-live="polite">
              <div><small>当前阶段</small><strong>{progressStageLabel(selected)}</strong></div>
              <div><small>当前页面</small><strong>{selected.status === 'completed' ? `全部 ${selected.progress.total_pages} 页` : selected.progress.current_page != null ? `${selected.progress.current_page + 1} / ${selected.progress.total_pages || selected.page_count}` : '等待开始'}</strong></div>
              <div><small>处理单元</small><strong>{selected.progress.completed_units} / {selected.progress.total_units || '—'}</strong></div>
              <div><small>失败单元</small><strong>{selected.progress.failed_units}</strong></div>
              <div><small>处理速度</small><strong>{selected.progress.seconds_per_megapixel == null ? '正在估算' : `${selected.progress.seconds_per_megapixel.toFixed(1)} 秒 / 百万像素`}</strong></div>
              <div><small>预计剩余</small><strong>{formatEta(selected.progress.eta_seconds)}</strong></div>
              <div><small>建书读取</small><strong>{selected.progress.bytes_total ? `${formatBytes(selected.progress.bytes_processed || 0)} / ${formatBytes(selected.progress.bytes_total)}` : '已完成'}</strong></div>
              <div className="progress-actions">{selected.status === 'completed' ? <button onClick={() => action(selected.id, 'download')}>下载成品</button> : selected.status === 'archived' ? <span>已在回收站</span> : <><button onClick={() => action(selected.id, selected.status === 'running' || selected.status === 'queued' ? 'pause' : 'start')}><Icon name={selected.status === 'running' || selected.status === 'queued' ? 'pause' : 'play'} />{selected.status === 'running' || selected.status === 'queued' ? '暂停' : '运行或继续'}</button><button onClick={() => action(selected.id, 'cancel')}>取消</button></>}</div>
            </div>
            <div className="page-status-grid" aria-label="每页状态">
              {(selected.progress.page_states || []).map(page => <button key={page.page_index} className={`page-state ${page.status} ${page.page_index === pageIndex ? 'current' : ''}`} title={page.error || `第 ${page.page_index + 1} 页`} aria-label={`第 ${page.page_index + 1} 页 ${page.status === 'qa_passed' ? '完成' : page.status === 'running' ? '处理中' : page.status === 'failed' || page.status === 'qa_failed' ? '失败' : '等待'}`} onClick={() => { selectPage(page.page_index); setMobileTab('preview') }}><span className={`page-state-mark ${page.status === 'qa_passed' ? 'completed' : page.status === 'running' ? 'processing' : page.status === 'failed' || page.status === 'qa_failed' ? 'failed' : 'pending'}`} aria-hidden="true" /></button>)}
            </div>
          </div>
          <section className="activity-log" aria-label="后台日志">
            <header>
              <div className="log-tabs">{(['activity', 'raw', 'gpu'] as const).map(kind => <button key={kind} className={logKind === kind ? 'active' : ''} onClick={() => setLogKind(kind)}>{kind === 'activity' ? '活动' : kind === 'raw' ? '原始服务' : 'GPU'}</button>)}</div>
              <span>{logKind === 'activity' ? (logs.length ? '实时更新' : '暂无记录') : `${visibleRawLogs.length} / ${rawLogs.length} 条`}</span>
            </header>
            {logKind !== 'activity' && <div className="log-filters">
              <label><span>级别</span><select value={logLevel} onChange={event => setLogLevel(event.target.value)} aria-label="日志级别"><option value="all">全部</option><option value="INFO">INFO</option><option value="WARNING">WARNING</option><option value="ERROR">ERROR</option></select></label>
              <label><span>组件</span><select value={logComponent} onChange={event => setLogComponent(event.target.value)} aria-label="日志组件"><option value="all">全部</option>{logComponents.map(component => <option key={component} value={component}>{component}</option>)}</select></label>
              <label className="log-check"><input type="checkbox" checked={autoScrollLogs} onChange={event => setAutoScrollLogs(event.target.checked)} />自动滚动</label>
              <button onClick={copyLogs}>复制</button>
            </div>}
            <div ref={logListRef}>{logKind === 'activity' ? logs.slice(-12).reverse().map(item => <p key={item.id}><time>{formatLogTime(item.created_at)}</time><span>{item.message}</span></p>) : logKind === 'gpu' ? visibleRawLogs.slice(-12).reverse().map(item => <p key={`${item.id}-${item.timestamp}`}><time>{formatLogTime(item.timestamp)}</time><span>{item.message}{item.metrics?.utilization_percent != null ? ` · GPU ${item.metrics.utilization_percent}%` : ''}</span></p>) : visibleRawLogs.slice(-12).reverse().map(item => <p key={`${item.id}-${item.timestamp}`}><time>{formatLogTime(item.timestamp)}</time><span>[{item.level}] {item.component} · {item.message}</span></p>)}</div>
            <footer><button onClick={() => { if (!window.confirm('原始日志可能包含本机路径，请确认后再分享')) return; window.location.href = `/api/logs/download?kind=${logKind === 'activity' ? 'raw' : logKind}${selected && logKind !== 'gpu' ? `&job_id=${selected.id}` : ''}` }}>下载日志</button>{gpuMetrics && <span>{gpuMetrics.available ? `${gpuMetrics.name || 'GPU'} · ${gpuMetrics.utilization_percent ?? '—'}% · ${gpuMetrics.memory_used_mib ?? '—'} / ${gpuMetrics.memory_total_mib ?? '—'} MiB` : `GPU 不可用：${gpuMetrics.reason || '未提供原因'}`}</span>}</footer>
          </section>
        </div>}
      </section>}

      {libraryContext && <div ref={libraryContextRef} className="library-context-menu" role="menu" style={{ left: Math.min(libraryContext.x, window.innerWidth - 230), top: Math.min(libraryContext.y, window.innerHeight - 240) }} onContextMenu={event => event.preventDefault()}>
        {libraryContext.type === 'folder' ? (() => {
          const folder = libraryTree ? findFolder(libraryTree.folders, libraryContext.id) : undefined
          if (!folder) return null
          return <>
            <strong className="context-title">目录：{folder.name}</strong>
            {!folder.archived_at && <button role="menuitem" onClick={() => libraryFolderAction('create', folder)}>新建子目录</button>}
            {!folder.archived_at && <button role="menuitem" onClick={() => libraryFolderAction('rename', folder)}>重命名</button>}
            {folder.archived_at ? <button role="menuitem" onClick={() => libraryFolderAction('restore', folder)}>恢复目录</button> : <button role="menuitem" onClick={() => libraryFolderAction('archive', folder)}>移到回收站</button>}
            {folder.archived_at && <button role="menuitem" onClick={() => libraryFolderAction('delete', folder)}>永久删除</button>}
            {!folder.archived_at && <><div className="context-group-label">同级排序</div><button role="menuitem" onClick={() => { nudgeLibraryNode('folder', folder.id, -1); setLibraryContext(null) }}>上移目录</button><button role="menuitem" onClick={() => { nudgeLibraryNode('folder', folder.id, 1); setLibraryContext(null) }}>下移目录</button></>}
          </>
        })() : (() => {
          const job = jobs.find(item => item.id === libraryContext.id)
          if (!job) return null
          const folders = libraryTree ? flattenFolders(libraryTree.folders) : []
          const canArchive = !['created', 'ingesting', 'running', 'queued', 'waiting_model', 'archived'].includes(job.status)
          return <>
            <strong className="context-title">任务：{job.display_name}</strong>
            <div className="context-group-label">运行与结果</div>
            <button role="menuitem" disabled={['running', 'queued', 'completed', 'archived', 'ingesting', 'created'].includes(job.status)} onClick={() => libraryJobAction(job.id, 'start')}>开始或继续</button>
            <button role="menuitem" disabled={!['failed', 'needs_attention'].includes(job.status)} onClick={() => libraryJobAction(job.id, 'retry')}>重试失败页</button>
            <button role="menuitem" onClick={() => libraryJobAction(job.id, 'rename')}>修改书名</button>
            <button role="menuitem" onClick={() => libraryJobAction(job.id, 'duplicate')}>复制并调整</button>
            <div className="context-group-label">移动到目录</div>
            <button role="menuitem" disabled={job.folder_id == null} onClick={() => { moveLibraryJob(job.id, null); setLibraryContext(null) }}>根目录</button>
            {folders.map(({ folder, depth }) => <button role="menuitem" key={folder.id} disabled={Boolean(folder.archived_at) || folder.id === job.folder_id} onClick={() => { moveLibraryJob(job.id, folder.id); setLibraryContext(null) }} style={{ paddingLeft: `${12 + depth * 12}px` }}>↳ {folder.name}</button>)}
            <div className="context-group-label">同级排序</div>
            <button role="menuitem" onClick={() => { nudgeLibraryNode('job', job.id, -1); setLibraryContext(null) }}>上移任务</button>
            <button role="menuitem" onClick={() => { nudgeLibraryNode('job', job.id, 1); setLibraryContext(null) }}>下移任务</button>
            <div className="context-group-label">回收站</div>
            {job.status === 'archived' ? <><button role="menuitem" onClick={() => libraryJobAction(job.id, 'restore')}>恢复任务</button><button role="menuitem" onClick={() => libraryJobAction(job.id, 'delete')}>永久删除</button></> : <button role="menuitem" disabled={!canArchive} onClick={() => libraryJobAction(job.id, 'archive')}>移到回收站</button>}
          </>
        })()}
      </div>}

      <nav className="mobile-nav" aria-label="移动端导航">
        {[['books', 'books', '书籍'], ['preview', 'preview', '预览'], ['settings', 'settings', '设置'], ['progress', 'progress', '进度']].map(([id, icon, label]) =>
          <button key={id} className={mobileTab === id ? 'active' : ''} onClick={() => setMobileTab(id as typeof mobileTab)}><Icon name={icon} />{label}</button>)}
      </nav>

      {importOpen && <ImportDialog presets={presets} health={health} onClose={() => setImportOpen(false)} onCreated={async (jobsCreated, createdMessage) => { setImportOpen(false); setMobileTab('preview'); if (createdMessage) setMessage(createdMessage); await refreshJobs(); await refreshLibrary(); if (jobsCreated[0]) setSelectedId(jobsCreated[0]); }} />}
      {settingsOpen && <ModelDialog onClose={() => setSettingsOpen(false)} />}
      {renameTarget && <RenameDialog job={renameTarget} onClose={() => setRenameTarget(null)} onSave={rename} />}
      {deleteTarget && <DeleteDialog job={deleteTarget} onClose={() => setDeleteTarget(null)} onDelete={permanentlyDelete} />}
      {message && <button className="toast" onClick={() => setMessage('')}>{message}<Icon name="close" /></button>}
    </div>
  )
}

function Inspector({ selected, presets, health, currentPage, collapsed, onToggle, onAction, onRename, onDelete, onRetry }: {
  selected: Job | null
  presets: Presets | null
  health: Record<string, EngineHealth>
  currentPage?: Page
  collapsed: boolean
  onToggle: () => void
  onAction: (jobId: string, name: string) => void
  onRename: () => void
  onDelete: () => void
  onRetry: (jobId: string, pageIndex: number) => void
}) {
  if (!selected) return <aside className={`inspector-panel ${collapsed ? 'is-collapsed' : ''}`}><button className="rail-toggle right-toggle" onClick={onToggle} aria-label={collapsed ? '展开设置' : '折叠设置'} title={`${collapsed ? '展开' : '折叠'}设置 · Ctrl+Shift+I`} aria-expanded={!collapsed}><Icon name={collapsed ? 'chevron-left' : 'chevron-right'} /></button><div className="inspector-empty">选择任务后查看设置</div></aside>
  const color = presets?.colors.find(item => item.id === selected.spec.color_preset)
  const style = presets?.styles.find(item => item.id === selected.spec.style_preset)
  const semanticStatus = String(currentPage?.semantic_mask?.status || currentPage?.mask_status || 'pending')
  const semanticLabel = semanticStatus === 'fallback'
    ? '基础保护模式'
    : semanticStatus === 'ready'
      ? '语义分层已缓存'
      : semanticStatus === 'pending'
        ? '页面完成后准备'
        : semanticStatus
  return <aside className={`inspector-panel ${collapsed ? 'is-collapsed' : ''}`}>
    <button className="rail-toggle right-toggle" onClick={onToggle} aria-label={collapsed ? '展开设置' : '折叠设置'} title={`${collapsed ? '展开' : '折叠'}设置 · Ctrl+Shift+I`} aria-expanded={!collapsed}><Icon name={collapsed ? 'chevron-left' : 'chevron-right'} /></button>
    <div className="panel-heading"><div><span className="kicker">任务设置</span><h2>处理方案</h2></div><span className="lock">已锁定</span></div>
    <section className="setting-section"><h3>处理方式</h3><strong>{presets?.modes.find(item => item.id === selected.spec.mode)?.name || selected.spec.mode}</strong><OptionInfo item={presets?.modes.find(item => item.id === selected.spec.mode)} /></section>
    <section className="setting-section"><h3>配色</h3><div className="swatch-row"><span className={`swatch ${selected.spec.color_preset}`} /><strong>{color?.name}</strong></div><OptionInfo item={color} /></section>
    <section className="setting-section"><h3>画风</h3><strong>{style?.name}</strong><OptionInfo item={style} /></section>
    <section className="setting-section compact"><div><small>细节保护</small><strong>{presets?.details.find(item => item.id === selected.spec.detail_mode)?.name}</strong></div><div><small>处理单位</small><strong>{presets?.panels.find(item => item.id === selected.spec.panel_mode)?.name}</strong></div><div><small>导出</small><strong>{presets?.outputs.find(item => item.id === selected.spec.output_format)?.name}</strong></div></section>
    <section className={`setting-section semantic-state ${semanticStatus === 'fallback' ? 'warning' : ''}`} aria-label="语义分层状态"><h3>语义分层</h3><strong>{semanticLabel}</strong><small>{semanticStatus === 'fallback' ? '确定性保护文字、气泡、墨线和边框；人物与场景采用完整彩色合成' : semanticStatus === 'ready' ? 'Koharu 保护文字、气泡和框线，确定性检测补充核心墨线；不虚构人体分层' : '页面资产可用后显示遮罩状态'}</small></section>
    <section className="setting-section"><h3>本地引擎</h3><div className="engine-state"><span className={`health-dot ${health[selected.spec.engine as string]?.ok ? 'ready' : ''}`} /><strong>{selected.spec.engine}</strong></div></section>
    <div className="inspector-actions">
      {selected.status === 'completed' && <button className="button primary" onClick={() => onAction(selected.id, 'download')}>下载成品</button>}
      {selected.status === 'archived' ? <>
        <button className="button secondary" onClick={() => onAction(selected.id, 'restore')}>从回收站恢复</button>
        <button className="danger-button" onClick={onDelete}>永久删除</button>
      </> : <>
        {currentPage && currentPage.status !== 'qa_passed' && ['failed', 'needs_attention'].includes(selected.status) && <button className="button primary" onClick={() => onRetry(selected.id, currentPage.page_index)}>重试当前失败页</button>}
        <button className="button secondary" onClick={() => onAction(selected.id, 'duplicate')}>复制并调整</button>
        <button className="button secondary" onClick={onRename}>修改书名</button>
        {!['running', 'queued'].includes(selected.status) && <button className="text-button" onClick={() => onAction(selected.id, 'archive')}>移到回收站</button>}
      </>}
    </div>
  </aside>
}

function RenameDialog({ job, onClose, onSave }: {
  job: Job
  onClose: () => void
  onSave: (jobId: string, displayName: string) => void
}) {
  const [name, setName] = useState(job.display_name)
  const [error, setError] = useState('')
  async function save() {
    try { await onSave(job.id, name.trim()) } catch (reason) { setError((reason as Error).message) }
  }
  return <div className="dialog-backdrop"><div className="dialog compact-dialog" role="dialog" aria-modal="true" aria-labelledby="rename-title"><header><div><span className="kicker">书籍信息</span><h2 id="rename-title">修改书名</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></header><div className="dialog-body"><label className="field"><span>新书名</span><input autoFocus value={name} maxLength={120} onChange={event => setName(event.target.value)} onKeyDown={event => event.key === 'Enter' && save()} /></label><p className="help-copy">只修改界面显示名称，不会重新处理页面</p></div><footer><span className="error-text">{error}</span><button className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={!name.trim()} onClick={save}>保存</button></footer></div></div>
}

function DeleteDialog({ job, onClose, onDelete }: {
  job: Job
  onClose: () => void
  onDelete: (jobId: string) => void
}) {
  const [confirmation, setConfirmation] = useState('')
  const [error, setError] = useState('')
  async function remove() {
    try { await onDelete(job.id) } catch (reason) { setError((reason as Error).message) }
  }
  return <div className="dialog-backdrop"><div className="dialog compact-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-title"><header><div><span className="kicker">不可恢复</span><h2 id="delete-title">永久删除“{job.display_name}”</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></header><div className="dialog-body"><p className="warning-copy">任务数据库、页面和输出会从本机删除，无法从回收站恢复</p><label className="field"><span>输入“永久删除”以确认</span><input autoFocus value={confirmation} onChange={event => setConfirmation(event.target.value)} /></label></div><footer><span className="error-text">{error}</span><button className="button secondary" onClick={onClose}>取消</button><button className="danger-button" disabled={confirmation !== '永久删除'} onClick={remove}>永久删除</button></footer></div></div>
}

function ImportDialog({ presets, health, onClose, onCreated }: { presets: Presets | null; health: Record<string, { ok: boolean }>; onClose: () => void; onCreated: (jobIds: string[], message?: string) => void }) {
  const [files, setFiles] = useState<File[]>([])
  const [uploads, setUploads] = useState<Uploaded[]>([])
  const [name, setName] = useState('图片合集')
  const [localPath, setLocalPath] = useState('')
  const [profile, setProfile] = useState('quality')
  const [advanced, setAdvanced] = useState(false)
  const [mode, setMode] = useState('colorize')
  const [color, setColor] = useState('natural')
  const [style, setStyle] = useState('original_ink')
  const [detail, setDetail] = useState('strict')
  const [panel, setPanel] = useState('page')
  const [output, setOutput] = useState('cbz')
  const [engine, setEngine] = useState(Object.keys(health).find(id => id !== 'palette') || 'palette')
  const [adult, setAdult] = useState(false)
  const [busy, setBusy] = useState(false)
  const [phase, setPhase] = useState('准备提交')
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const directoryInput = useRef<HTMLInputElement>(null)
  const naturalNames = useMemo(() => new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' }), [])
  const uploadIds = useRef(new Map<string, string>())
  const uploadResults = useRef(new Map<string, Uploaded>())
  const uploadPercent = files.length
    ? Math.round(files.reduce((total, _file, index) => total + (uploads[index]?.progress || 0), 0) / files.length)
    : 0

  useEffect(() => {
    const warnWhileUploading = (event: BeforeUnloadEvent) => {
      if (!busy || !phase.startsWith('正在上传')) return
      event.preventDefault()
      event.returnValue = '文件仍在上传，离开会中断上传'
    }
    window.addEventListener('beforeunload', warnWhileUploading)
    return () => window.removeEventListener('beforeunload', warnWhileUploading)
  }, [busy, phase])

  function fileKey(file: File) {
    return `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`
  }

  function addFiles(list: FileList | File[]) {
    const incoming = Array.from(list)
    setFiles(current => [...current, ...incoming.filter(file => !current.some(item => fileKey(item) === fileKey(file)))].sort((left, right) => naturalNames.compare(left.webkitRelativePath || left.name, right.webkitRelativePath || right.name)))
  }
  function drop(event: DragEvent) { event.preventDefault(); addFiles(event.dataTransfer.files) }
  function move(index: number, offset: number) { setFiles(current => { const copy = [...current]; const next = index + offset; if (next < 0 || next >= copy.length) return current; [copy[index], copy[next]] = [copy[next], copy[index]]; return copy }) }

  async function create() {
    if (!files.length && !localPath.trim()) return setError('请先选择漫画文件或填写本地路径')
    if (files.length && localPath.trim()) return setError('文件上传和本地路径请分开建立任务')
    setBusy(true); setError(''); setPhase('正在准备来源')
    try {
      const common = { mode, color_preset: color, style_preset: style, detail_mode: detail, panel_mode: panel, output_format: output, engine, max_retries: profile === 'fast' ? 0 : profile === 'repair' ? 4 : 2, adult_fictional_content: adult, preserve_text: true, preserve_ink: detail !== 'generative' }
      if (localPath.trim()) {
        setPhase('已提交，正在展开页面')
        const response = await api<{ job_id: string }>('/api/jobs?async=true', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...common, source: localPath.trim(), display_name: name }) })
        return onCreated([response.job_id], '任务已提交，正在展开页面')
      }
      const uploaded: Uploaded[] = []
      for (let index = 0; index < files.length; index++) {
        setPhase(`正在上传 ${index + 1} / ${files.length}`)
        const key = fileKey(files[index])
        const existing = uploadResults.current.get(key)
        if (existing?.source_id) { uploaded.push(existing); continue }
        const clientUploadId = uploadIds.current.get(key) || crypto.randomUUID().replaceAll('-', '')
        uploadIds.current.set(key, clientUploadId)
        const item = await uploadFile(files[index], clientUploadId, value => setUploads(current => {
          const copy = [...current]
          copy[index] = { source_id: '', name: files[index].name, kind: 'image', size: files[index].size, duplicate: false, progress: value }
          return copy
        }))
         uploadResults.current.set(key, item)
         uploaded.push(item)
         setUploads(current => {
           const copy = [...current]
           copy[index] = item
           return copy
         })
      }
      const uniqueUploaded = uploaded.filter((item, index) => uploaded.findIndex(candidate => candidate.source_id === item.source_id) === index)
      setPhase('已提交，正在建立页面')
      const response = await api<{ jobs: { job_id: string }[] }>('/api/jobs/batch?async=true', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...common, source_ids: uniqueUploaded.map(item => item.source_id), image_order: uniqueUploaded.filter(item => item.kind === 'image').map(item => item.source_id), image_book_name: name }) })
      onCreated(response.jobs.map(job => job.job_id), '任务已提交，正在建立页面')
    } catch (reason) {
      setError((reason as Error).message)
      setPhase(files.length ? '上传或提交失败，可点击“重试”继续' : '提交失败，可点击“重试”继续')
    } finally { setBusy(false) }
  }

  const chosenMode = presets?.modes.find(item => item.id === mode)
  return <div className="dialog-backdrop" role="presentation"><div className="dialog import-dialog" role="dialog" aria-modal="true" aria-labelledby="import-title">
    <header><div><span className="kicker">新任务</span><h2 id="import-title">导入漫画</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></header>
    <div className="dialog-grid">
      <section className="source-step">
        <button className="dropzone" onClick={() => input.current?.click()} onDrop={drop} onDragOver={event => event.preventDefault()}><Icon name="upload" /><strong>拖入图片或漫画包</strong><span>支持多张图片、PDF、ZIP、CBZ、RAR 和 CBR</span></button>
        <input ref={input} hidden type="file" multiple accept=".png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,.pdf,.zip,.cbz,.rar,.cbr" onChange={(event: ChangeEvent<HTMLInputElement>) => event.target.files && addFiles(event.target.files)} />
        <input ref={directoryInput} hidden type="file" multiple {...({ webkitdirectory: '', directory: '' } as Record<string, string>)} onChange={(event: ChangeEvent<HTMLInputElement>) => event.target.files && addFiles(event.target.files)} />
        <div className="source-actions"><button className="plain-button" onClick={() => directoryInput.current?.click()}>选择图片文件夹</button><span>或</span><button className="plain-button" onClick={() => input.current?.click()}>选择多个文件</button></div>
        {files.length > 0 && <div className="source-list">
          <div className="source-list-heading"><strong>{files.length} 个文件</strong><span>图片将合成一本，漫画包各自成书</span></div>
          {files.map((file, index) => { const upload = uploadResults.current.get(fileKey(file)); return <div className="source-row" key={`${file.webkitRelativePath || file.name}-${file.size}-${file.lastModified}`}><span className="file-type">{file.name.split('.').pop()?.toUpperCase()}</span><span><strong>{file.name}</strong><small>{formatBytes(file.size)}{upload ? ` · ${upload.progress}%${upload.duplicate ? ' · 已跳过重复来源' : ''}` : ''}</small></span><div><button disabled={busy} onClick={() => move(index, -1)} aria-label="上移">↑</button><button disabled={busy} onClick={() => move(index, 1)} aria-label="下移">↓</button><button disabled={busy} onClick={() => { setFiles(current => current.filter((_, item) => item !== index)); setUploads(current => current.filter((_, item) => item !== index)) }} aria-label="移除">×</button></div></div> })}
        </div>}
        <label className="field"><span>图片合集书名</span><input value={name} onChange={event => setName(event.target.value)} /></label>
        <details className="local-path"><summary>从本机路径直接导入</summary><label className="field"><span>文件或文件夹路径</span><input value={localPath} placeholder="漫画包或图片文件夹的完整路径" onChange={event => setLocalPath(event.target.value)} /></label><p>路径只发送给本机服务，不会显示在任务接口中</p></details>
      </section>
      <section className="option-step">
        <div className="choice-group"><h3>处理档位</h3><div className="choice-cards profiles">{[['fast', '速度优先', '较少重试，适合整本预览'], ['quality', '质量优先', '严格保护并检查每一页'], ['repair', '高级修复', '用于失败页和复杂人物细节']].map(([id, label, help]) => <button className={profile === id ? 'selected' : ''} onClick={() => setProfile(id)} key={id}><strong>{label}</strong><small>{help}</small></button>)}</div></div>
        <label className="field"><span>处理方式</span><select value={mode} onChange={event => setMode(event.target.value)}>{presets?.modes.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><OptionInfo item={chosenMode} />
        <div className="choice-group"><h3>配色</h3><div className="choice-cards">{presets?.colors.map(item => <button className={color === item.id ? 'selected' : ''} onClick={() => setColor(item.id)} key={item.id}><span className={`swatch ${item.id}`} /><strong>{item.name}</strong><small>{item.description}</small></button>)}</div></div>
        <div className="choice-group"><h3>画风</h3><div className="choice-cards styles">{presets?.styles.map(item => <button className={style === item.id ? 'selected' : ''} onClick={() => setStyle(item.id)} key={item.id}><strong>{item.name}</strong><small>{item.description}</small></button>)}</div></div>
        <div className="field-grid"><div><label className="field"><span>细节保护</span><select value={detail} onChange={event => setDetail(event.target.value)}>{presets?.details.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><OptionInfo item={presets?.details.find(item => item.id === detail)} /></div><div><label className="field"><span>处理单位</span><select value={panel} onChange={event => setPanel(event.target.value)}>{presets?.panels.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><OptionInfo item={presets?.panels.find(item => item.id === panel)} /></div><div><label className="field"><span>导出格式</span><select value={output} onChange={event => setOutput(event.target.value)}>{presets?.outputs.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><OptionInfo item={presets?.outputs.find(item => item.id === output)} /></div></div>
        <button className="advanced-toggle" onClick={() => setAdvanced(value => !value)} aria-expanded={advanced}>{advanced ? '收起高级设置' : '显示高级设置'}</button>
        {advanced && <label className="field"><span>本地引擎</span><select value={engine} onChange={event => setEngine(event.target.value)}>{Object.entries(health).map(([id, state]) => <option key={id} value={id} disabled={!state.ok}>{id}{state.ok ? '' : ' 未就绪'}</option>)}</select><small>普通使用无需修改；这里只显示本机已经配置的引擎</small></label>}
        <label className="check-field"><input type="checkbox" checked={adult} onChange={event => setAdult(event.target.checked)} /><span><strong>内容为已获授权的成年虚构作品</strong><small>只处理你有权使用的合法内容</small></span></label>
      </section>
    </div>
     {busy && <div className="upload-status" role="status"><div><strong>{phase}</strong>{files.length > 0 && <span>{uploadPercent}%</span>}</div>{files.length > 0 && <progress max="100" value={uploadPercent}>{uploadPercent}%</progress>}</div>}
     <footer><span className="error-text" aria-live="polite">{error || (busy ? phase : '')}</span>{error && !busy && <button className="button secondary" onClick={create}>重试</button>}<button className="button secondary" disabled={busy} onClick={onClose}>取消</button><button className="button primary" disabled={busy || (!files.length && !localPath.trim())} onClick={create}>{busy ? phase : localPath.trim() ? '提交任务并返回处理页' : `建立 ${files.filter(file => !/\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name)).length + (files.some(file => /\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name)) ? 1 : 0)} 本并开始`}</button></footer>
  </div></div>
}

function ModelDialog({ onClose }: { onClose: () => void }) {
  const [models, setModels] = useState<Array<Record<string, string | boolean>>>([])
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  async function refresh() {
    try {
      setModels(await api<Array<Record<string, string | boolean>>>('/api/models'))
    } catch (reason) {
      setMessage((reason as Error).message)
    }
  }
  useEffect(() => { refresh() }, [])
  async function download(id: string) {
    setLoading(true)
    setMessage('正在启动模型下载')
    try {
      const result = await api<{ status: string }>(`/api/models/${id}/download`, { method: 'POST' })
      setMessage(result.status === 'already_downloading' ? '模型已经在后台下载' : '下载已在后台开始')
      await refresh()
    } catch (reason) {
      setMessage((reason as Error).message)
    } finally {
      setLoading(false)
    }
  }
  async function release(id: string) {
    setLoading(true)
    setMessage('正在请求释放空闲显存')
    try {
      await api(`/api/models/${id}/release`, { method: 'POST' })
      setMessage('空闲模型已释放；再次处理时会自动加载')
      await refresh()
    } catch (reason) {
      setMessage((reason as Error).message)
    } finally {
      setLoading(false)
    }
  }
  return <div className="dialog-backdrop"><div className="dialog model-dialog" role="dialog" aria-modal="true"><header><div><span className="kicker">设置</span><h2>模型与本地存储</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></header><div className="model-list">{models.map(model => { const supportsRelease = model.supports_release === true; return <article key={String(model.id)}><div><span className={`health-dot ${model.connected || model.installed ? 'ready' : ''}`} /><div><strong>{String(model.name)}</strong><small>{String(model.purpose)}</small></div></div><dl><dt>来源</dt><dd>{String(model.repository)}</dd><dt>许可证</dt><dd>{String(model.license_url) ? <a href={String(model.license_url)} target="_blank" rel="noreferrer">{String(model.license)}</a> : String(model.license)}</dd><dt>显存提示</dt><dd>{String(model.memory)}</dd><dt>下载大小</dt><dd>{String(model.download_size)}</dd><dt>保存位置</dt><dd>{String(model.storage)}</dd><dt>状态</dt><dd>{model.connected ? '本地模型服务已连接' : model.status === 'downloading' ? '正在后台下载' : model.installed ? '权重已下载，等待服务连接' : model.downloadable === false ? String(model.unavailable_reason || '当前只提供模型插槽') : '尚未下载'}</dd></dl><div className="model-actions">{model.downloadable !== false && !model.installed && !model.connected && model.status !== 'downloading' && <button className="button primary" disabled={loading} onClick={() => download(String(model.id))}>确认许可证并下载</button>}{model.connected && <button className="button secondary" disabled={loading || !supportsRelease} title={supportsRelease ? '仅在模型空闲时释放' : '当前服务尚未提供释放接口，需维护重启后启用'} onClick={() => release(String(model.id))}>{supportsRelease ? '释放空闲显存' : '需维护重启后释放'}</button>}</div></article>})}</div><footer><span>{message}</span><button className="button secondary" onClick={onClose}>完成</button></footer></div></div>
}

export default App
