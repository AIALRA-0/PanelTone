import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react'

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
}

type Page = {
  page_index: number
  status: string
  source_url: string
  final_url: string | null
  thumbnail_url: string | null
}

type Uploaded = {
  source_id: string
  name: string
  kind: 'image' | 'book'
  size: number
  duplicate: boolean
  progress: number
}

const statusLabel: Record<string, string> = {
  created: '正在建立', ingesting: '正在展开', ready: '可以开始', queued: '队列等待',
  running: '正在处理', paused: '已暂停', waiting_model: '等待模型',
  needs_attention: '需要检查', completed: '已完成', failed: '处理失败',
  cancelled: '已取消', archived: '回收站',
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.detail || `请求失败 ${response.status}`)
  }
  return response.json()
}

function uploadFile(file: File, onProgress: (value: number) => void): Promise<Uploaded> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    const form = new FormData()
    form.append('file', file)
    request.open('POST', '/api/import')
    request.upload.onprogress = event => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100))
    }
    request.onerror = () => reject(new Error(`无法上传 ${file.name}`))
    request.onload = () => {
      const data = JSON.parse(request.responseText || '{}')
      if (request.status >= 200 && request.status < 300) resolve({ ...data, progress: 100 })
      else reject(new Error(data.detail || `无法上传 ${file.name}`))
    }
    request.send(form)
  })
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

function Icon({ name }: { name: string }) {
  const icons: Record<string, string> = {
    add: '+', settings: '⚙', books: '▤', preview: '◫', progress: '◒',
    play: '▶', pause: 'Ⅱ', more: '•••', close: '×', upload: '↑', check: '✓',
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
  const [presets, setPresets] = useState<Presets | null>(null)
  const [health, setHealth] = useState<Record<string, { ok: boolean; detail?: string }>>({})
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [pages, setPages] = useState<Page[]>([])
  const [pageIndex, setPageIndex] = useState(0)
  const [importOpen, setImportOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [progressOpen, setProgressOpen] = useState(true)
  const [previewDark, setPreviewDark] = useState(true)
  const [previewMode, setPreviewMode] = useState<'compare' | 'source' | 'final' | 'mask'>('compare')
  const [compare, setCompare] = useState(50)
  const [mobileTab, setMobileTab] = useState<'books' | 'preview' | 'settings' | 'progress'>('preview')
  const [filter, setFilter] = useState('all')
  const [message, setMessage] = useState('')
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set())
  const [renameTarget, setRenameTarget] = useState<Job | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Job | null>(null)
  const eventSource = useRef<EventSource | null>(null)
  const selectedRef = useRef<string | null>(null)

  const selected = jobs.find(job => job.id === selectedId) || jobs[0] || null
  const currentPage = pages.find(page => page.page_index === pageIndex) || pages[0]

  useEffect(() => { selectedRef.current = selected?.id || null }, [selected?.id])

  async function refreshJobs() {
    const result = await api<Job[]>('/api/jobs?include_archived=true')
    setJobs(result)
    if (!selectedId && result.length) setSelectedId(result[0].id)
  }

  async function refreshPages(jobId = selected?.id) {
    if (!jobId) return setPages([])
    const result = await api<Page[]>(`/api/jobs/${jobId}/pages`)
    setPages(result)
    if (!result.some(page => page.page_index === pageIndex)) setPageIndex(result[0]?.page_index || 0)
  }

  useEffect(() => {
    Promise.all([
      api<Presets>('/api/presets').then(setPresets),
      api<{ engines: Record<string, { ok: boolean; detail?: string }> }>('/api/health').then(data => setHealth(data.engines)),
      refreshJobs(),
    ]).catch(error => setMessage(error.message))
  }, [])

  useEffect(() => {
    eventSource.current?.close()
    const stream = new EventSource('/api/events')
    eventSource.current = stream
    const update = (event: MessageEvent) => {
      const data = JSON.parse(event.data || '{}')
      refreshJobs().catch(() => undefined)
      if (event.type === 'page_ready' && data.job_id === selectedRef.current) refreshPages(data.job_id)
    }
    ;['snapshot', 'job_status', 'job_queued', 'job_progress', 'page_ready', 'job_error', 'model_progress'].forEach(kind => stream.addEventListener(kind, update))
    stream.onerror = () => setMessage('实时连接正在自动恢复')
    stream.onopen = () => setMessage('')
    return () => stream.close()
  }, [])

  useEffect(() => { refreshPages(selected?.id).catch(() => setPages([])) }, [selected?.id])

  const visibleJobs = useMemo(() => jobs.filter(job => {
    if (filter === 'active') return ['queued', 'running', 'waiting_model'].includes(job.status)
    if (filter === 'attention') return ['failed', 'needs_attention'].includes(job.status)
    if (filter === 'done') return job.status === 'completed'
    if (filter === 'trash') return job.status === 'archived'
    return job.status !== 'archived'
  }), [jobs, filter])

  async function action(jobId: string, name: string) {
    try {
      if (name === 'download') return void (window.location.href = `/api/jobs/${jobId}/download`)
      await api(`/api/jobs/${jobId}/${name}`, { method: 'POST' })
      await refreshJobs()
    } catch (error) {
      setMessage((error as Error).message)
    }
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
      if (name === 'archive') return !['running', 'queued', 'archived'].includes(job.status)
      return ['running', 'queued'].includes(job.status)
    })
    try {
      await Promise.all(allowed.map(job => api(`/api/jobs/${job.id}/${name}`, { method: 'POST' })))
      setSelectedJobs(new Set())
      setMessage(`${allowed.length} 个任务已执行“${name === 'start' ? '开始' : name === 'pause' ? '暂停' : name === 'cancel' ? '取消' : '归档'}”`)
      await refreshJobs()
    } catch (reason) { setMessage((reason as Error).message) }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setMobileTab('preview')} aria-label="PanelTone 首页">
          <span className="brand-mark">PT</span>
          <span><strong>PanelTone</strong><small>本地漫画工作台</small></span>
        </button>
        <div className="top-status" aria-live="polite">
          <span className={`health-dot ${Object.entries(health).some(([id, item]) => id !== 'palette' && item.ok) ? 'ready' : ''}`} />
          <span>{Object.values(health).some(item => item.ok) ? '本地引擎已就绪' : '模型未就绪'}</span>
          <span className="divider" />
          <span>{jobs.filter(job => job.status === 'running').length} 个处理中</span>
          <span>{jobs.filter(job => job.status === 'queued').length} 个等待</span>
        </div>
        <div className="top-actions">
          <button className="button primary" onClick={() => setImportOpen(true)}><Icon name="add" />导入漫画</button>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="模型与设置"><Icon name="settings" /></button>
        </div>
      </header>

      <main className={`workspace mobile-${mobileTab}`}>
        <aside className="library-panel">
          <div className="panel-heading">
            <div><span className="kicker">书库</span><h1>处理任务</h1></div>
            <span className="count">{jobs.length}</span>
          </div>
          <div className="filter-row" role="tablist" aria-label="筛选任务">
            {[['all', '全部'], ['active', '进行中'], ['attention', '需处理'], ['done', '完成'], ['trash', '回收站']].map(([id, label]) =>
              <button key={id} className={filter === id ? 'active' : ''} onClick={() => setFilter(id)}>{label}</button>)}
          </div>
          {selectedJobs.size > 0 && <div className="batch-bar" aria-label="批量任务操作"><strong>{selectedJobs.size} 项</strong><button onClick={() => batchAction('start')}>开始</button><button onClick={() => batchAction('pause')}>暂停</button><button onClick={() => batchAction('cancel')}>取消</button><button onClick={() => batchAction('archive')}>归档</button></div>}
          <div className="job-list">
            {visibleJobs.map(job => (
              <div className="job-item" key={job.id}>
              <input type="checkbox" aria-label={`选择 ${job.display_name}`} checked={selectedJobs.has(job.id)} onChange={event => setSelectedJobs(current => { const next = new Set(current); if (event.target.checked) next.add(job.id); else next.delete(job.id); return next })} />
              <button className={`job-row ${selected?.id === job.id ? 'selected' : ''}`} onClick={() => { setSelectedId(job.id); setMobileTab('preview') }}>
                <span className="job-cover">{job.progress.completed_pages || 0}<small>/{job.page_count}</small></span>
                <span className="job-copy"><strong>{job.display_name}</strong><small>{statusLabel[job.status] || job.status}{job.progress.current_page != null ? ` · 第 ${job.progress.current_page + 1} 页` : ''}</small></span>
                <span className={`status-shape ${job.status}`} aria-label={statusLabel[job.status]} />
              </button>
              </div>
            ))}
            {!visibleJobs.length && <div className="empty-list">当前筛选下没有任务</div>}
          </div>
          <button className="new-book" onClick={() => setImportOpen(true)}><Icon name="add" />导入新漫画</button>
        </aside>

        <section className="preview-panel">
          {selected ? <>
            <div className="preview-toolbar">
              <div className="title-block"><span className="kicker">当前书籍</span><h2>{selected.display_name}</h2></div>
              <div className="toolbar-actions" role="toolbar" aria-label="预览工具">
                <div className="segmented">
                  {([['compare', '对比'], ['source', '原图'], ['final', '结果'], ['mask', '遮罩']] as const).map(([id, label]) =>
                    <button key={id} className={previewMode === id ? 'active' : ''} onClick={() => setPreviewMode(id)}>{label}</button>)}
                </div>
                <button className="plain-button" onClick={() => setPreviewDark(value => !value)}>{previewDark ? '白底' : '深色底'}</button>
                <button className="more-button" aria-label="任务操作"><Icon name="more" /></button>
              </div>
            </div>
            <div className={`canvas ${previewDark ? 'dark' : ''}`}>
              {currentPage ? <>
                {previewMode === 'compare' && currentPage.final_url ? <div className="compare-stage">
                  <img src={currentPage.source_url} alt={`第 ${currentPage.page_index + 1} 页原图`} />
                  <div className="compare-result" style={{ clipPath: `inset(0 0 0 ${compare}%)` }}><img src={currentPage.final_url} alt={`第 ${currentPage.page_index + 1} 页结果`} /></div>
                  <div className="compare-line" style={{ left: `${compare}%` }}><span /></div>
                  <input className="compare-range" aria-label="拖动比较原图和结果" type="range" min="0" max="100" value={compare} onChange={event => setCompare(Number(event.target.value))} />
                </div> : previewMode === 'source' ? <img className="single-page" src={currentPage.source_url} alt={`第 ${currentPage.page_index + 1} 页原图`} /> : previewMode === 'mask' ? <img className="single-page" src={`/api/jobs/${selected.id}/pages/${currentPage.page_index}/mask`} alt={`第 ${currentPage.page_index + 1} 页保护遮罩`} /> : currentPage.final_url ? <img className="single-page" src={currentPage.final_url} alt={`第 ${currentPage.page_index + 1} 页结果`} /> : <div className="waiting-page"><div className="spinner" /><strong>这一页正在处理</strong><span>完成后会自动出现在这里</span></div>}
              </> : <div className="waiting-page"><strong>正在准备页面</strong><span>页面展开后会显示缩略图</span></div>}
            </div>
            <div className="filmstrip" aria-label="漫画页面">
              {pages.map(page => <button key={page.page_index} className={page.page_index === pageIndex ? 'active' : ''} onClick={() => setPageIndex(page.page_index)}>
                <img src={page.thumbnail_url || page.source_url} alt={`第 ${page.page_index + 1} 页`} /><span>{page.page_index + 1}</span>{page.final_url && <i><Icon name="check" /></i>}
              </button>)}
            </div>
          </> : <div className="empty-workspace"><span className="empty-mark">PT</span><h2>从一本漫画开始</h2><p>拖入图片、PDF、ZIP、CBZ、RAR 或 CBR</p><button className="button primary" onClick={() => setImportOpen(true)}><Icon name="upload" />选择漫画</button></div>}
        </section>

        <Inspector selected={selected} presets={presets} health={health} currentPage={currentPage} onAction={action} onRename={() => selected && setRenameTarget(selected)} onDelete={() => selected && setDeleteTarget(selected)} onRetry={retryPage} />
      </main>

      {selected && <section className={`progress-drawer ${progressOpen ? 'open' : ''}`}>
        <button className="progress-summary" onClick={() => setProgressOpen(value => !value)} aria-expanded={progressOpen}>
          <span className={`status-shape ${selected.status}`} />
          <strong>{statusLabel[selected.status] || selected.status}</strong>
          <span>{selected.progress.completed_pages}/{selected.progress.total_pages} 页</span>
          <progress max="100" value={selected.progress.percent}>{selected.progress.percent}%</progress>
          <b>{selected.progress.percent.toFixed(0)}%</b>
          <span>{formatEta(selected.progress.eta_seconds)}</span>
          <span className="drawer-toggle">{progressOpen ? '收起' : '详情'}</span>
        </button>
        {progressOpen && <div className="progress-details" aria-live="polite">
          <div><small>当前阶段</small><strong>{selected.status === 'running' ? '生成与细节保护' : statusLabel[selected.status]}</strong></div>
          <div><small>当前页面</small><strong>{selected.status === 'completed' ? `全部 ${selected.progress.total_pages} 页` : selected.progress.current_page != null ? `${selected.progress.current_page + 1} / ${selected.progress.total_pages}` : '等待开始'}</strong></div>
          <div><small>处理单元</small><strong>{selected.progress.completed_units} / {selected.progress.total_units}</strong></div>
          <div><small>失败单元</small><strong>{selected.progress.failed_units}</strong></div>
          <div><small>处理速度</small><strong>{selected.progress.seconds_per_megapixel == null ? '正在估算' : `${selected.progress.seconds_per_megapixel.toFixed(1)} 秒 / 百万像素`}</strong></div>
          <div><small>预计剩余</small><strong>{formatEta(selected.progress.eta_seconds)}</strong></div>
          <div className="progress-actions">{selected.status === 'completed' ? <button onClick={() => action(selected.id, 'download')}>下载成品</button> : selected.status === 'archived' ? <span>已在回收站</span> : <><button onClick={() => action(selected.id, selected.status === 'running' || selected.status === 'queued' ? 'pause' : 'start')}><Icon name={selected.status === 'running' || selected.status === 'queued' ? 'pause' : 'play'} />{selected.status === 'running' || selected.status === 'queued' ? '暂停' : '运行或继续'}</button><button onClick={() => action(selected.id, 'cancel')}>取消</button></>}</div>
        </div>}
      </section>}

      <nav className="mobile-nav" aria-label="移动端导航">
        {[['books', 'books', '书籍'], ['preview', 'preview', '预览'], ['settings', 'settings', '设置'], ['progress', 'progress', '进度']].map(([id, icon, label]) =>
          <button key={id} className={mobileTab === id ? 'active' : ''} onClick={() => setMobileTab(id as typeof mobileTab)}><Icon name={icon} />{label}</button>)}
      </nav>

      {importOpen && <ImportDialog presets={presets} health={health} onClose={() => setImportOpen(false)} onCreated={async jobsCreated => { setImportOpen(false); await refreshJobs(); if (jobsCreated[0]) setSelectedId(jobsCreated[0]); }} />}
      {settingsOpen && <ModelDialog onClose={() => setSettingsOpen(false)} />}
      {renameTarget && <RenameDialog job={renameTarget} onClose={() => setRenameTarget(null)} onSave={rename} />}
      {deleteTarget && <DeleteDialog job={deleteTarget} onClose={() => setDeleteTarget(null)} onDelete={permanentlyDelete} />}
      {message && <button className="toast" onClick={() => setMessage('')}>{message}<Icon name="close" /></button>}
    </div>
  )
}

function Inspector({ selected, presets, health, currentPage, onAction, onRename, onDelete, onRetry }: {
  selected: Job | null
  presets: Presets | null
  health: Record<string, { ok: boolean }>
  currentPage?: Page
  onAction: (jobId: string, name: string) => void
  onRename: () => void
  onDelete: () => void
  onRetry: (jobId: string, pageIndex: number) => void
}) {
  if (!selected) return <aside className="inspector-panel"><div className="inspector-empty">选择任务后查看设置</div></aside>
  const color = presets?.colors.find(item => item.id === selected.spec.color_preset)
  const style = presets?.styles.find(item => item.id === selected.spec.style_preset)
  return <aside className="inspector-panel">
    <div className="panel-heading"><div><span className="kicker">任务设置</span><h2>处理方案</h2></div><span className="lock">已锁定</span></div>
    <section className="setting-section"><h3>处理方式</h3><strong>{presets?.modes.find(item => item.id === selected.spec.mode)?.name || selected.spec.mode}</strong><OptionInfo item={presets?.modes.find(item => item.id === selected.spec.mode)} /></section>
    <section className="setting-section"><h3>配色</h3><div className="swatch-row"><span className={`swatch ${selected.spec.color_preset}`} /><strong>{color?.name}</strong></div><OptionInfo item={color} /></section>
    <section className="setting-section"><h3>画风</h3><strong>{style?.name}</strong><OptionInfo item={style} /></section>
    <section className="setting-section compact"><div><small>细节保护</small><strong>{presets?.details.find(item => item.id === selected.spec.detail_mode)?.name}</strong></div><div><small>处理单位</small><strong>{presets?.panels.find(item => item.id === selected.spec.panel_mode)?.name}</strong></div><div><small>导出</small><strong>{presets?.outputs.find(item => item.id === selected.spec.output_format)?.name}</strong></div></section>
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

function ImportDialog({ presets, health, onClose, onCreated }: { presets: Presets | null; health: Record<string, { ok: boolean }>; onClose: () => void; onCreated: (jobIds: string[]) => void }) {
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
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const directoryInput = useRef<HTMLInputElement>(null)
  const naturalNames = useMemo(() => new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' }), [])

  function addFiles(list: FileList | File[]) {
    const incoming = Array.from(list)
    setFiles(current => [...current, ...incoming.filter(file => !current.some(item => item.name === file.name && item.size === file.size))].sort((left, right) => naturalNames.compare(left.webkitRelativePath || left.name, right.webkitRelativePath || right.name)))
  }
  function drop(event: DragEvent) { event.preventDefault(); addFiles(event.dataTransfer.files) }
  function move(index: number, offset: number) { setFiles(current => { const copy = [...current]; const next = index + offset; if (next < 0 || next >= copy.length) return current; [copy[index], copy[next]] = [copy[next], copy[index]]; return copy }) }

  async function create() {
    if (!files.length && !localPath.trim()) return setError('请先选择漫画文件或填写本地路径')
    if (files.length && localPath.trim()) return setError('文件上传和本地路径请分开建立任务')
    setBusy(true); setError('')
    try {
      const common = { mode, color_preset: color, style_preset: style, detail_mode: detail, panel_mode: panel, output_format: output, engine, max_retries: profile === 'fast' ? 0 : profile === 'repair' ? 4 : 2, adult_fictional_content: adult, preserve_text: true, preserve_ink: detail !== 'generative' }
      if (localPath.trim()) {
        const response = await api<{ job_id: string }>('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...common, source: localPath.trim(), display_name: name }) })
        await api(`/api/jobs/${response.job_id}/start`, { method: 'POST' })
        return onCreated([response.job_id])
      }
      const uploaded: Uploaded[] = []
      for (let index = 0; index < files.length; index++) {
        const item = await uploadFile(files[index], value => setUploads(current => {
          const copy = [...current]
          copy[index] = { source_id: '', name: files[index].name, kind: 'image', size: files[index].size, duplicate: false, progress: value }
          return copy
        }))
        uploaded.push(item); setUploads([...uploaded])
      }
      const uniqueUploaded = uploaded.filter((item, index) => uploaded.findIndex(candidate => candidate.source_id === item.source_id) === index)
      const response = await api<{ jobs: { job_id: string }[] }>('/api/jobs/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...common, source_ids: uniqueUploaded.map(item => item.source_id), image_order: uniqueUploaded.filter(item => item.kind === 'image').map(item => item.source_id), image_book_name: name }) })
      for (const job of response.jobs) await api(`/api/jobs/${job.job_id}/start`, { method: 'POST' })
      onCreated(response.jobs.map(job => job.job_id))
    } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
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
          {files.map((file, index) => <div className="source-row" key={`${file.name}-${file.size}`}><span className="file-type">{file.name.split('.').pop()?.toUpperCase()}</span><span><strong>{file.name}</strong><small>{formatBytes(file.size)}{uploads[index] ? ` · ${uploads[index].progress}%` : ''}</small></span><div><button onClick={() => move(index, -1)} aria-label="上移">↑</button><button onClick={() => move(index, 1)} aria-label="下移">↓</button><button onClick={() => setFiles(current => current.filter((_, item) => item !== index))} aria-label="移除">×</button></div></div>)}
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
    <footer><span className="error-text" aria-live="polite">{error}</span><button className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={busy || (!files.length && !localPath.trim())} onClick={create}>{busy ? '正在导入与建书' : localPath.trim() ? '建立任务并开始' : `建立 ${files.filter(file => !/\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name)).length + (files.some(file => /\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name)) ? 1 : 0)} 本并开始`}</button></footer>
  </div></div>
}

function ModelDialog({ onClose }: { onClose: () => void }) {
  const [models, setModels] = useState<Array<Record<string, string | boolean>>>([])
  const [message, setMessage] = useState('')
  useEffect(() => { api<Array<Record<string, string | boolean>>>('/api/models').then(setModels) }, [])
  async function download(id: string) { setMessage('正在启动模型下载'); await api(`/api/models/${id}/download`, { method: 'POST' }); setMessage('下载已在后台开始') }
  return <div className="dialog-backdrop"><div className="dialog model-dialog" role="dialog" aria-modal="true"><header><div><span className="kicker">设置</span><h2>模型与本地存储</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></header><div className="model-list">{models.map(model => <article key={String(model.id)}><div><span className={`health-dot ${model.connected || model.installed ? 'ready' : ''}`} /><div><strong>{String(model.name)}</strong><small>{String(model.purpose)}</small></div></div><dl><dt>来源</dt><dd>{String(model.repository)}</dd><dt>许可证</dt><dd><a href={String(model.license_url)} target="_blank" rel="noreferrer">{String(model.license)}</a></dd><dt>显存提示</dt><dd>{String(model.memory)}</dd><dt>下载大小</dt><dd>{String(model.download_size)}</dd><dt>保存位置</dt><dd>{String(model.storage)}</dd><dt>状态</dt><dd>{model.connected ? '本地模型服务已连接' : model.installed ? '权重已下载，等待服务连接' : '尚未下载'}</dd></dl>{!model.installed && <button className="button primary" onClick={() => download(String(model.id))}>确认许可证并下载</button>}</article>)}</div><footer><span>{message}</span><button className="button secondary" onClick={onClose}>完成</button></footer></div></div>
}

export default App
