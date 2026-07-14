export type SourceInfo = {
  id: string
  source: string
  chapter: string
  section: string
  page_start: number | null
  page_end: number | null
  score: number
  doc_type: 'textbook' | 'question' | string
}

export type KBStatus = {
  id: string
  state: 'ready' | 'building' | 'cancelling' | 'cancelled' | 'error' | 'missing'
  documents: number
  indexed_documents?: number
  failed_documents?: number
  chunks: number
  questions: number
  relations: number
  message: string
  source_warnings?: Array<{ source: string; warnings: string[] }>
  available?: boolean
  progress?: number
  stage?: string
  cancellable?: boolean
  circuits?: number
  layout_elements?: number
  schema_version?: string
  pipeline_layers?: Record<string, unknown>
  validation?: Record<string, unknown>
}

export type TutorAction = 'auto' | 'understand' | 'method' | 'hint' | 'check_step' | 'explain_error' | 'full_solution'
export type TutoringMode = 'guided' | 'full'

export type AttachmentInfo = {
  id: string
  name: string
  content_type: string
  size: number
  kind: 'image' | 'document'
  url: string
}

export type ModelProviderId = 'ollama' | 'lmstudio' | 'deepseek' | 'qwen' | 'custom'

export type ModelConfig = {
  provider: ModelProviderId
  model: string
  apiKey: string
  baseUrl: string
}

export type ModelProviderInfo = {
  id: ModelProviderId
  label: string
  description: string
  models: string[]
  default_model: string
  base_url: string
  requires_api_key: boolean
  configured: boolean
}

export type ModelCatalog = {
  default: { provider: ModelProviderId; model: string }
  providers: ModelProviderInfo[]
}

export type SessionSummary = {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export type StoredMessage = {
  role: 'user' | 'assistant'
  content: string
  created_at: string
  agent?: string
  provider?: ModelProviderId
  model?: string
  failed?: boolean
  retry_message?: string
  attachment_ids?: string[]
  retry_attachment_ids?: string[]
}

export type WrongQuestionMessage = {
  role: 'user' | 'assistant'
  content: string
  agent?: string
  model?: string
  created_at?: string
}

export type WrongQuestion = {
  id: string
  title: string
  category_id: string
  session_id: string
  knowledge_base: string
  knowledge_points: string[]
  messages: WrongQuestionMessage[]
  created_at: string
  updated_at: string
}

export type WrongQuestionCategory = {
  id: string
  name: string
  created_at: string
  updated_at: string
}

export type WrongNotebook = {
  categories: WrongQuestionCategory[]
  items: WrongQuestion[]
}

export type KnowledgeGraphQuestion = { id: string; title: string; difficulty?: string; updated_at?: string }

export type KnowledgeGraphNode = {
  id: string
  type: 'knowledge_point'
  label: string
  category_id: string
  category_label: string
  summary: string
  definition: string
  key_points: string[]
  sources: { name: string; chunks: number }[]
  sections: string[]
  questions: KnowledgeGraphQuestion[]
  wrong_questions: KnowledgeGraphQuestion[]
  chunk_count: number
}

export type KnowledgeGraph = {
  knowledge_base: string
  root: { id: string; label: string }
  categories: { id: string; label: string; count: number }[]
  nodes: KnowledgeGraphNode[]
  edges: { source: string; target: string; type: string }[]
  stats: { sources: number; knowledge_points: number; questions: number; wrong_questions: number }
}

type SSECallbacks = {
  onStatus: (data: { stage: string; message: string; agent: string }) => void
  onMeta: (data: { intent: string; agent: string; provider: ModelProviderId; model: string; sources: SourceInfo[]; verification?: Record<string, unknown>; tutor_action?: TutorAction; hint_level?: number; problem?: Record<string, unknown>; diagnosis?: Record<string, unknown> }) => void
  onDelta: (content: string) => void
  onDone: () => void
  onError: (message: string) => void
}

function parseEvent(block: string) {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  const raw = dataLines.join('\n')
  return { event, data: raw ? JSON.parse(raw) : {} }
}

export async function streamChat(
  payload: {
    session_id: string
    message: string
    mode: string
    tutor_action: TutorAction
    hint_level: number
    tutoring_mode: TutoringMode
    knowledge_base: string
    attachment_ids: string[]
    model_provider: ModelProviderId
    model: string
    api_key: string
    base_url: string
  },
  callbacks: SSECallbacks,
  signal?: AbortSignal,
) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  })
  if (!response.ok || !response.body) {
    const detail = await response.text()
    throw new Error(detail || `请求失败 (${response.status})`)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const readWithIdleTimeout = async () => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined
    try {
      return await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(
            () => reject(new Error('模型连接长时间没有任何进度，请重新生成')),
            75_000,
          )
        }),
      ])
    } catch (error) {
      await reader.cancel().catch(() => undefined)
      throw error
    } finally {
      if (timeoutId) clearTimeout(timeoutId)
    }
  }
  let buffer = ''
  let receivedTerminalEvent = false
  const dispatchBlock = (block: string) => {
    if (!block.trim()) return
    const { event, data } = parseEvent(block)
    if (event === 'status') callbacks.onStatus(data)
    if (event === 'meta') callbacks.onMeta(data)
    if (event === 'delta') callbacks.onDelta(data.content || '')
    if (event === 'done') {
      receivedTerminalEvent = true
      callbacks.onDone()
    }
    if (event === 'error') {
      receivedTerminalEvent = true
      callbacks.onError(data.message || '生成失败')
    }
  }
  while (true) {
    const { done, value } = await readWithIdleTimeout()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const blocks = buffer.split(/\r?\n\r?\n/)
    buffer = blocks.pop() || ''
    for (const block of blocks) dispatchBlock(block)
    if (done) {
      dispatchBlock(buffer)
      break
    }
  }
  if (!receivedTerminalEvent && !signal?.aborted) {
    throw new Error('回答连接提前结束，已保留收到的内容，请重新生成')
  }
}

export async function fetchModels(): Promise<ModelCatalog> {
  const response = await fetch('/api/models')
  if (!response.ok) throw new Error('无法读取模型列表')
  return response.json()
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const response = await fetch('/api/sessions')
  if (!response.ok) throw new Error('无法读取历史会话')
  return (await response.json()).sessions || []
}

export async function fetchSession(sessionId: string): Promise<StoredMessage[]> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`)
  if (!response.ok) throw new Error('无法恢复历史会话')
  return (await response.json()).messages || []
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    const result = await response.json().catch(() => ({}))
    throw new Error(result.detail || '历史会话删除失败')
  }
}

export async function uploadChatAttachment(file: File, sessionId: string): Promise<AttachmentInfo> {
  const data = new FormData()
  data.append('file', file)
  data.append('session_id', sessionId)
  const response = await fetch('/api/attachments', { method: 'POST', body: data })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '附件上传失败')
  return result.attachment
}

export async function fetchKnowledgeBases(): Promise<KBStatus[]> {
  const response = await fetch('/api/kb/status')
  if (!response.ok) throw new Error('无法读取知识库状态')
  return (await response.json()).knowledge_bases || []
}

export async function uploadKnowledgeFile(file: File, knowledgeBase: string) {
  const data = new FormData()
  data.append('file', file)
  data.append('knowledge_base', knowledgeBase)
  data.append('rebuild', 'true')
  const response = await fetch('/api/upload', { method: 'POST', body: data })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '上传失败')
  return result
}

export async function uploadKnowledgeFiles(
  files: File[],
  knowledgeBase: string,
  documentType: 'auto' | 'textbook' | 'exam' | 'question_bank' | 'notes' = 'auto',
) {
  const data = new FormData()
  files.forEach((file) => data.append('files', file))
  data.append('knowledge_base', knowledgeBase)
  data.append('document_type', documentType)
  const response = await fetch('/api/kb/ingest', { method: 'POST', body: data })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '批量导入失败')
  return result
}

export async function rebuildKnowledgeBase(
  knowledgeBase: string,
  modelConfig: ModelConfig,
) {
  return jsonRequest('/api/kb/rebuild', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      knowledge_base: knowledgeBase,
      model_provider: modelConfig.provider,
      model: modelConfig.model,
      api_key: modelConfig.apiKey,
      base_url: modelConfig.baseUrl,
      chapter_limit: null,
    }),
  })
}

export async function cancelKnowledgeBaseBuild(knowledgeBase: string) {
  return jsonRequest(`/api/kb/${encodeURIComponent(knowledgeBase)}/build`, {
    method: 'DELETE',
  })
}

export async function deleteKnowledgeBase(knowledgeBase: string) {
  return jsonRequest(`/api/kb/${encodeURIComponent(knowledgeBase)}`, {
    method: 'DELETE',
  })
}

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(result.detail || result.error || `请求失败 (${response.status})`)
  return result as T
}

export async function fetchWrongNotebook(): Promise<WrongNotebook> {
  return jsonRequest<WrongNotebook>('/api/wrong-questions')
}

export async function createWrongQuestion(payload: {
  session_id: string
  title?: string
  category_id?: string
  knowledge_base: string
  messages: WrongQuestionMessage[]
}): Promise<WrongQuestion> {
  const result = await jsonRequest<{ item: WrongQuestion }>('/api/wrong-questions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return result.item
}

export async function updateWrongQuestion(
  id: string,
  changes: { title?: string; category_id?: string },
): Promise<WrongQuestion> {
  const result = await jsonRequest<{ item: WrongQuestion }>(`/api/wrong-questions/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(changes),
  })
  return result.item
}

export async function deleteWrongQuestion(id: string): Promise<void> {
  await jsonRequest(`/api/wrong-questions/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function createWrongQuestionCategory(name: string): Promise<WrongQuestionCategory> {
  const result = await jsonRequest<{ category: WrongQuestionCategory }>('/api/wrong-questions/categories', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  return result.category
}

export async function fetchKnowledgeGraph(knowledgeBase: string): Promise<KnowledgeGraph> {
  return jsonRequest<KnowledgeGraph>(`/api/knowledge-graph?knowledge_base=${encodeURIComponent(knowledgeBase)}`)
}
