import { create } from 'zustand'
import { AttachmentInfo, KBStatus, ModelConfig, ModelProviderId, SourceInfo, StoredMessage, streamChat, uploadChatAttachment } from '../lib/api'

export type ChatMode = 'auto' | 'answer' | 'quiz' | 'plan'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  agent?: string
  sources?: SourceInfo[]
  failed?: boolean
  attachments?: AttachmentInfo[]
  model?: string
  provider?: ModelProviderId
  knowledgeBase?: string
}

export type PendingAttachment = {
  localId: string
  name: string
  size: number
  contentType: string
  kind: 'image' | 'document'
  status: 'uploading' | 'ready' | 'error'
  attachment?: AttachmentInfo
  error?: string
}

const sessionKey = 'circuitmind-session-id'
const studentKey = 'circuitmind-student-id'
const modelConfigKey = 'circuitmind-model-config'
const defaultKnowledgeBaseKey = 'circuitmind-default-knowledge-base'
export const CHAT_MODEL_PROVIDER: ModelProviderId = 'qwen'
export const CHAT_MODEL = 'qwen3-vl-flash'
export const QWEN_CHAT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

const defaultModelConfig: ModelConfig = {
  provider: CHAT_MODEL_PROVIDER,
  model: CHAT_MODEL,
  apiKey: '',
  baseUrl: QWEN_CHAT_BASE_URL,
}

function getSessionId() {
  let value = localStorage.getItem(sessionKey)
  if (!value) {
    value = `student-${crypto.randomUUID()}`
    localStorage.setItem(sessionKey, value)
  }
  return value
}

function fixedChatModelConfig(value: Partial<ModelConfig>): ModelConfig {
  const qwenConfig = value.provider === CHAT_MODEL_PROVIDER
  return {
    provider: CHAT_MODEL_PROVIDER,
    model: CHAT_MODEL,
    apiKey: qwenConfig && typeof value.apiKey === 'string' ? value.apiKey : '',
    baseUrl:
      qwenConfig && typeof value.baseUrl === 'string' && value.baseUrl.trim()
        ? value.baseUrl.trim()
        : QWEN_CHAT_BASE_URL,
  }
}

function getStudentId() {
  let value = localStorage.getItem(studentKey)
  if (!value) {
    value = `learner-${crypto.randomUUID()}`
    localStorage.setItem(studentKey, value)
  }
  return value
}

function getModelConfig(): ModelConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(modelConfigKey) || '{}')
    const config = fixedChatModelConfig(stored)
    localStorage.setItem(modelConfigKey, JSON.stringify(config))
    return config
  } catch {
    return defaultModelConfig
  }
}

function getDefaultKnowledgeBase(): string {
  const stored = localStorage.getItem(defaultKnowledgeBaseKey)?.trim() || ''
  return /^[A-Za-z0-9_-]{1,48}$/.test(stored) ? stored : ''
}

const initialKnowledgeBase = getDefaultKnowledgeBase()

function legacySourcesFromContent(content: string): SourceInfo[] {
  const sources: SourceInfo[] = []
  const pattern = /^-\s+\[资料(\d+)\]\s+(.+?)\s+·\s+(.+?)\s+·\s+第\s*(\d+)(?:[–-](\d+))?\s*页\s*$/gm
  for (const match of content.matchAll(pattern)) {
    const pageStart = Number(match[4])
    const pageEnd = Number(match[5] || match[4])
    sources.push({
      id: `history-source-${match[1]}-${match[2]}-${pageStart}`,
      source: match[2].trim(),
      chapter: match[3].trim(),
      section: match[3].trim(),
      page_start: pageStart,
      page_end: pageEnd,
      score: 0,
      doc_type: 'textbook',
      historical: true,
    })
  }
  return sources
}

type ChatState = {
  studentId: string
  sessionId: string
  mode: ChatMode
  knowledgeBase: string
  defaultKnowledgeBase: string
  modelConfig: ModelConfig
  messages: ChatMessage[]
  streaming: boolean
  stage: string
  stageAgent: string
  activeSources: SourceInfo[]
  activeMessageId?: string
  pendingAttachments: PendingAttachment[]
  controller?: AbortController
  setMode: (mode: ChatMode) => void
  setKnowledgeBase: (id: string) => void
  setDefaultKnowledgeBase: (id: string) => void
  syncKnowledgeBases: (knowledgeBases: KBStatus[]) => void
  setModelConfig: (config: ModelConfig) => void
  addAttachments: (files: File[]) => Promise<void>
  removeAttachment: (localId: string) => void
  activateMessage: (messageId: string) => void
  loadSession: (sessionId: string, messages: StoredMessage[]) => void
  send: (message: string) => Promise<void>
  stop: () => void
  clear: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  studentId: getStudentId(),
  sessionId: getSessionId(),
  mode: 'auto',
  knowledgeBase: initialKnowledgeBase,
  defaultKnowledgeBase: initialKnowledgeBase,
  modelConfig: getModelConfig(),
  messages: [],
  streaming: false,
  stage: '',
  stageAgent: '',
  activeSources: [],
  activeMessageId: undefined,
  pendingAttachments: [],
  setMode: (mode) => set({ mode }),
  setKnowledgeBase: (knowledgeBase) => set({ knowledgeBase }),
  setDefaultKnowledgeBase: (defaultKnowledgeBase) => {
    if (!defaultKnowledgeBase) {
      localStorage.removeItem(defaultKnowledgeBaseKey)
      set({ defaultKnowledgeBase: '', knowledgeBase: '' })
      return
    }
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(defaultKnowledgeBase)) return
    localStorage.setItem(defaultKnowledgeBaseKey, defaultKnowledgeBase)
    set({ defaultKnowledgeBase, knowledgeBase: defaultKnowledgeBase })
  },
  syncKnowledgeBases: (knowledgeBases) => {
    const currentDefault = get().defaultKnowledgeBase
    const available = knowledgeBases.filter((item) => item.state === 'ready' || item.available)
    if (available.some((item) => item.id === currentDefault)) return

    const replacement = available[0]?.id || ''
    if (replacement) {
      localStorage.setItem(defaultKnowledgeBaseKey, replacement)
    } else {
      localStorage.removeItem(defaultKnowledgeBaseKey)
    }
    set((state) => ({
      defaultKnowledgeBase: replacement,
      knowledgeBase:
        !state.knowledgeBase || state.knowledgeBase === currentDefault
          ? replacement
          : state.knowledgeBase,
    }))
  },
  setModelConfig: (modelConfig) => {
    const normalized = fixedChatModelConfig(modelConfig)
    localStorage.setItem(modelConfigKey, JSON.stringify(normalized))
    set({ modelConfig: normalized })
  },
  addAttachments: async (files) => {
    const available = Math.max(0, 5 - get().pendingAttachments.length)
    const selected = files.slice(0, available)
    const pending = selected.map<PendingAttachment>((file) => ({
      localId: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      contentType: file.type,
      kind: file.type.startsWith('image/') ? 'image' : 'document',
      status: 'uploading',
    }))
    set((state) => ({ pendingAttachments: [...state.pendingAttachments, ...pending] }))
    await Promise.all(
      selected.map(async (file, index) => {
        const localId = pending[index].localId
        try {
          const attachment = await uploadChatAttachment(file, get().sessionId)
          set((state) => ({
            pendingAttachments: state.pendingAttachments.map((item) =>
              item.localId === localId ? { ...item, status: 'ready', attachment } : item,
            ),
          }))
        } catch (error) {
          const detail = error instanceof Error ? error.message : '上传失败'
          set((state) => ({
            pendingAttachments: state.pendingAttachments.map((item) =>
              item.localId === localId ? { ...item, status: 'error', error: detail } : item,
            ),
          }))
        }
      }),
    )
  },
  removeAttachment: (localId) => set((state) => ({
    pendingAttachments: state.pendingAttachments.filter((item) => item.localId !== localId),
  })),
  activateMessage: (messageId) => set((state) => {
    if (state.activeMessageId === messageId) return state
    const message = state.messages.find((item) => item.id === messageId)
    if (!message || message.role !== 'assistant') return state
    return {
      activeMessageId: messageId,
      activeSources: message.sources || [],
    }
  }),
  loadSession: (sessionId, storedMessages) => {
    get().controller?.abort()
    localStorage.setItem(sessionKey, sessionId)
    const messages = storedMessages.map<ChatMessage>((item, index) => {
      const sources = item.sources?.length
        ? item.sources
        : legacySourcesFromContent(item.content)
      return {
        id: `history-${item.created_at}-${index}`,
        role: item.role,
        content: item.content,
        agent: item.agent,
        provider: item.provider,
        model: item.model,
        knowledgeBase: item.knowledge_base,
        attachments: item.attachments || [],
        sources,
      }
    })
    const latestAssistant = [...messages].reverse().find((item) => item.role === 'assistant')
    set({
      sessionId,
      messages,
      streaming: false,
      stage: '',
      stageAgent: '',
      activeSources: latestAssistant?.sources || [],
      activeMessageId: latestAssistant?.id,
      pendingAttachments: [],
      controller: undefined,
    })
  },
  send: async (rawMessage) => {
    const readyAttachments = get().pendingAttachments
      .filter((item) => item.status === 'ready' && item.attachment)
      .map((item) => item.attachment!)
    const hasUnfinished = get().pendingAttachments.some((item) => item.status !== 'ready')
    const message = rawMessage.trim() || (
      readyAttachments.length
        ? get().mode === 'quiz'
          ? '请根据附件中的原题生成一道同类型新题。'
          : '请识别并解答附件中的电路题。'
        : ''
    )
    if ((!message && !readyAttachments.length) || get().streaming || hasUnfinished) return
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: message,
      attachments: readyAttachments,
      knowledgeBase: get().knowledgeBase,
    }
    const assistantId = crypto.randomUUID()
    const selectedModel = get().modelConfig
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      model: selectedModel.model,
      provider: selectedModel.provider,
      knowledgeBase: get().knowledgeBase,
    }
    const controller = new AbortController()
    set((state) => ({
      messages: [...state.messages, userMessage, assistantMessage],
      streaming: true,
      stage: `正在连接 ${selectedModel.model}…`,
      stageAgent: '系统',
      activeSources: [],
      activeMessageId: assistantId,
      pendingAttachments: [],
      controller,
    }))
    try {
      await streamChat(
        {
          session_id: get().sessionId,
          message,
          mode: get().mode,
          knowledge_base: get().knowledgeBase,
          attachment_ids: readyAttachments.map((item) => item.id),
          model_provider: selectedModel.provider,
          model: selectedModel.model,
          api_key: selectedModel.apiKey,
          base_url: selectedModel.baseUrl,
        },
        {
          onStatus: (data) => set({ stage: data.message, stageAgent: data.agent }),
          onMeta: (data) => {
            set((state) => ({
              activeSources:
                state.activeMessageId === assistantId
                  ? data.sources || []
                  : state.activeSources,
              messages: state.messages.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      agent: data.agent,
                      provider: data.provider,
                      model: data.model,
                      sources: data.sources,
                    }
                  : item,
              ),
            }))
          },
          onDelta: (content) => {
            set((state) => ({
              messages: state.messages.map((item) =>
                item.id === assistantId ? { ...item, content: item.content + content } : item,
              ),
            }))
          },
          onDone: () => set({ streaming: false, stage: '', stageAgent: '', controller: undefined }),
          onError: (error) => {
            set((state) => ({
              streaming: false,
              stage: '',
              messages: state.messages.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      content: item.content
                        ? `${item.content}\n\n> ⚠️ 生成未完整结束：${error}`
                        : `生成失败：${error}`,
                      failed: true,
                    }
                  : item,
              ),
            }))
          },
        },
        controller.signal,
      )
      set({ streaming: false, stage: '', stageAgent: '', controller: undefined })
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        set({ streaming: false, stage: '已停止生成', stageAgent: '', controller: undefined })
        return
      }
      const detail = error instanceof Error ? error.message : '未知错误'
      set((state) => ({
        streaming: false,
        stage: '',
        controller: undefined,
        messages: state.messages.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: item.content
                  ? `${item.content}\n\n> ⚠️ 回答连接提前结束：${detail}`
                  : `连接失败：${detail}`,
                failed: true,
              }
            : item,
        ),
      }))
    }
  },
  stop: () => {
    get().controller?.abort()
  },
  clear: () => {
    const sessionId = `student-${crypto.randomUUID()}`
    localStorage.setItem(sessionKey, sessionId)
    get().controller?.abort()
    set({
      sessionId,
      knowledgeBase: get().defaultKnowledgeBase,
      messages: [],
      streaming: false,
      stage: '',
      activeSources: [],
      activeMessageId: undefined,
      pendingAttachments: [],
      controller: undefined,
    })
  },
}))
