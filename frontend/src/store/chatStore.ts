import { create } from 'zustand'
import { AttachmentInfo, ModelConfig, ModelProviderId, SourceInfo, StoredMessage, streamChat, TutorAction, TutoringMode, uploadChatAttachment } from '../lib/api'

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
  hintLevel?: number
  tutorAction?: TutorAction
  diagnosis?: Record<string, unknown>
  createdAt?: string
  retryContent?: string
  retryAttachmentIds?: string[]
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
const modelConfigKey = 'circuitmind-model-config'
const knowledgeBaseKey = 'circuitmind-knowledge-base'

const defaultModelConfig: ModelConfig = {
  provider: 'lmstudio',
  model: 'qwen/qwen3.5-9b',
  apiKey: '',
  baseUrl: 'http://127.0.0.1:1234/v1',
}

function getSessionId() {
  let value = localStorage.getItem(sessionKey)
  if (!value) {
    value = `student-${crypto.randomUUID()}`
    localStorage.setItem(sessionKey, value)
  }
  return value
}

function getKnowledgeBase() {
  const value = localStorage.getItem(knowledgeBaseKey)?.trim() || 'default'
  return /^[A-Za-z0-9_-]{1,48}$/.test(value) ? value : 'default'
}

function getModelConfig(): ModelConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(modelConfigKey) || '{}')
    const providers: ModelProviderId[] = ['ollama', 'lmstudio', 'deepseek', 'qwen', 'custom']
    if (!providers.includes(stored.provider) || typeof stored.model !== 'string') {
      return defaultModelConfig
    }
    return {
      provider: stored.provider,
      model: stored.model || defaultModelConfig.model,
      apiKey: typeof stored.apiKey === 'string' ? stored.apiKey : '',
      baseUrl: typeof stored.baseUrl === 'string' ? stored.baseUrl : '',
    }
  } catch {
    return defaultModelConfig
  }
}

type ChatState = {
  sessionId: string
  tutoringMode: TutoringMode
  knowledgeBase: string
  modelConfig: ModelConfig
  messages: ChatMessage[]
  streaming: boolean
  stage: string
  stageAgent: string
  activeSources: SourceInfo[]
  hintLevel: number
  pendingAttachments: PendingAttachment[]
  controller?: AbortController
  setTutoringMode: (mode: TutoringMode) => void
  setKnowledgeBase: (id: string) => void
  setModelConfig: (config: ModelConfig) => void
  addAttachments: (files: File[]) => Promise<void>
  removeAttachment: (localId: string) => void
  loadSession: (sessionId: string, messages: StoredMessage[]) => void
  send: (message: string, attachmentIds?: string[]) => Promise<void>
  stop: () => void
  clear: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessionId: getSessionId(),
  tutoringMode: 'guided',
  knowledgeBase: getKnowledgeBase(),
  modelConfig: getModelConfig(),
  messages: [],
  streaming: false,
  stage: '',
  stageAgent: '',
  activeSources: [],
  hintLevel: 1,
  pendingAttachments: [],
  setTutoringMode: (tutoringMode) => set({ tutoringMode }),
  setKnowledgeBase: (knowledgeBase) => {
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(knowledgeBase)) return
    localStorage.setItem(knowledgeBaseKey, knowledgeBase)
    set({ knowledgeBase })
  },
  setModelConfig: (modelConfig) => {
    localStorage.setItem(modelConfigKey, JSON.stringify(modelConfig))
    set({ modelConfig })
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
  loadSession: (sessionId, storedMessages) => {
    get().controller?.abort()
    localStorage.setItem(sessionKey, sessionId)
    const messages = storedMessages.map<ChatMessage>((item, index) => ({
      id: `history-${item.created_at}-${index}`,
      role: item.role,
      content: item.content,
      agent: item.agent,
      provider: item.provider,
      model: item.model,
      failed: item.failed,
      retryContent: item.retry_message,
      retryAttachmentIds: item.retry_attachment_ids || (item.role === 'user' ? item.attachment_ids : undefined),
      createdAt: item.created_at,
    }))
    const lastMessage = messages.at(-1)
    if (lastMessage?.role === 'user') {
      messages.push({
        id: `recovery-${lastMessage.id}`,
        role: 'assistant',
        content: '> ⚠️ 上一次回答在完成前中断，原问题已经保留。',
        agent: '系统恢复',
        provider: get().modelConfig.provider,
        model: get().modelConfig.model,
        failed: true,
        retryContent: lastMessage.content.split('\n[附件：', 1)[0],
        retryAttachmentIds: lastMessage.retryAttachmentIds,
        createdAt: new Date().toISOString(),
      })
    }
    set({
      sessionId,
      messages,
      streaming: false,
      stage: '',
      stageAgent: '',
      activeSources: [],
      hintLevel: 1,
      pendingAttachments: [],
      controller: undefined,
    })
  },
  send: async (rawMessage, retryAttachmentIds = []) => {
    const readyAttachments = get().pendingAttachments
      .filter((item) => item.status === 'ready' && item.attachment)
      .map((item) => item.attachment!)
    const hasUnfinished = get().pendingAttachments.some((item) => item.status !== 'ready')
    const attachmentIds = retryAttachmentIds.length
      ? retryAttachmentIds
      : readyAttachments.map((item) => item.id)
    const message = rawMessage.trim() || (
      readyAttachments.length
        ? '请识别附件中的电路题，并按我选择的辅导模式帮助我。'
        : ''
    )
    if ((!message && !attachmentIds.length) || get().streaming || hasUnfinished) return
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: message,
      attachments: readyAttachments,
      createdAt: new Date().toISOString(),
    }
    const assistantId = crypto.randomUUID()
    const selectedModel = get().modelConfig
    const effectiveHintLevel = get().tutoringMode === 'full' ? 5 : Math.max(1, Math.min(5, get().hintLevel))
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      model: selectedModel.model,
      provider: selectedModel.provider,
      createdAt: new Date().toISOString(),
    }
    const controller = new AbortController()
    set((state) => ({
      messages: [...state.messages, userMessage, assistantMessage],
      streaming: true,
      stage: `正在连接 ${selectedModel.model}…`,
      stageAgent: '系统',
      activeSources: [],
      pendingAttachments: [],
      controller,
    }))
    try {
      await streamChat(
        {
          session_id: get().sessionId,
          message,
          mode: 'auto',
          tutor_action: 'auto',
          hint_level: effectiveHintLevel,
          tutoring_mode: get().tutoringMode,
          knowledge_base: get().knowledgeBase,
          attachment_ids: attachmentIds,
          model_provider: selectedModel.provider,
          model: selectedModel.model,
          api_key: selectedModel.apiKey,
          base_url: selectedModel.baseUrl,
        },
        {
          onStatus: (data) => set({ stage: data.message, stageAgent: data.agent }),
          onMeta: (data) => {
            set((state) => ({
              activeSources: data.sources || [],
              hintLevel: data.hint_level || state.hintLevel,
              messages: state.messages.map((item) =>
                item.id === assistantId
                  ? { ...item, agent: data.agent, provider: data.provider, model: data.model, sources: data.sources, hintLevel: data.hint_level, tutorAction: data.tutor_action, diagnosis: data.diagnosis }
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
                      retryContent: message,
                      retryAttachmentIds: attachmentIds,
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
        set((state) => ({
          streaming: false,
          stage: '',
          stageAgent: '',
          controller: undefined,
          messages: state.messages.map((item) =>
            item.id === assistantId
              ? {
                  ...item,
                  content: item.content
                    ? `${item.content}\n\n> ⚠️ 生成已停止，可以重新生成本题。`
                    : '> ⚠️ 生成已停止，原问题已经保留。',
                  failed: true,
                  retryContent: message,
                  retryAttachmentIds: attachmentIds,
                }
              : item,
          ),
        }))
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
                retryContent: message,
                retryAttachmentIds: attachmentIds,
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
    set({ sessionId, messages: [], streaming: false, stage: '', activeSources: [], hintLevel: 1, pendingAttachments: [], controller: undefined })
  },
}))
