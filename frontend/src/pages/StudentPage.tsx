import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  App as AntApp,
  Button,
  Input,
  Modal,
  Popconfirm,
  Segmented,
  Select,
  Tag,
  Tooltip,
  Upload,
  type UploadFile,
} from 'antd'
import {
  ArrowUp,
  BookmarkPlus,
  BookOpen,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  CircleStop,
  Cloud,
  Clock3,
  Cpu,
  Database,
  FileText,
  GraduationCap,
  HelpCircle,
  Layers3,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Plus,
  Paperclip,
  Search,
  RefreshCcw,
  KeyRound,
  ServerCog,
  ShieldCheck,
  Trash2,
  UploadCloud,
  UserRound,
  WandSparkles,
  X,
  Zap,
} from 'lucide-react'
import MathMarkdown from '../components/MathMarkdown'
import {
  createWrongQuestion,
  deleteSession,
  fetchKnowledgeBases,
  fetchModels,
  fetchSession,
  fetchSessions,
  KBStatus,
  ModelCatalog,
  ModelConfig,
  ModelProviderId,
  SessionSummary,
  SourceInfo,
  TutoringMode,
  uploadKnowledgeFiles,
} from '../lib/api'
import { useChatStore } from '../store/chatStore'

const KnowledgeGraphView = lazy(() =>
  import('./LearningViews').then((module) => ({ default: module.KnowledgeGraphView })),
)
const WrongNotebookView = lazy(() =>
  import('./LearningViews').then((module) => ({ default: module.WrongNotebookView })),
)

const { TextArea } = Input
type WorkspaceView = 'chat' | 'graph' | 'wrongbook'

const providerLabels: Record<ModelProviderId, string> = {
  ollama: '本地',
  lmstudio: 'LM Studio',
  deepseek: 'DeepSeek',
  qwen: '通义千问',
  custom: '自定义 API',
}

const fallbackModelCatalog: ModelCatalog = {
  default: { provider: 'lmstudio', model: 'qwen/qwen3.5-9b' },
  providers: [
    {
      id: 'lmstudio',
      label: '本地 LM Studio',
      description: '通过本机 OpenAI 兼容接口运行模型，数据不离开本机',
      models: ['qwen/qwen3.5-9b'],
      default_model: 'qwen/qwen3.5-9b',
      base_url: 'http://127.0.0.1:1234/v1',
      requires_api_key: false,
      configured: true,
    },
    {
      id: 'ollama',
      label: '本地 Ollama',
      description: '使用本机已安装模型，数据不离开本机',
      models: ['qwen3.5:2b'],
      default_model: 'qwen3.5:2b',
      base_url: 'http://127.0.0.1:11434',
      requires_api_key: false,
      configured: true,
    },
    {
      id: 'deepseek',
      label: 'DeepSeek API',
      description: 'DeepSeek 官方 OpenAI 兼容接口',
      models: ['deepseek-v4-flash', 'deepseek-v4-pro'],
      default_model: 'deepseek-v4-flash',
      base_url: 'https://api.deepseek.com',
      requires_api_key: true,
      configured: false,
    },
    {
      id: 'qwen',
      label: '通义千问 API',
      description: '阿里云百炼 OpenAI 兼容接口',
      models: ['qwen-plus', 'qwen-max', 'qwen-turbo'],
      default_model: 'qwen-plus',
      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      requires_api_key: true,
      configured: false,
    },
    {
      id: 'custom',
      label: '自定义 API',
      description: '连接其他 OpenAI Chat Completions 兼容服务',
      models: [],
      default_model: '',
      base_url: '',
      requires_api_key: true,
      configured: false,
    },
  ],
}

const quickPrompts = [
  {
    icon: <Zap size={19} />,
    eyebrow: '概念答疑',
    title: 'PN 结为什么具有单向导电性？',
    hint: '从势垒与载流子运动解释',
  },
  {
    icon: <BrainCircuit size={19} />,
    eyebrow: '分步计算',
    title: '二极管导通后该如何建立等效电路？',
    hint: '结合恒压降模型进行分析',
  },
  {
    icon: <WandSparkles size={19} />,
    eyebrow: '同类出题',
    title: '根据二极管伏安特性出一道基础题',
    hint: '生成新参数并用 SymPy 验算',
  },
]

function LogoMark() {
  return (
    <span className="logo-mark" aria-hidden="true">
      <span className="logo-node logo-node-a" />
      <span className="logo-node logo-node-b" />
      <span className="logo-node logo-node-c" />
    </span>
  )
}

function sessionTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '历史会话'
  const today = new Date()
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

function Sidebar({
  open,
  onClose,
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  onNewSession,
  activeView,
  onNavigate,
}: {
  open: boolean
  onClose: () => void
  sessions: SessionSummary[]
  activeSessionId: string
  onSelectSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string, title: string) => void
  onNewSession: () => void
  activeView: WorkspaceView
  onNavigate: (view: WorkspaceView) => void
}) {
  const modelProvider = useChatStore((state) => state.modelConfig.provider)
  const streaming = useChatStore((state) => state.streaming)
  return (
    <>
      {open && <button className="sidebar-backdrop" onClick={onClose} aria-label="关闭导航" />}
      <aside className={`sidebar ${open ? 'is-open' : ''}`}>
        <div className="brand-row">
          <LogoMark />
          <div>
            <strong>CircuitMind</strong>
            <span>多智能体电路助教</span>
          </div>
          <button className="mobile-close" onClick={onClose} aria-label="关闭导航">
            <X size={18} />
          </button>
        </div>

        <Button className="new-chat-button" icon={<Plus size={16} />} onClick={onNewSession} disabled={streaming} block>
          开始新对话
        </Button>

        <nav className="main-nav" aria-label="学生端主导航">
          <div className="nav-label">学习空间</div>
          <button className={`nav-item ${activeView === 'chat' ? 'active' : ''}`} onClick={() => onNavigate('chat')}>
            <MessageSquareText size={17} />
            <span>智能学习台</span>
            {activeView === 'chat' && <span className="nav-live-dot" />}
          </button>
          <button className={`nav-item ${activeView === 'graph' ? 'active' : ''}`} onClick={() => onNavigate('graph')}>
            <BookOpen size={17} />
            <span>知识图谱</span>
            <ChevronRight size={14} />
          </button>
          <button className={`nav-item ${activeView === 'wrongbook' ? 'active' : ''}`} onClick={() => onNavigate('wrongbook')}>
            <Layers3 size={17} />
            <span>错题本</span>
            <ChevronRight size={14} />
          </button>
        </nav>

        <div className="recent-section">
          <div className="nav-label">最近学习</div>
          <div className="recent-list">
            {sessions.length ? sessions.map((session) => (
              <div
                key={session.session_id}
                className={`recent-row ${session.session_id === activeSessionId ? 'active' : ''}`}
              >
                <button
                  className="recent-item"
                  onClick={() => onSelectSession(session.session_id)}
                  title={session.title}
                  disabled={streaming}
                >
                  <span className="recent-icon"><Clock3 size={14} /></span>
                  <span>
                    <strong>{session.title}</strong>
                    <small>{sessionTime(session.updated_at)} · {Math.max(1, Math.ceil(session.message_count / 2))} 轮</small>
                  </span>
                </button>
                <Popconfirm
                  title="删除这条历史对话？"
                  description="对话记录和该会话上传的附件将一并删除。"
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                  onConfirm={() => onDeleteSession(session.session_id, session.title)}
                >
                  <button
                    type="button"
                    className="recent-delete"
                    aria-label={`删除历史对话 ${session.title}`}
                    title="删除历史对话"
                    disabled={streaming}
                  >
                    <Trash2 size={14} />
                  </button>
                </Popconfirm>
              </div>
            )) : (
              <div className="recent-empty">完成一次提问后，会话会显示在这里</div>
            )}
          </div>
        </div>

        <div className="sidebar-bottom">
          <Link to="/teacher" className="teacher-link">
            <GraduationCap size={17} />
            <span>切换到教师端</span>
            <ChevronRight size={15} />
          </Link>
          <div className="profile-row">
            <span className="profile-avatar"><UserRound size={17} /></span>
            <span>
              <strong>电路学习者</strong>
              <small>学生端 · {['ollama', 'lmstudio'].includes(modelProvider) ? '本地模型' : '云端模型'}</small>
            </span>
          </div>
        </div>
      </aside>
    </>
  )
}

function Welcome({ onAsk }: { onAsk: (prompt: string) => void }) {
  return (
    <div className="welcome-wrap">
      <div className="welcome-hero">
        <div className="hero-circuit" aria-hidden="true">
          <span className="circuit-line line-a" />
          <span className="circuit-line line-b" />
          <span className="circuit-node node-a" />
          <span className="circuit-node node-b" />
          <span className="circuit-chip"><BrainCircuit size={28} /></span>
        </div>
        <div className="hero-copy">
          <h1>你好，今天想弄懂哪一道电路题？</h1>
        </div>
      </div>
      <div className="quick-grid">
        {quickPrompts.map((item) => (
          <button key={item.title} className="quick-card" onClick={() => onAsk(item.title)}>
            <span className="quick-card-icon">{item.icon}</span>
            <span className="quick-card-copy">
              <small>{item.eyebrow}</small>
              <strong>{item.title}</strong>
              <span>{item.hint}</span>
            </span>
            <ArrowUp className="quick-arrow" size={16} />
          </button>
        ))}
      </div>
      <div className="ability-row">
        <span><Search size={15} /> 混合检索</span>
        <span><Bot size={15} /> 模型推理解答</span>
        <span><Check size={15} /> 答案自动验算</span>
        <span><FileText size={15} /> 来源可追溯</span>
      </div>
    </div>
  )
}

function SourceCard({ source, index }: { source: SourceInfo; index: number }) {
  const page = source.page_start
    ? source.page_start === source.page_end
      ? `第 ${source.page_start} 页`
      : `第 ${source.page_start}–${source.page_end} 页`
    : '结构化题库'
  return (
    <article className="source-card">
      <div className="source-card-top">
        <span className={`source-type ${source.doc_type === 'question' ? 'question' : ''}`}>
          {source.doc_type === 'question' ? <WandSparkles size={13} /> : <FileText size={13} />}
          资料 {index + 1}
        </span>
        <span className="source-score">{Math.round(source.score * 100)}%</span>
      </div>
      <strong>{source.section || source.chapter || source.source}</strong>
      <p>{source.source}</p>
      <div className="source-meta"><span>{page}</span><span>已重排</span></div>
    </article>
  )
}

function KnowledgePanel({ statuses, onCreate }: { statuses: KBStatus[]; onCreate: () => void }) {
  const activeSources = useChatStore((state) => state.activeSources)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const modelProvider = useChatStore((state) => state.modelConfig.provider)
  const messages = useChatStore((state) => state.messages)
  const current = statuses.find((item) => item.id === knowledgeBase)
  const latestAssistant = [...messages].reverse().find((item) => item.role === 'assistant')
  const quizContext = latestAssistant?.agent === '出题 Agent'
  const indexedDocuments = current?.indexed_documents ?? current?.documents ?? 0
  const failedDocuments = current?.failed_documents ?? 0
  return (
    <aside className="knowledge-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">{quizContext ? 'QUIZ CONTEXT' : 'RAG CONTEXT'}</span>
          <h2>{quizContext ? '命题依据' : '检索依据'}</h2>
        </div>
        <Tooltip title={quizContext ? '命题智能体会综合教材定义、结构化题库与原题约束，并经过独立验算' : '检索结果已通过向量、BM25、来源多样化与重排综合评分'}>
          <HelpCircle size={17} />
        </Tooltip>
      </div>

      <div className="kb-summary-card">
        <span className="kb-icon">{quizContext ? <BrainCircuit size={18} /> : <Database size={18} />}</span>
        <div>
          <strong>{quizContext ? '课程依据驱动命题' : knowledgeBase === 'default' ? '默认课程知识库' : knowledgeBase}</strong>
          <span>
            {quizContext
              ? `${activeSources.length} 条教材 / 题库依据 · 已独立验算`
              : failedDocuments
                ? `${current?.documents || 0} 份资料 · ${indexedDocuments} 份可检索 · ${failedDocuments} 份待解析`
                : `${current?.documents || 0} 份资料 · ${current?.questions || 0} 道题 · ${current?.relations || 0} 条关联`}
          </span>
        </div>
        <span className={`kb-state ${quizContext ? 'ready' : current?.state || 'missing'}`}>
          {quizContext ? '已锁定' : current?.state === 'building' ? '构建中' : current?.state === 'ready' ? '就绪' : '待构建'}
        </span>
      </div>

      <div className="source-list">
        {quizContext && !activeSources.length ? (
          <div className="source-empty quiz-reference-empty">
            <span><WandSparkles size={22} /></span>
            <strong>暂无可展示依据</strong>
            <p>命题仍会执行结构检查；完善教材 OCR 后可显示跨教材依据。</p>
          </div>
        ) : activeSources.length ? (
          activeSources.slice(0, 5).map((source, index) => (
            <SourceCard key={`${source.id}-${index}`} source={source} index={index} />
          ))
        ) : (
          <div className="source-empty">
            <span><Search size={22} /></span>
            <strong>等待你的问题</strong>
            <p>提问后，这里会展示命中的教材章节、页码与相关度。</p>
          </div>
        )}
      </div>

      <div className="panel-bottom">
        <button className="manage-kb-button" onClick={onCreate}>
          <UploadCloud size={16} />
          <span>添加教材 / 新建知识库</span>
          <ChevronRight size={15} />
        </button>
        <div className="privacy-note">
          <span className={`privacy-dot ${['ollama', 'lmstudio'].includes(modelProvider) ? '' : 'cloud'}`} />
          {['ollama', 'lmstudio'].includes(modelProvider) ? '资料与模型推理均保留在本机' : '提问内容将发送至所选模型 API'}
        </div>
      </div>
    </aside>
  )
}

function ChatComposer({ onSend }: { onSend: (value: string) => void }) {
  const [value, setValue] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const tutoringMode = useChatStore((state) => state.tutoringMode)
  const setTutoringMode = useChatStore((state) => state.setTutoringMode)
  const streaming = useChatStore((state) => state.streaming)
  const stop = useChatStore((state) => state.stop)
  const pendingAttachments = useChatStore((state) => state.pendingAttachments)
  const addAttachments = useChatStore((state) => state.addAttachments)
  const removeAttachment = useChatStore((state) => state.removeAttachment)
  const hasReadyAttachment = pendingAttachments.some((item) => item.status === 'ready')
  const hasUnfinishedAttachment = pendingAttachments.some((item) => item.status !== 'ready')

  const submit = () => {
    if ((!value.trim() && !hasReadyAttachment) || streaming || hasUnfinishedAttachment) return
    onSend(value)
    setValue('')
  }

  const selectFiles = (files: File[]) => {
    if (!files.length) return
    void addAttachments(files)
  }

  return (
    <div className="composer-shell">
      <div className="composer-card">
        <div className="composer-topline">
          <Segmented<TutoringMode>
            size="small"
            value={tutoringMode}
            onChange={setTutoringMode}
            options={[
              { label: '逐步引导模式', value: 'guided' },
              { label: '完整解答模式', value: 'full' },
            ]}
          />
          <span className="composer-tip">Shift + Enter 换行</span>
        </div>
        {pendingAttachments.length > 0 && (
          <div className="pending-attachments" aria-label="待发送附件">
            {pendingAttachments.map((item) => (
              <div key={item.localId} className={`pending-attachment ${item.status}`}>
                <span className="pending-file-icon">
                  {item.status === 'uploading' ? <LoaderCircle size={15} /> : item.kind === 'image' ? <FileText size={15} /> : <Paperclip size={15} />}
                </span>
                <span className="pending-file-copy">
                  <strong>{item.name}</strong>
                  <small>{item.status === 'uploading' ? '正在上传…' : item.status === 'error' ? item.error : `${Math.max(1, Math.round(item.size / 1024))} KB · 已就绪`}</small>
                </span>
                <button type="button" onClick={() => removeAttachment(item.localId)} aria-label={`移除附件 ${item.name}`}>
                  <X size={13} />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="composer-input-row">
          <input
            ref={fileInputRef}
            className="sr-only-file"
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.webp,.bmp,.pdf,.docx,.txt,.md,.xlsx,.json"
            onChange={(event) => {
              selectFiles(Array.from(event.target.files || []))
              event.target.value = ''
            }}
          />
          <Tooltip title="添加题目图片或附件">
            <Button
              className="attach-button"
              shape="circle"
              onClick={() => fileInputRef.current?.click()}
              disabled={streaming || pendingAttachments.length >= 5}
              icon={<Paperclip size={17} />}
              aria-label="添加题目图片或附件"
            />
          </Tooltip>
          <TextArea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                submit()
              }
            }}
            onPaste={(event) => {
              const files = Array.from(event.clipboardData.files || [])
              if (files.length) {
                event.preventDefault()
                selectFiles(files)
              }
            }}
            autoSize={{ minRows: 1, maxRows: 5 }}
            placeholder="输入题目、解题步骤或你想做的操作，系统会自动识别…"
            variant="borderless"
            aria-label="输入电路问题"
          />
          {streaming ? (
            <Tooltip title="停止生成">
              <Button className="send-button stop" shape="circle" onClick={stop} icon={<CircleStop size={18} />} />
            </Tooltip>
          ) : (
            <Tooltip title="发送">
              <Button
                type="primary"
                className="send-button"
                shape="circle"
                onClick={submit}
                disabled={(!value.trim() && !hasReadyAttachment) || hasUnfinishedAttachment}
                icon={<ArrowUp size={18} />}
              />
            </Tooltip>
          )}
        </div>
      </div>
      <p className="composer-footnote">AI 可能犯错，重要计算请结合教材与实验结果复核。</p>
    </div>
  )
}

function normalizeQuizTitle(content: string) {
  return content.replace(
    /^(#{1,3}\s*同类型新题)(?:\s*[·•・—-]\s*[^\r\n]+)?\s*$/m,
    '$1',
  )
}

function Conversation({ onWrongQuestionSaved }: { onWrongQuestionSaved: () => void }) {
  const messages = useChatStore((state) => state.messages)
  const streaming = useChatStore((state) => state.streaming)
  const stage = useChatStore((state) => state.stage)
  const stageAgent = useChatStore((state) => state.stageAgent)
  const sessionId = useChatStore((state) => state.sessionId)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const send = useChatStore((state) => state.send)
  const [selecting, setSelecting] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [saving, setSaving] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const { message: toast } = AntApp.useApp()

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, stage])

  const toggleMessage = (messageId: string) => {
    setSelectedIds((current) => {
      const next = new Set(current)
      if (next.has(messageId)) next.delete(messageId)
      else next.add(messageId)
      return next
    })
  }

  const cancelSelection = () => {
    setSelecting(false)
    setSelectedIds(new Set())
  }

  const saveWrongQuestion = async () => {
    const selectedMessages = messages.filter((item) => selectedIds.has(item.id) && item.content.trim())
    if (!selectedMessages.length) {
      toast.warning('请至少选择一条对话记录')
      return
    }
    setSaving(true)
    try {
      await createWrongQuestion({
        session_id: sessionId,
        knowledge_base: knowledgeBase,
        messages: selectedMessages.map((item) => ({
          role: item.role,
          content: item.content,
          agent: item.agent,
          model: item.model,
          created_at: item.createdAt,
        })),
      })
      toast.success(`已将 ${selectedMessages.length} 条对话保存为一道错题`)
      cancelSelection()
      onWrongQuestionSaved()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '错题保存失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="conversation">
      {messages.map((message, index) => (
        <div key={message.id} className={`message-row ${message.role} ${selecting ? 'selectable' : ''} ${selectedIds.has(message.id) ? 'selected' : ''}`}>
          {selecting && message.content && (
            <button
              type="button"
              className="message-select-toggle"
              aria-label={selectedIds.has(message.id) ? '取消选择这条对话' : '选择这条对话'}
              aria-pressed={selectedIds.has(message.id)}
              onClick={() => toggleMessage(message.id)}
            >
              {selectedIds.has(message.id) && <Check size={12} />}
            </button>
          )}
          {message.role === 'assistant' && (
            <span className="assistant-avatar"><LogoMark /></span>
          )}
          <div className={`message-bubble ${message.failed ? 'failed' : ''}`}>
            {message.role === 'assistant' && (
              <div className="message-agent">
                <span>{message.agent || (streaming && index === messages.length - 1 ? stageAgent || '多智能体助教' : '多智能体助教')}</span>
                <Tag bordered={false} title={`${providerLabels[message.provider || 'ollama']} · ${message.model || ''}`}>
                  {providerLabels[message.provider || 'lmstudio']} · {message.model || 'qwen/qwen3.5-9b'}
                </Tag>
              </div>
            )}
            {message.attachments?.length ? (
              <div className="message-attachments">
                {message.attachments.map((attachment) =>
                  attachment.kind === 'image' ? (
                    <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer" className="message-image-attachment">
                      <img src={attachment.url} alt={attachment.name} />
                      <span>{attachment.name}</span>
                    </a>
                  ) : (
                    <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer" className="message-file-attachment">
                      <FileText size={16} />
                      <span>{attachment.name}</span>
                    </a>
                  ),
                )}
              </div>
            ) : null}
            {message.content ? (
              message.role === 'assistant'
                ? <MathMarkdown content={normalizeQuizTitle(message.content)} />
                : <p>{message.content}</p>
            ) : (
              <div className="thinking-placeholder">
                <span className="thinking-dots"><i /><i /><i /></span>
                <span>{stage || '正在准备…'}</span>
              </div>
            )}
            {message.role === 'assistant'
              && message.failed
              && message.retryContent
              && !streaming
              && !messages.slice(index + 1).some((item) => item.role === 'assistant' && !item.failed && item.content.trim())
              && (
              <Button
                size="small"
                className="retry-answer-button"
                icon={<RefreshCcw size={13} />}
                onClick={() => void send(message.retryContent!, message.retryAttachmentIds)}
              >
                重新生成本题
              </Button>
            )}
          </div>
        </div>
      ))}
      {streaming && messages.at(-1)?.content && stage && (
        <div className="stage-pill"><span className="thinking-dots"><i /><i /><i /></span>{stageAgent} · {stage}</div>
      )}
      {!streaming && messages.some((message) => message.role === 'assistant' && message.content.trim()) && (
        <div className={`wrong-question-toolbar ${selecting ? 'selecting' : ''}`}>
          {selecting ? (
            <>
              <span>已选 {selectedIds.size} 条，将按当前顺序合并为一道错题</span>
              <button type="button" onClick={cancelSelection}>取消</button>
              <Button size="small" type="primary" loading={saving} disabled={!selectedIds.size} onClick={() => void saveWrongQuestion()}>保存错题</Button>
            </>
          ) : (
            <button type="button" className="start-wrong-selection" onClick={() => setSelecting(true)}>
              <BookmarkPlus size={13} /> 加入错题本
            </button>
          )}
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}

function ModelSettingsModal({
  open,
  onClose,
  catalog,
}: {
  open: boolean
  onClose: () => void
  catalog: ModelCatalog
}) {
  const active = useChatStore((state) => state.modelConfig)
  const setModelConfig = useChatStore((state) => state.setModelConfig)
  const [draft, setDraft] = useState<ModelConfig>(active)
  const { message: toast } = AntApp.useApp()

  useEffect(() => {
    if (open) setDraft(active)
  }, [open, active])

  const provider = catalog.providers.find((item) => item.id === draft.provider)
    || fallbackModelCatalog.providers[0]

  const chooseProvider = (id: ModelProviderId) => {
    const next = catalog.providers.find((item) => item.id === id)
      || fallbackModelCatalog.providers.find((item) => item.id === id)!
    setDraft({
      provider: id,
      model: next.default_model || '',
      apiKey: '',
      baseUrl: next.base_url,
    })
  }

  const applyModel = () => {
    if (!draft.model.trim()) {
      toast.warning('请填写模型名称')
      return
    }
    if (!['ollama', 'lmstudio'].includes(draft.provider) && !draft.baseUrl.trim()) {
      toast.warning('请填写 API Base URL')
      return
    }
    if (provider.requires_api_key && !provider.configured && !draft.apiKey.trim()) {
      toast.warning('请填写 API Key，或在后端环境变量中配置')
      return
    }
    setModelConfig({ ...draft, model: draft.model.trim(), baseUrl: draft.baseUrl.trim() })
    onClose()
    toast.success(`已切换到 ${draft.model.trim()}`)
  }

  const clearSavedApiKey = () => {
    const cleared = { ...active, apiKey: '' }
    setModelConfig(cleared)
    setDraft((value) => ({ ...value, apiKey: '' }))
    toast.success('已清除当前浏览器保存的 API Key')
  }

  const providerIcon = (id: ModelProviderId) => {
    if (id === 'ollama' || id === 'lmstudio') return <Cpu size={18} />
    if (id === 'custom') return <ServerCog size={18} />
    return <Cloud size={18} />
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={null}
      width={650}
      className="model-modal"
    >
      <div className="modal-heading model-modal-heading">
        <span className="modal-icon"><ServerCog size={22} /></span>
        <div>
          <h2>选择与配置模型</h2>
          <p>本地模型可从 LM Studio 或 Ollama 自动读取；云端模型通过 OpenAI 兼容接口接入。</p>
        </div>
      </div>

      <div className="provider-grid" role="radiogroup" aria-label="模型提供商">
        {catalog.providers.map((item) => (
          <button
            type="button"
            role="radio"
            aria-checked={draft.provider === item.id}
            key={item.id}
            className={`provider-card ${draft.provider === item.id ? 'active' : ''}`}
            onClick={() => chooseProvider(item.id)}
          >
            <span className="provider-icon">{providerIcon(item.id)}</span>
            <span>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </span>
            {draft.provider === item.id && <Check size={15} className="provider-check" />}
          </button>
        ))}
      </div>

      <div className="model-config-panel">
        <div className="model-field">
          <label>模型名称</label>
          {['ollama', 'lmstudio'].includes(draft.provider) ? (
            <Select
              value={draft.model}
              options={provider.models.map((model) => ({ value: model, label: model }))}
              onChange={(model) => setDraft((value) => ({ ...value, model }))}
              style={{ width: '100%' }}
              showSearch
              aria-label="选择本地模型"
            />
          ) : (
            <>
              <Input
                value={draft.model}
                onChange={(event) => setDraft((value) => ({ ...value, model: event.target.value }))}
                placeholder="输入模型名称"
                prefix={<Bot size={15} />}
              />
              {provider.models.length > 0 && (
                <div className="suggested-models">
                  {provider.models.map((model) => (
                    <button type="button" key={model} onClick={() => setDraft((value) => ({ ...value, model }))}>
                      {model}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {!['ollama', 'lmstudio'].includes(draft.provider) && (
          <>
            <div className="model-field">
              <label>API Key</label>
              <Input.Password
                value={draft.apiKey}
                onChange={(event) => setDraft((value) => ({ ...value, apiKey: event.target.value }))}
                placeholder={provider.configured ? '后端已配置；留空即可使用' : '保存后在当前浏览器中保留'}
                prefix={<KeyRound size={15} />}
                autoComplete="off"
              />
              {active.provider === draft.provider && active.apiKey && (
                <button type="button" className="clear-api-key" onClick={clearSavedApiKey}>
                  清除已保存的 API Key
                </button>
              )}
            </div>
            <div className="model-field">
              <label>API Base URL</label>
              <Input
                value={draft.baseUrl}
                onChange={(event) => setDraft((value) => ({ ...value, baseUrl: event.target.value }))}
                placeholder="https://example.com/v1"
                prefix={<Cloud size={15} />}
              />
            </div>
          </>
        )}

        <div className={`model-security-note ${['ollama', 'lmstudio'].includes(draft.provider) ? 'local' : 'cloud'}`}>
          <ShieldCheck size={16} />
          <span>
            {['ollama', 'lmstudio'].includes(draft.provider)
              ? '模型在本机运行；题目、检索上下文和回答不会发送到第三方模型服务。'
              : '使用云端模型时，题目、最近对话及检索上下文会发送到所选 API；配置和 API Key 会保存在此浏览器的本地存储中，不写入项目文件。'}
          </span>
        </div>
      </div>

      <div className="model-modal-actions">
        <Button onClick={onClose}>取消</Button>
        <Button type="primary" onClick={applyModel}>应用模型</Button>
      </div>
    </Modal>
  )
}

function StudentPageContent() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeView, setActiveView] = useState<WorkspaceView>('chat')
  const [learningDataVersion, setLearningDataVersion] = useState(0)
  const [kbModalOpen, setKbModalOpen] = useState(false)
  const [modelModalOpen, setModelModalOpen] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [kbFiles, setKbFiles] = useState<UploadFile[]>([])
  const [documentType, setDocumentType] = useState<'auto' | 'textbook' | 'exam' | 'question_bank' | 'notes'>('auto')
  const [ingesting, setIngesting] = useState(false)
  const [statuses, setStatuses] = useState<KBStatus[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog>(fallbackModelCatalog)
  const messages = useChatStore((state) => state.messages)
  const sessionId = useChatStore((state) => state.sessionId)
  const send = useChatStore((state) => state.send)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const setKnowledgeBase = useChatStore((state) => state.setKnowledgeBase)
  const modelConfig = useChatStore((state) => state.modelConfig)
  const streaming = useChatStore((state) => state.streaming)
  const loadSession = useChatStore((state) => state.loadSession)
  const clear = useChatStore((state) => state.clear)
  const { message: toast } = AntApp.useApp()

  const refreshStatuses = async () => {
    try {
      setStatuses(await fetchKnowledgeBases())
    } catch {
      setStatuses([{ id: 'default', state: 'missing', documents: 0, chunks: 0, questions: 0, relations: 0, message: '后端未连接' }])
    }
  }

  const refreshSessions = async () => {
    try {
      setSessions(await fetchSessions())
    } catch {
      setSessions([])
    }
  }

  useEffect(() => {
    void refreshStatuses()
    void fetchSession(sessionId).then((stored) => {
      if (stored.length) loadSession(sessionId, stored)
    }).catch(() => undefined)
    void refreshSessions()
    void fetchModels().then(setModelCatalog).catch(() => setModelCatalog(fallbackModelCatalog))
    const timer = window.setInterval(() => {
      void refreshStatuses()
      void refreshSessions()
    }, 5000)
    return () => window.clearInterval(timer)
  }, [])

  const kbOptions = useMemo(() => {
    const base = statuses.map((item) => ({
      value: item.id,
      label: item.id === 'default' ? '默认课程知识库' : item.id,
    }))
    if (!base.some((item) => item.value === knowledgeBase)) {
      base.push({ value: knowledgeBase, label: knowledgeBase })
    }
    return base
  }, [statuses, knowledgeBase])

  const ask = (prompt: string) => {
    void send(prompt).then(() => refreshSessions())
  }

  const selectHistorySession = async (selectedSessionId: string) => {
    if (selectedSessionId === sessionId) {
      setActiveView('chat')
      setSidebarOpen(false)
      return
    }
    if (streaming) {
      toast.info('当前回答仍在生成，请先停止生成再切换会话')
      return
    }
    try {
      const stored = await fetchSession(selectedSessionId)
      loadSession(selectedSessionId, stored)
      setActiveView('chat')
      setSidebarOpen(false)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '历史会话恢复失败')
    }
  }

  const startNewSession = () => {
    if (streaming) {
      toast.info('当前回答仍在生成，请先停止生成再开始新对话')
      return
    }
    clear()
    setActiveView('chat')
    setSidebarOpen(false)
  }

  const deleteHistorySession = async (deletedSessionId: string, title: string) => {
    if (streaming) {
      toast.info('当前回答仍在生成，请完成或停止后再删除会话')
      return
    }
    try {
      await deleteSession(deletedSessionId)
      setSessions((current) => current.filter((item) => item.session_id !== deletedSessionId))
      if (deletedSessionId === sessionId) {
        clear()
        setSidebarOpen(false)
      }
      toast.success(`已删除“${title}”`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '历史会话删除失败')
    }
  }

  const ingestFiles = async () => {
    if (!kbFiles.length) {
      toast.warning('请先选择教材、试卷或题库文件')
      return
    }
    setIngesting(true)
    try {
      const files = kbFiles.map((item) => (item.originFileObj || item) as File)
      const result = await uploadKnowledgeFiles(files, knowledgeBase, documentType)
      toast.success(result.message)
      setKbFiles([])
      void refreshStatuses()
    } catch (error) {
      const detail = error instanceof Error ? error.message : '上传失败'
      toast.error(detail)
    } finally {
      setIngesting(false)
    }
  }

  const createKnowledgeBase = () => {
    const normalized = newKbName.trim().replace(/\s+/g, '-')
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(normalized)) {
      toast.warning('名称仅支持字母、数字、连字符和下划线')
      return
    }
    setKnowledgeBase(normalized)
    setNewKbName('')
    toast.success(`已切换到新知识库 ${normalized}，请上传第一份资料`)
  }

  return (
    <div className="student-app">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={sessionId}
        onSelectSession={(selectedSessionId) => void selectHistorySession(selectedSessionId)}
        onDeleteSession={(deletedSessionId, title) => void deleteHistorySession(deletedSessionId, title)}
        onNewSession={startNewSession}
        activeView={activeView}
        onNavigate={(view) => { setActiveView(view); setSidebarOpen(false) }}
      />
      <main className="main-workspace">
        <header className="topbar">
          <div className="topbar-left">
            <button className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="打开导航"><Menu size={19} /></button>
            <div>
              <span className="breadcrumb">学生工作台 /</span>
              <strong>{activeView === 'graph' ? '知识图谱' : activeView === 'wrongbook' ? '错题本' : '智能学习台'}</strong>
            </div>
          </div>
          <div className="topbar-actions">
            <Select
              className="kb-select"
              value={knowledgeBase}
              options={kbOptions}
              onChange={setKnowledgeBase}
              suffixIcon={<Database size={14} />}
              aria-label="选择知识库"
            />
            <button
              type="button"
              className="model-badge model-picker-button"
              onClick={() => setModelModalOpen(true)}
              aria-label="选择和配置模型"
            >
              <span className={`online-dot ${['ollama', 'lmstudio'].includes(modelConfig.provider) ? '' : 'cloud'}`} />
              <span>{modelConfig.model}</span>
              <small>{providerLabels[modelConfig.provider]}</small>
              <ChevronDown size={13} />
            </button>
          </div>
        </header>

        {activeView === 'chat' ? (
          <section className="learning-grid">
            <div className="chat-column">
              <div className="chat-scroll">
                {messages.length === 0 ? <Welcome onAsk={ask} /> : (
                  <Conversation onWrongQuestionSaved={() => setLearningDataVersion((value) => value + 1)} />
                )}
              </div>
              <ChatComposer onSend={ask} />
            </div>
            <KnowledgePanel statuses={statuses} onCreate={() => setKbModalOpen(true)} />
          </section>
        ) : (
          <Suspense fallback={<div className="workspace-loading">正在加载学习空间…</div>}>
            {activeView === 'graph' ? (
              <KnowledgeGraphView
                knowledgeBase={knowledgeBase}
                refreshKey={learningDataVersion}
                onOpenWrongBook={() => setActiveView('wrongbook')}
              />
            ) : (
              <WrongNotebookView refreshKey={learningDataVersion} />
            )}
          </Suspense>
        )}
      </main>

      <ModelSettingsModal
        open={modelModalOpen}
        onClose={() => setModelModalOpen(false)}
        catalog={modelCatalog}
      />

      <Modal
        open={kbModalOpen}
        onCancel={() => setKbModalOpen(false)}
        footer={null}
        title={null}
        width={560}
        className="kb-modal"
      >
        <div className="modal-heading">
          <span className="modal-icon"><Database size={22} /></span>
          <div>
            <h2>扩充课程知识库</h2>
            <p>批量导入教材与试卷后，系统会自动分类、抽取题目，并把题目关联到教材知识。</p>
          </div>
        </div>
        <div className="modal-section">
          <label>当前目标知识库</label>
          <Select value={knowledgeBase} options={kbOptions} onChange={setKnowledgeBase} style={{ width: '100%' }} />
        </div>
        <div className="new-kb-row">
          <Input
            value={newKbName}
            onChange={(event) => setNewKbName(event.target.value)}
            placeholder="新知识库英文标识，如 analog-circuits"
            prefix={<Plus size={15} />}
          />
          <Button onClick={createKnowledgeBase}>新建并切换</Button>
        </div>
        <div className="modal-section">
          <label>资料类型</label>
          <Select
            value={documentType}
            onChange={setDocumentType}
            style={{ width: '100%' }}
            options={[
              { value: 'auto', label: '自动识别（推荐）' },
              { value: 'textbook', label: '教材 / 讲义' },
              { value: 'exam', label: '试卷' },
              { value: 'question_bank', label: '结构化题库' },
              { value: 'notes', label: '课程笔记' },
            ]}
          />
        </div>
        <Upload.Dragger
          multiple
          accept=".pdf,.md,.txt,.docx,.xlsx,.json"
          fileList={kbFiles}
          beforeUpload={(file) => {
            setKbFiles((current) => current.length >= 20 ? current : [...current, file])
            return false
          }}
          onRemove={(file) => setKbFiles((current) => current.filter((item) => item.uid !== file.uid))}
          showUploadList={{ showRemoveIcon: true }}
          className="kb-dragger"
        >
          <p className="ant-upload-drag-icon"><UploadCloud size={28} /></p>
          <p className="ant-upload-text">拖入多份教材、试卷或题库，或点击选择</p>
          <p className="ant-upload-hint">每批最多 20 个文件；PDF、Word、Markdown、Excel、JSON，单文件最大 80 MB</p>
        </Upload.Dragger>
        <Button type="primary" block loading={ingesting} disabled={!kbFiles.length} onClick={() => void ingestFiles()}>
          导入并构建知识库 / 题库
        </Button>
        <div className="modal-note">
          <Check size={15} /> 构建结果会保留来源页码、题目候选、教材关联和解析警告
        </div>
      </Modal>
    </div>
  )
}

export default function StudentPage() {
  return (
    <AntApp>
      <StudentPageContent />
    </AntApp>
  )
}
