import { useEffect, useMemo, useState } from 'react'
import { App as AntApp, Button, Empty, Input, Popconfirm, Select, Spin, Tag } from 'antd'
import {
  BookOpen,
  ChevronRight,
  Edit3,
  FileText,
  FolderPlus,
  Layers3,
  Search,
  Trash2,
} from 'lucide-react'
import MathMarkdown from '../components/MathMarkdown'
import {
  createWrongQuestionCategory,
  deleteWrongQuestion,
  fetchKnowledgeGraph,
  fetchWrongNotebook,
  KnowledgeGraph,
  KnowledgeGraphNode,
  updateWrongQuestion,
  WrongNotebook,
  WrongQuestion,
} from '../lib/api'

const EMPTY_NOTEBOOK: WrongNotebook = { categories: [], items: [] }

function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? '最近更新'
    : date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

export function WrongNotebookView({ refreshKey = 0 }: { refreshKey?: number }) {
  const [notebook, setNotebook] = useState<WrongNotebook>(EMPTY_NOTEBOOK)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [categoryId, setCategoryId] = useState('all')
  const [selected, setSelected] = useState<WrongQuestion | null>(null)
  const [newCategory, setNewCategory] = useState('')
  const [renamingId, setRenamingId] = useState('')
  const [renameValue, setRenameValue] = useState('')
  const { message: toast } = AntApp.useApp()

  const load = async () => {
    setLoading(true)
    try {
      const value = await fetchWrongNotebook()
      setNotebook(value)
      if (selected) setSelected(value.items.find((item) => item.id === selected.id) || null)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '错题本读取失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [refreshKey])

  const counts = useMemo(() => {
    const value: Record<string, number> = { all: notebook.items.length }
    notebook.items.forEach((item) => { value[item.category_id] = (value[item.category_id] || 0) + 1 })
    return value
  }, [notebook.items])

  const visibleItems = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return notebook.items.filter((item) => {
      if (categoryId !== 'all' && item.category_id !== categoryId) return false
      if (!needle) return true
      return [
        item.title,
        item.knowledge_points.join(' '),
        item.messages.map((message) => message.content).join(' '),
      ].join(' ').toLocaleLowerCase().includes(needle)
    })
  }, [notebook.items, categoryId, query])

  const addCategory = async () => {
    const name = newCategory.trim()
    if (!name) return
    try {
      await createWrongQuestionCategory(name)
      setNewCategory('')
      await load()
      toast.success(`已创建分类“${name}”`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '分类创建失败')
    }
  }

  const rename = async (item: WrongQuestion) => {
    const title = renameValue.trim()
    if (!title) return
    try {
      await updateWrongQuestion(item.id, { title })
      setRenamingId('')
      await load()
      toast.success('题目已重命名')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '重命名失败')
    }
  }

  const move = async (item: WrongQuestion, nextCategory: string) => {
    try {
      await updateWrongQuestion(item.id, { category_id: nextCategory })
      await load()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '分类更新失败')
    }
  }

  const remove = async (item: WrongQuestion) => {
    try {
      await deleteWrongQuestion(item.id)
      if (selected?.id === item.id) setSelected(null)
      await load()
      toast.success('已从错题本删除')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '删除失败')
    }
  }

  return (
    <section className="workspace-view wrongbook-view">
      <header className="workspace-hero">
        <span className="workspace-hero-icon"><Layers3 size={22} /></span>
        <div>
          <span className="panel-kicker">MISTAKE NOTEBOOK</span>
          <h1>错题本</h1>
          <p>把多条对话作为一道完整错题保存，按分类和知识点快速回顾。</p>
        </div>
        <div className="workspace-stat"><strong>{notebook.items.length}</strong><span>道错题</span></div>
      </header>

      <div className="wrongbook-layout">
        <aside className="wrongbook-categories">
          <button className={categoryId === 'all' ? 'active' : ''} onClick={() => setCategoryId('all')}>
            <span>全部错题</span><small>{counts.all || 0}</small>
          </button>
          {notebook.categories.map((category) => (
            <button key={category.id} className={categoryId === category.id ? 'active' : ''} onClick={() => setCategoryId(category.id)}>
              <span>{category.name}</span><small>{counts[category.id] || 0}</small>
            </button>
          ))}
          <div className="new-category-box">
            <Input
              size="small"
              value={newCategory}
              onChange={(event) => setNewCategory(event.target.value)}
              onPressEnter={() => void addCategory()}
              placeholder="新分类名称"
            />
            <Button size="small" icon={<FolderPlus size={14} />} onClick={() => void addCategory()} aria-label="创建分类" />
          </div>
        </aside>

        <div className="wrongbook-list-panel">
          <Input
            allowClear
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            prefix={<Search size={15} />}
            placeholder="搜索题目、知识点或对话内容"
            className="workspace-search"
          />
          {loading ? <div className="view-loading"><Spin /></div> : visibleItems.length ? (
            <div className="wrongbook-cards">
              {visibleItems.map((item) => (
                <article key={item.id} className={`wrongbook-card ${selected?.id === item.id ? 'active' : ''}`}>
                  <div
                    className="wrongbook-card-main"
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelected(item)}
                    onKeyDown={(event) => { if (event.key === 'Enter') setSelected(item) }}
                  >
                    <span className="wrongbook-file"><FileText size={16} /></span>
                    <span>
                      {renamingId === item.id ? (
                        <Input
                          size="small"
                          autoFocus
                          value={renameValue}
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) => setRenameValue(event.target.value)}
                          onPressEnter={() => void rename(item)}
                          onBlur={() => void rename(item)}
                        />
                      ) : <strong>{item.title}</strong>}
                      <small>{item.messages.length} 条对话 · {formatDate(item.updated_at)}</small>
                    </span>
                    <ChevronRight size={15} />
                  </div>
                  <div className="wrongbook-card-meta">
                    <div>{item.knowledge_points.length ? item.knowledge_points.map((point) => <Tag key={point}>{point}</Tag>) : <Tag>待归类知识点</Tag>}</div>
                    <div className="wrongbook-card-actions">
                      <Select
                        size="small"
                        value={item.category_id}
                        options={notebook.categories.map((category) => ({ value: category.id, label: category.name }))}
                        onChange={(value) => void move(item, value)}
                        aria-label="修改错题分类"
                      />
                      <button onClick={() => { setRenamingId(item.id); setRenameValue(item.title) }} aria-label="重命名"><Edit3 size={13} /></button>
                      <Popconfirm title="从错题本删除这道题？" okText="删除" cancelText="取消" onConfirm={() => void remove(item)}>
                        <button aria-label="删除错题"><Trash2 size={13} /></button>
                      </Popconfirm>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="这个分类还没有错题" />}
        </div>

        <aside className="wrongbook-detail">
          {selected ? (
            <>
              <span className="panel-kicker">SELECTED RECORD</span>
              <h2>{selected.title}</h2>
              <div className="wrongbook-detail-tags">
                {selected.knowledge_points.map((point) => <Tag key={point}>{point}</Tag>)}
              </div>
              <div className="wrongbook-dialogue">
                {selected.messages.map((message, index) => (
                  <div key={`${message.role}-${index}`} className={`wrongbook-message ${message.role}`}>
                    <small>{message.role === 'user' ? '我的提问 / 步骤' : message.agent || 'AI 助教'}</small>
                    {message.role === 'assistant' ? <MathMarkdown content={message.content} /> : <p>{message.content}</p>}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="detail-empty"><Layers3 size={28} /><strong>选择一道错题</strong><p>这里会完整展示当时保存的多条对话。</p></div>
          )}
        </aside>
      </div>
    </section>
  )
}

export function KnowledgeGraphView({
  knowledgeBase,
  refreshKey = 0,
  onOpenWrongBook,
}: {
  knowledgeBase: string
  refreshKey?: number
  onOpenWrongBook: () => void
}) {
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [categoryId, setCategoryId] = useState('all')
  const [selected, setSelected] = useState<KnowledgeGraphNode | null>(null)
  const { message: toast } = AntApp.useApp()

  useEffect(() => {
    setLoading(true)
    void fetchKnowledgeGraph(knowledgeBase)
      .then((value) => {
        setGraph(value)
        if (selected) setSelected(value.nodes.find((node) => node.id === selected.id) || null)
      })
      .catch((error) => toast.error(error instanceof Error ? error.message : '知识图谱读取失败'))
      .finally(() => setLoading(false))
  }, [knowledgeBase, refreshKey])

  const visibleNodes = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return (graph?.nodes || []).filter((node) => {
      if (categoryId !== 'all' && node.category_id !== categoryId) return false
      if (!needle) return true
      return [node.label, node.summary, node.definition, node.key_points.join(' '), node.sections.join(' '), node.sources.map((source) => source.name).join(' ')]
        .join(' ').toLocaleLowerCase().includes(needle)
    })
  }, [graph, categoryId, query])

  if (loading) return <div className="workspace-view view-loading"><Spin size="large" /></div>
  if (!graph) return <div className="workspace-view"><Empty description="知识图谱暂时不可用" /></div>

  return (
    <section className="workspace-view graph-view">
      <header className="workspace-hero">
        <span className="workspace-hero-icon"><BookOpen size={22} /></span>
        <div>
          <span className="panel-kicker">COURSE KNOWLEDGE GRAPH</span>
          <h1>{graph.root.label}</h1>
          <p>以知识点为中心合并不同教材、题库与错题，避免同一概念重复建节点。</p>
        </div>
        <div className="graph-stats">
          <span><strong>{graph.stats.sources}</strong>份资料</span>
          <span><strong>{graph.stats.knowledge_points}</strong>个知识点</span>
          <span><strong>{graph.stats.wrong_questions}</strong>道错题</span>
        </div>
      </header>

      <div className="graph-toolbar">
        <Input allowClear value={query} onChange={(event) => setQuery(event.target.value)} prefix={<Search size={15} />} placeholder="搜索知识点、章节或教材" />
        <div className="graph-category-tabs">
          <button className={categoryId === 'all' ? 'active' : ''} onClick={() => setCategoryId('all')}>全部</button>
          {graph.categories.map((category) => (
            <button key={category.id} className={categoryId === category.id.replace('category-', '') ? 'active' : ''} onClick={() => setCategoryId(category.id.replace('category-', ''))}>
              {category.label}<small>{category.count}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="graph-layout">
        <div className="graph-canvas">
          <div className="graph-root-node"><span><BookOpen size={18} /></span><strong>{graph.root.label}</strong></div>
          <div className="graph-category-columns">
            {graph.categories
              .filter((category) => categoryId === 'all' || category.id === `category-${categoryId}`)
              .map((category) => {
                const allCategoryNodes = visibleNodes.filter((node) => `category-${node.category_id}` === category.id)
                const nodes = categoryId === 'all' && !query.trim() ? allCategoryNodes.slice(0, 12) : allCategoryNodes
                if (!nodes.length) return null
                return (
                  <section key={category.id} className="graph-branch">
                    <div className="graph-category-node"><strong>{category.label}</strong><small>{allCategoryNodes.length} 节点</small></div>
                    <div className="graph-node-grid">
                      {nodes.map((node) => (
                        <button key={node.id} className={`graph-knowledge-node ${selected?.id === node.id ? 'active' : ''}`} onClick={() => setSelected(node)}>
                          <span className="graph-node-dot" />
                          <strong>{node.label}</strong>
                          <small>{node.sources.length} 来源 · {node.questions.length} 例题</small>
                          {node.wrong_questions.length > 0 && <em>{node.wrong_questions.length} 错题</em>}
                        </button>
                      ))}
                    </div>
                    {allCategoryNodes.length > nodes.length && (
                      <button className="graph-expand-category" onClick={() => setCategoryId(category.id.replace('category-', ''))}>
                        查看本类全部 {allCategoryNodes.length} 个知识点 <ChevronRight size={13} />
                      </button>
                    )}
                  </section>
                )
              })}
          </div>
          {!visibleNodes.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有匹配的知识点" />}
        </div>

        <aside className="graph-detail">
          {selected ? (
            <>
              <span className="panel-kicker">KNOWLEDGE NODE</span>
              <h2>{selected.label}</h2>
              <Tag>{selected.category_label}</Tag>
              {selected.definition && (
                <div className="graph-definition">
                  <h3>概念定义</h3>
                  <MathMarkdown content={selected.definition} />
                </div>
              )}
              <div className="graph-detail-section graph-key-points">
                <h3>知识要点</h3>
                {selected.key_points.length ? (
                  <ul>{selected.key_points.map((point) => <li key={point}><MathMarkdown content={point} /></li>)}</ul>
                ) : <div className="graph-summary"><MathMarkdown content={selected.summary} /></div>}
              </div>
              <div className="graph-detail-section">
                <h3>教材来源</h3>
                {selected.sources.length ? selected.sources.map((source) => (
                  <div key={source.name} className="graph-source-row"><FileText size={14} /><span>{source.name}</span><small>{source.chunks} 片段</small></div>
                )) : <p>暂无教材来源，该节点来自错题标签。</p>}
              </div>
              {selected.sections.length > 0 && (
                <div className="graph-detail-section"><h3>相关章节</h3><ul>{selected.sections.slice(0, 6).map((section) => <li key={section}>{section}</li>)}</ul></div>
              )}
              <div className="graph-detail-section">
                <h3>关联错题</h3>
                {selected.wrong_questions.length ? selected.wrong_questions.map((item) => (
                  <button key={item.id} className="graph-wrong-link" onClick={onOpenWrongBook}><Layers3 size={14} /><span>{item.title}</span><ChevronRight size={14} /></button>
                )) : <p>还没有错题挂接到这个知识点。</p>}
              </div>
            </>
          ) : (
            <div className="detail-empty"><BookOpen size={28} /><strong>点击知识节点</strong><p>可查看概要、教材来源、相关章节和错题。</p></div>
          )}
        </aside>
      </div>
    </section>
  )
}
