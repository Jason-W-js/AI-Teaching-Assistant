import { Link } from 'react-router-dom'
import { ArrowLeft, BarChart3, BookOpenCheck, BrainCircuit, Construction, Database, UsersRound } from 'lucide-react'

const futureModules = [
  { icon: <UsersRound />, title: '班级与学生', text: '查看学习进度、薄弱知识点与个性化建议。' },
  { icon: <BookOpenCheck />, title: '课程与题库', text: '维护课程章节、题目标签、标准答案与易错点。' },
  { icon: <BarChart3 />, title: '教学分析', text: '汇总高频问题、检索命中率与学习成效。' },
  { icon: <Database />, title: '知识库管理', text: '审核资料清洗结果、Chunk 元数据与索引版本。' },
]

export default function TeacherPage() {
  return (
    <div className="teacher-page">
      <header className="teacher-topbar">
        <div className="brand-row teacher-brand"><span className="teacher-brand-icon"><BrainCircuit size={22} /></span><div><strong>CircuitMind</strong><span>教师工作台</span></div></div>
        <Link to="/student" className="back-student"><ArrowLeft size={16} /> 返回学生端</Link>
      </header>
      <main className="teacher-main">
        <section className="teacher-hero">
          <span className="teacher-status"><Construction size={15} /> 接口已预留</span>
          <h1>教师工作台将在下一阶段开放</h1>
          <p>当前版本优先跑通学生端答疑、同类出题、知识库扩充和多轮记忆。教师端路由与服务接口已经保留。</p>
        </section>
        <section className="future-grid">
          {futureModules.map((module) => (
            <article key={module.title} className="future-card">
              <span>{module.icon}</span>
              <h2>{module.title}</h2>
              <p>{module.text}</p>
              <small>后续版本</small>
            </article>
          ))}
        </section>
      </main>
    </div>
  )
}

