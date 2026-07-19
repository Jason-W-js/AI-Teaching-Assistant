import MathMarkdown from './MathMarkdown'
import type { Homework, HomeworkAsset, HomeworkQuestion } from '../lib/api'

type PaperMode = 'questions' | 'answers'

function groupedSections(questions: HomeworkQuestion[]) {
  const sections: Array<{ key: string; title: string; questions: HomeworkQuestion[] }> = []
  const byKey = new Map<string, (typeof sections)[number]>()
  questions.forEach((question) => {
    const key = question.section_key || 'questions'
    let section = byKey.get(key)
    if (!section) {
      section = {
        key,
        title: question.section_title || `${key}、题目`,
        questions: [],
      }
      byKey.set(key, section)
      sections.push(section)
    } else if (section.title === `${key}、题目` && question.section_title) {
      section.title = question.section_title
    }
    section.questions.push(question)
  })
  return sections
}

function QuestionFigures({ figures, label }: { figures: HomeworkAsset[]; label: string }) {
  if (!figures.length) return null
  return (
    <div className={`homework-paper-figures count-${Math.min(figures.length, 3)}`}>
      {figures.map((figure, index) => (
        <figure key={figure.file}>
          <img src={figure.url} alt={`${label}题图 ${index + 1}`} />
        </figure>
      ))}
    </div>
  )
}

function ReflowedQuestion({ question }: { question: HomeworkQuestion }) {
  const position = question.figure_position || 'after_question'
  const figures = question.figures || []
  const options = question.options || []
  return (
    <article className="homework-paper-question">
      <div className="homework-paper-number">{question.number}.</div>
      <div className="homework-paper-question-body">
        {position === 'before_question' && <QuestionFigures figures={figures} label={`第 ${question.number} 题`} />}
        <div className="homework-paper-stem"><MathMarkdown content={question.prompt} /></div>
        {position === 'after_question' && <QuestionFigures figures={figures} label={`第 ${question.number} 题`} />}
        {options.length > 0 && (
          <div className={`homework-paper-options columns-${Math.max(1, Math.min(4, question.option_columns || 1))}`}>
            {options.map((option) => (
              <div key={option.label}>
                <strong>{option.label}.</strong>
                <MathMarkdown content={option.text} />
              </div>
            ))}
          </div>
        )}
        {position === 'after_options' && <QuestionFigures figures={figures} label={`第 ${question.number} 题`} />}
        {question.points > 0 && <span className="homework-paper-points">（{question.points} 分）</span>}
      </div>
    </article>
  )
}

function ReflowedAnswer({ question }: { question: HomeworkQuestion }) {
  return (
    <article className="homework-paper-answer">
      <div className="homework-paper-number">{question.number}.</div>
      <div>
        <div className="homework-paper-answer-content">
          <MathMarkdown content={question.answer || '未识别到标准答案'} />
        </div>
        {question.rubric && (
          <div className="homework-paper-rubric">
            <span>评分标准</span>
            <MathMarkdown content={question.rubric} />
          </div>
        )}
      </div>
      <strong className="homework-paper-answer-score">{question.points || 0} 分</strong>
    </article>
  )
}

export default function HomeworkPaper({
  homework,
  mode,
  printable = false,
}: {
  homework: Homework
  mode: PaperMode
  printable?: boolean
}) {
  const sections = groupedSections(homework.questions)
  const dueDate = homework.due_at ? new Date(homework.due_at) : null
  const deadline = dueDate && !Number.isNaN(dueDate.getTime()) ? dueDate.toLocaleString('zh-CN') : ''
  return (
    <div className={`homework-reflow-paper mode-${mode} ${printable ? 'homework-print-target' : ''}`}>
      <header className="homework-paper-header">
        <span>CIRCUITMIND · {mode === 'questions' ? 'QUESTION PAPER' : 'ANSWER PAPER'}</span>
        <h1>{homework.title}</h1>
        <p>{mode === 'questions' ? homework.instructions || '请按题目要求作答' : '标准答案与评分标准'}</p>
        <div>
          <span>共 {homework.question_count} 题</span>
          <span>满分 {homework.max_score} 分</span>
          {deadline && <span>截止 {deadline}</span>}
        </div>
      </header>
      <main className="homework-paper-content">
        {sections.map((section) => (
          <section className="homework-paper-section" key={section.key}>
            <h2>{section.title}</h2>
            {section.questions.map((question) => mode === 'questions'
              ? <ReflowedQuestion key={question.id} question={question} />
              : <ReflowedAnswer key={question.id} question={question} />)}
          </section>
        ))}
      </main>
    </div>
  )
}
