import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

function unwrapAccidentalProseFences(input: string): string {
  return input.replace(
    /^```(?:text|markdown)?\s*\n([\s\S]*?)\n```\s*$/gim,
    (block, body: string) => {
      const containsChinese = /[\u3400-\u9fff]/.test(body)
      const looksLikeProgram = /(^|\n)\s*(?:import |from |def |class |function |const |let |var |#include|\.[A-Za-z]+\s|[A-Z]\w*\s+\S+\s+\S+)/m.test(body)
      return containsChinese && !looksLikeProgram ? body.trim() : block
    },
  )
}

export function normalizeMarkdownIndentation(input: string): string {
  const lines = unwrapAccidentalProseFences(input).split('\n')
  let insideFence = false
  return lines.map((line) => {
    if (/^\s*```/.test(line)) {
      insideFence = !insideFence
      return line
    }
    if (insideFence || !/^\s{4,}\S/.test(line)) return line

    const body = line.trimStart()
    const isNestedList = /^(?:[-*+] |\d+[.)] )/.test(body)
    const isIndentedProse = /[\u3400-\u9fff]/.test(body) || /\$/.test(body)
    // LLMs often indent explanatory paragraphs by four spaces after a list.
    // CommonMark interprets those lines as code, exposing $...$ and **...**.
    return isIndentedProse && !isNestedList ? body : line
  }).join('\n')
}

export function normalizeLatex(input: string): string {
  let text = normalizeMarkdownIndentation(input)
    .replace(/\r\n?/g, '\n')
    .replace(/＄/g, '$')
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => `\n$$${body.trim()}$$\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, body) => `$${body.trim()}$`)
    .replace(/\\begin\{(?:equation\*?|displaymath)\}([\s\S]*?)\\end\{(?:equation\*?|displaymath)\}/g, (_, body) => `\n$$${body.trim()}$$\n`)
    .replace(/\$\$[ \t]*([\s\S]*?)[ \t]*\$\$/g, (_, body) => `\n$$\n${body.trim()}\n$$\n`)

  const protectedBlocks: string[] = []
  text = text.replace(/\$\$[\s\S]*?\$\$/g, (block) => {
    protectedBlocks.push(block)
    return `@@MATH_BLOCK_${protectedBlocks.length - 1}@@`
  })
  const singleDollarCount = (text.match(/(?<!\\)\$/g) || []).length
  if (singleDollarCount % 2 === 1) text += '$'
  text = text.replace(/@@MATH_BLOCK_(\d+)@@/g, (_, index) => protectedBlocks[Number(index)])
  return text
}

export default function MathMarkdown({ content }: { content: string }) {
  return (
    <div className="math-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false, trust: false }]]}
      >
        {normalizeLatex(content)}
      </ReactMarkdown>
    </div>
  )
}
