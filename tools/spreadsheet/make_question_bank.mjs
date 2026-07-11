import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '..', '..')
const outputPath = path.join(root, 'RAG_Resources', '电路课程示例题库.xlsx')
const previewDir = path.join(root, 'tmp', 'spreadsheets')
await fs.mkdir(previewDir, { recursive: true })

const workbook = Workbook.create()
const sheet = workbook.worksheets.add('示例题库')
sheet.showGridLines = false

sheet.getRange('A1:H1').merge()
sheet.getRange('A1').values = [['模拟电子技术 · 第一章示例题库']]
sheet.getRange('A2:H2').merge()
sheet.getRange('A2').values = [['结构化字段用于 RAG 检索与同类题生成；修改后可重新运行知识库构建脚本。']]

const headers = ['题号', '题目文本', '知识点标签', '标准答案', '易错点', '难度', '题型', '解题步骤']
const questions = [
  ['Q001', '为什么本征半导体在绝对零度附近几乎不导电？温度升高后导电能力如何变化？', '本征半导体、载流子、热激发', '绝对零度附近价电子被共价键束缚，几乎没有自由载流子；温度升高会产生电子-空穴对，载流子浓度上升，导电能力增强。', '误认为温度升高会像金属一样使电阻增大；忽略电子与空穴成对产生。', '基础', '简答题', '先判断载流子来源，再比较温度变化对电子-空穴对浓度的影响。'],
  ['Q002', 'N 型半导体整体是否带负电？说明理由。', 'N型半导体、多数载流子、电中性', '不带负电。自由电子是多数载流子，但施主原子电离后形成等量带正电的固定离子，宏观上仍保持电中性。', '把“电子多”直接等同于材料带负电；漏掉固定施主离子。', '基础', '判断题', '区分多数载流子的类型与材料宏观净电荷，两者不是同一概念。'],
  ['Q003', 'P 型半导体中的多数载流子和少数载流子分别是什么？', 'P型半导体、空穴、载流子', '多数载流子为空穴，少数载流子为自由电子。', '把空穴理解成真实带正电粒子；认为 P 型半导体只含空穴。', '基础', '填空题', '根据受主杂质产生空穴，再说明热激发仍会产生少量自由电子。'],
  ['Q004', 'PN 结外加正向电压时，耗尽层宽度和扩散电流如何变化？', 'PN结、正向偏置、耗尽层、扩散电流', '正向电压削弱内建电场，使耗尽层变窄，多数载流子的扩散运动增强，正向电流迅速增大。', '把正向偏置方向接反；混淆扩散电流和漂移电流。', '基础', '简答题', '依次分析外电场与内建电场方向、势垒变化、耗尽层宽度和载流子运动。'],
  ['Q005', 'PN 结外加反向电压且未击穿时，为什么仍有很小的反向电流？', 'PN结、反向偏置、漂移电流、少数载流子', '反向电流主要由热激发产生的少数载流子在电场作用下漂移形成，其数值较小且在一定温度下近似恒定。', '断言反向电流严格为零；错误地认为由多数载流子扩散形成。', '基础', '简答题', '先确认反向偏置使势垒升高，再指出多数载流子被抑制、少数载流子漂移仍存在。'],
  ['Q006', '硅二极管采用恒压降模型，电源 5 V 通过 1 kΩ 电阻和一只正向二极管串联。取导通压降 0.7 V，求电流。', '二极管、恒压降模型、欧姆定律', '$I=(5-0.7)/1000=4.3\\,\\mathrm{mA}$。', '忘记先减去二极管压降；把 kΩ 直接按 Ω 代入；未检查二极管是否正向导通。', '基础', '计算题', '判断二极管导通；用恒压降模型替换二极管；对剩余电压应用欧姆定律。'],
  ['Q007', '电源 3 V、串联电阻 2 kΩ 和硅二极管构成回路，二极管反向连接。忽略反向漏电，求回路电流。', '二极管、反向截止、等效电路', '$I\\approx 0$。', '仍按 0.7 V 恒压降计算；没有根据极性判断工作区。', '基础', '计算题', '先由极性判断二极管反向截止，再用开路模型替代，因此回路无电流。'],
  ['Q008', '稳压二极管正常稳压时应工作在哪个区域？串联限流电阻的作用是什么？', '稳压二极管、反向击穿、限流电阻', '应工作在反向击穿区；限流电阻吸收电源与稳压值之间的电压差，并限制电流，防止稳压管超过最大允许功耗。', '认为稳压管正向导通稳压；忽略限流电阻导致器件过流。', '基础', '简答题', '确定稳压二极管的反向工作方式，再由 KVL 说明多余电压落在限流电阻上。'],
  ['Q009', '某稳压电路输入 12 V，稳压值 6 V，串联电阻 300 Ω，负载电流 10 mA。求稳压管电流。', '稳压二极管、KCL、限流电阻', '电阻电流 $I_R=(12-6)/300=20\\,\\mathrm{mA}$，故 $I_Z=I_R-I_L=10\\,\\mathrm{mA}$。', '只计算电阻电流并把它当作稳压管电流；漏用节点电流定律。', '进阶', '计算题', '先用欧姆定律求限流电阻电流，再在输出节点应用 $I_R=I_Z+I_L$。'],
  ['Q010', '晶体管处于放大区时，发射结和集电结应分别处于什么偏置状态？', '晶体管、放大区、结偏置', '发射结正向偏置，集电结反向偏置。', '把集电结也设为正向偏置，导致进入饱和区；只背结论而不会结合电位判断。', '基础', '填空题', '从载流子注入与收集两个过程判断两个 PN 结的偏置状态。'],
  ['Q011', '某 NPN 晶体管 $\\beta=100$，基极电流 $I_B=20\\,\\mu A$，且工作在放大区。估算集电极电流。', '晶体管、电流放大系数、放大区', '$I_C=\\beta I_B=2\\,\\mathrm{mA}$。', '把 $\\mu A$ 与 $mA$ 的换算弄错；未注意题目已限定放大区。', '基础', '计算题', '使用放大区近似关系 $I_C=\\beta I_B$，再完成微安到毫安的单位换算。'],
  ['Q012', '比较场效应管与双极型晶体管的控制方式及输入电阻特点。', '场效应管、晶体管、电压控制、电流控制', '场效应管主要由栅源电压控制漏极电流，输入电阻很高；双极型晶体管由基极电流控制集电极电流，输入电阻通常较低。', '把两者都称为电压控制器件；忽略 MOS 管栅极绝缘带来的高输入电阻。', '进阶', '比较题', '分别指出控制量、被控制量和输入端电流，再归纳输入电阻差异。'],
]

sheet.getRange('A3:H3').values = [headers]
sheet.getRange(`A4:H${questions.length + 3}`).values = questions

sheet.getRange('A1:H1').format = {
  fill: '#0B5E5A',
  font: { bold: true, color: '#FFFFFF', size: 18 },
  verticalAlignment: 'center',
  horizontalAlignment: 'left',
}
sheet.getRange('A1:H1').format.rowHeight = 34
sheet.getRange('A2:H2').format = {
  fill: '#E7F3EF',
  font: { color: '#54726F', italic: true, size: 10 },
  verticalAlignment: 'center',
}
sheet.getRange('A2:H2').format.rowHeight = 25
sheet.getRange('A3:H3').format = {
  fill: '#D3EAE4',
  font: { bold: true, color: '#164844' },
  horizontalAlignment: 'center',
  verticalAlignment: 'center',
  borders: { preset: 'doubleBottom', style: 'thin', color: '#7BAAA1' },
}
sheet.getRange('A3:H3').format.rowHeight = 26

const dataRange = sheet.getRange(`A4:H${questions.length + 3}`)
dataRange.format = {
  font: { color: '#294846', size: 10 },
  verticalAlignment: 'top',
  wrapText: true,
  borders: {
    insideHorizontal: { style: 'thin', color: '#DCE7E4' },
    bottom: { style: 'thin', color: '#C7D7D3' },
  },
}
dataRange.format.rowHeight = 58
sheet.getRange(`A4:A${questions.length + 3}`).format = {
  font: { bold: true, color: '#0F766E' },
  horizontalAlignment: 'center',
  verticalAlignment: 'top',
}
sheet.getRange(`F4:G${questions.length + 3}`).format.horizontalAlignment = 'center'

const widths = [10, 42, 27, 38, 39, 10, 12, 42]
for (let index = 0; index < widths.length; index += 1) {
  sheet.getRangeByIndexes(0, index, questions.length + 3, 1).format.columnWidth = widths[index]
}

sheet.freezePanes.freezeRows(3)
const table = sheet.tables.add(`A3:H${questions.length + 3}`, true, 'CircuitQuestionBank')
table.style = 'TableStyleMedium2'
table.showBandedColumns = false
table.showFilterButton = true

const inspect = await workbook.inspect({
  kind: 'table',
  range: `示例题库!A1:H${questions.length + 3}`,
  include: 'values,formulas',
  tableMaxRows: 16,
  tableMaxCols: 8,
  maxChars: 12000,
})
const errors = await workbook.inspect({
  kind: 'match',
  searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',
  options: { useRegex: true, maxResults: 100 },
  summary: 'final formula error scan',
})
const preview = await workbook.render({
  sheetName: '示例题库',
  range: 'A1:H15',
  scale: 1,
  format: 'png',
})
await fs.writeFile(path.join(previewDir, 'question-bank.png'), new Uint8Array(await preview.arrayBuffer()))

const output = await SpreadsheetFile.exportXlsx(workbook)
await output.save(outputPath)
console.log(JSON.stringify({ outputPath, previewPath: path.join(previewDir, 'question-bank.png'), inspect: inspect.ndjson, errors: errors.ndjson }, null, 2))
