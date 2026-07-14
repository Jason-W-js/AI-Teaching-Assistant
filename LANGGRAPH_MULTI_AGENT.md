# CircuitMind LangGraph 多智能体工作流

本文档对应当前学生端的实际 LangGraph 编排。总图先结合最近对话与附件完成意图路由：概念答疑和上下文追问走轻量直答链路，明确的求解或步骤诊断才进入四智能体子图，同类出题进入独立出题子图，学习规划进入学情—资料—计划子图；寒暄、离题内容、噪声及不完整指令由会话引导链处理，不触发检索和解题。

```mermaid
flowchart TB
    Student["学生 Web 页面"] --> API["POST /api/chat<br/>SSE 流式接口"]

    API --> Memory["会话记忆<br/>Redis / 本地持久化"]
    API --> AttachmentStore["附件存储<br/>图片 / PDF / Word / Excel"]

    subgraph Orchestrator["LangGraph 总编排图"]
        direction TB
        AttachmentReader["附件理解 Agent<br/>题干、节点—支路拓扑、已知量、待求量"]
        IntentRouter{"意图路由 Agent"}
        DirectQA["上下文答疑 Agent<br/>概念解释 / 结果追问"]
        AnswerEntry["过程解题 Agent 子图"]
        QuizEntry["出题 Agent 子图"]
        PlanEntry["学习规划 Agent 子图"]
        Conversation["会话引导 Agent<br/>寒暄 / 离题 / 噪声 / 补问"]

        AttachmentReader --> IntentRouter
        IntentRouter -->|"概念答疑 / 上下文追问"| DirectQA
        IntentRouter -->|"求解 / 步骤诊断"| AnswerEntry
        IntentRouter -->|"同类出题"| QuizEntry
        IntentRouter -->|"复习 / 学习规划"| PlanEntry
        IntentRouter -->|"非任务 / 信息不足"| Conversation
    end

    AttachmentStore --> AttachmentReader
    Memory --> AttachmentReader
    VisionModel["LM Studio Qwen3.5<br/>本地视觉理解"] --> AttachmentReader

    subgraph AnswerGraph["过程型解题辅导工作流"]
        direction LR
        Understand["题目理解智能体<br/>多模态结构化"]
        AnswerRetrieve["有依据的知识关联<br/>向量 + BM25 + Rerank + 质量门控"]
        Solve["领域求解智能体<br/>内部参考解"]
        Diagnose["验证与错因诊断智能体<br/>参考解 + SymPy + 学生步骤"]
        Tutor["教学辅导智能体<br/>L1-L5 答案释放策略"]

        Understand --> AnswerRetrieve --> Solve --> Diagnose --> Tutor
    end

    AnswerEntry --> Understand
    DirectQA --> Stream
    Conversation --> Stream

    subgraph QuizGraph["同类出题 Agent 工作流"]
        direction TB
        Extract["原题分析 Agent<br/>知识点 + 题型 + 结构蓝图"]
        QuizRetrieve["命题检索器<br/>教材定义 + 相似例题"]
        Generate["出题 Agent<br/>保持拓扑与设问同构"]
        Verify{"验算与校验 Agent<br/>结构 / 去重 / SymPy"}
        Repair["修正 Agent<br/>生成同构可验证变式"]
        VerifyRepair{"二次校验"}
        Render["结果渲染<br/>题目、思路、答案、易错点"]

        Extract --> QuizRetrieve --> Generate --> Verify
        Verify -->|"通过"| Render
        Verify -->|"未通过"| Repair --> VerifyRepair --> Render
    end

    QuizEntry --> Extract

    subgraph PlanGraph["学习规划 Agent 工作流"]
        direction LR
        Profile["学情提取 Agent<br/>目标 / 时间 / 薄弱点"]
        PlanRetrieve["课程资料检索<br/>知识点 / 前置关系"]
        PlanGenerate["计划生成 Agent<br/>阶段任务 / 检查点"]
        Profile --> PlanRetrieve --> PlanGenerate
    end

    PlanEntry --> Profile
    PlanGenerate --> Stream

    subgraph RAG["课程知识库"]
        direction TB
        CleanDocs["清洗后的教材与结构化题库"]
        Chunks["带章、节、页码元数据的 Chunk"]
        VectorDB["FAISS 向量索引"]
        BM25["BM25 关键词索引"]

        CleanDocs --> Chunks
        Chunks --> VectorDB
        Chunks --> BM25
    end

    VectorDB --> AnswerRetrieve
    BM25 --> AnswerRetrieve
    VectorDB --> QuizRetrieve
    BM25 --> QuizRetrieve

    ModelGateway["模型网关<br/>默认 Qwen3.5-9B / 可按智能体异构"] --> Understand
    ModelGateway --> DirectQA
    ModelGateway --> Conversation
    ModelGateway --> Solve
    ModelGateway --> Diagnose
    ModelGateway --> Tutor
    ModelGateway --> Generate
    ModelGateway --> Profile
    ModelGateway --> PlanGenerate
    SymPy["Python / SymPy<br/>数值合理性验算"] --> Verify
    SymPy --> VerifyRepair

    Tutor --> Stream["SSE: status / delta / meta / done"]
    Render --> Stream
    Stream --> Student

    classDef agent fill:#e6f4f1,stroke:#0f766e,color:#173f3c,stroke-width:1.5px;
    classDef decision fill:#fff6df,stroke:#c58a18,color:#5e4412,stroke-width:1.5px;
    classDef storage fill:#edf2ff,stroke:#526fa8,color:#273d68,stroke-width:1.2px;
    classDef model fill:#f5ecff,stroke:#8057a6,color:#4b2c68,stroke-width:1.2px;

    class AttachmentReader,DirectQA,AnswerEntry,QuizEntry,PlanEntry,Conversation,Understand,AnswerRetrieve,Solve,Diagnose,Tutor,Extract,QuizRetrieve,Generate,Repair,Render,Profile,PlanRetrieve,PlanGenerate agent;
    class IntentRouter,Verify,VerifyRepair decision;
    class Memory,AttachmentStore,CleanDocs,Chunks,VectorDB,BM25 storage;
    class VisionModel,ModelGateway,SymPy model;
```

## 关键状态流转

LangGraph 状态中主要保存以下信息：

- `message`、`history`、`knowledge_base`：学生输入、最近对话和当前知识库。
- `attachment_context`、`attachment_blueprint`：附件识别文本以及电路拓扑、已知量、待求量蓝图。
- `intent`：路由结果，取值为 `qa`、`answer`、`quiz`、`plan` 或 `chat`；只有 `answer` 执行完整解题链，`chat` 不检索课程资料也不改写题目状态。
- `problem_analysis`：题型、知识点、已知量、待求量、拓扑和信息完整性。
- `reference_solution`：仅在后端保存的内部方法、计划、检查点、推导和最终答案。
- `diagnosis`、`verification`：学生步骤的结构化错因与 SymPy 数值链校验。
- `tutor_action`、`hint_level`：当前教学动作以及 L1–L5 答案释放级别。
- `knowledge_point`、`quiz_type`、`quiz_family`：出题知识点、数值/概念题型和同构题家族。
- `plan_profile`：学习目标、可用时间、薄弱知识点及计划约束。
- `draft`、`verification`：生成题草稿以及结构、去重和 SymPy 校验结果。
- `response`、`sources`：最终回复和可追溯资料来源。
