from __future__ import annotations

from typing import TypedDict


class ValidatedKnowledgeCard(TypedDict):
    definition: str
    key_points: tuple[str, ...]


# Short, foundational definitions are curated here because formula-heavy OCR is
# not reliable enough to be displayed as authoritative course content.  The
# graph still keeps every imported source and chapter association for evidence.
VALIDATED_KNOWLEDGE_CARDS: dict[str, ValidatedKnowledgeCard] = {
    "叠加定理": {
        "definition": (
            "在线性电路中，多个独立电源共同作用时，任一支路中的电压或电流，"
            "等于各独立电源单独作用时在该支路产生的电压或电流的代数和。\n\n"
            "$$u=\\sum_{k=1}^{n}u_k,\\qquad i=\\sum_{k=1}^{n}i_k$$"
        ),
        "key_points": (
            "分析某一独立电源单独作用时，其余独立电压源置零并以短路代替，其余独立电流源置零并以开路代替；受控源保留。",
            "叠加定理只适用于线性电路的电压和电流，功率不能直接叠加。",
        ),
    },
    "基尔霍夫电流定律": {
        "definition": (
            "在任一时刻，流入任一节点的电流代数和等于零。\n\n"
            "$$\\sum_{k=1}^{n} i_k=0$$"
        ),
        "key_points": (
            "列方程前应先统一规定各支路电流的参考方向。",
            "该定律反映电荷守恒，与元件的具体性质无关。",
        ),
    },
    "基尔霍夫电压定律": {
        "definition": (
            "在任一时刻，沿任一闭合回路各段电压的代数和等于零。\n\n"
            "$$\\sum_{k=1}^{n} u_k=0$$"
        ),
        "key_points": (
            "列方程时应统一回路绕行方向，并按电压参考极性确定正负号。",
            "该定律反映能量守恒，适用于集总参数电路。",
        ),
    },
    "欧姆定律": {
        "definition": "在线性电阻元件中，电压与电流成正比。\n\n$$u=Ri$$",
        "key_points": (
            "公式中的电压与电流应采用关联参考方向；若采用非关联方向，表达式为 $u=-Ri$。",
            "电阻 $R$ 的单位是欧姆（$\\Omega$）。",
        ),
    },
    "戴维南定理": {
        "definition": (
            "任何一个线性含源二端网络，对外电路都可等效为一个电压源与一个电阻串联的支路。"
        ),
        "key_points": (
            "等效电压等于二端网络的开路电压。",
            "等效电阻可由端口输入电阻求得；含受控源时应保留受控源并采用外加测试源。",
        ),
    },
    "诺顿定理": {
        "definition": (
            "任何一个线性含源二端网络，对外电路都可等效为一个电流源与一个电阻并联的支路。"
        ),
        "key_points": (
            "等效电流等于二端网络的短路电流。",
            "诺顿等效电阻与戴维南等效电阻相同。",
        ),
    },
    "负反馈对输入电阻的影响": {
        "definition": (
            "输入电阻是从放大电路输入端看进去的等效电阻。负反馈对输入电阻的影响，"
            "取决于反馈网络与基本放大电路在输入端采用串联还是并联连接。"
        ),
        "key_points": (
            "串联负反馈使反馈环内的输入电阻增大；在基本模型中，闭环输入电阻约为开环输入电阻的 $1+AF$ 倍。",
            "并联负反馈使反馈环内的输入电阻减小；这一结论不受输出端采用电压取样还是电流取样的影响。",
        ),
    },
    "场效应管有源电阻及电流源电路": {
        "definition": (
            "在集成电路中，可利用工作在特定区域的场效应管代替普通电阻或构成恒流偏置单元，"
            "形成有源电阻和电流源电路。"
        ),
        "key_points": (
            "场效应管有源负载能够提供较高的等效交流电阻，从而提高单级放大电路的电压增益。",
            "场效应管电流源可提供较稳定的静态工作点，并有利于提高集成度和输出动态范围。",
        ),
    },
}
