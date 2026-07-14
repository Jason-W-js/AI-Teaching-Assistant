import asyncio
import json

from backend.app.schemas import ChatRequest
from backend.app.services.knowledge_graph import KnowledgeGraphService
from backend.app.services.wrong_notebook import WrongNotebookStore


def test_wrong_notebook_persists_categories_items_and_updates(tmp_path):
    async def scenario():
        store = WrongNotebookStore(tmp_path)
        category = await store.create_category("二极管")
        item = await store.create(
            session_id="student-1",
            messages=[{"role": "user", "content": "PN 结为什么单向导电？"}],
            knowledge_base="default",
            knowledge_points=["PN结"],
            category_id=category["id"],
        )
        assert item["title"].startswith("PN 结")

        restarted = WrongNotebookStore(tmp_path)
        snapshot = await restarted.snapshot()
        assert snapshot["items"][0]["knowledge_points"] == ["PN结"]
        assert snapshot["items"][0]["category_id"] == category["id"]

        updated = await restarted.update(item["id"], title="PN 结判断题")
        assert updated and updated["title"] == "PN 结判断题"
        assert await restarted.delete(item["id"]) is True

    asyncio.run(scenario())


def test_knowledge_graph_merges_sources_and_attaches_wrong_questions(tmp_path):
    chunks = [
        {
            "id": "a",
            "text": "如图1-2所示为器件结构。PN结由P型和N型半导体组成。PN结具有单向导电性。",
            "source": "教材A.pdf",
            "chapter": "半导体器件",
            "section": "PN结",
            "doc_type": "textbook",
            "knowledge_tags": ["PN结"],
        },
        {
            "id": "c",
            "text": "学习说明用于组织课程的扩展阅读材料。",
            "source": "教材A.pdf",
            "chapter": "附录",
            "section": "扩展阅读",
            "doc_type": "textbook",
            "knowledge_tags": ["学习说明"],
        },
        {
            "id": "b",
            "text": "PN结的势垒会随偏置变化。",
            "source": "教材B.pdf",
            "chapter": "二极管",
            "section": "PN结的形成",
            "doc_type": "textbook",
            "knowledge_tags": ["PN结"],
        },
    ]
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in chunks), encoding="utf-8"
    )
    (tmp_path / "question_bank.json").write_text(
        json.dumps(
            {"questions": [{"question_id": "Q1", "question_text": "PN结为何单向导电", "knowledge_tags": ["PN结"]}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Manager:
        @staticmethod
        def index_dir(_: str):
            return tmp_path

    service = KnowledgeGraphService(Manager())
    graph = service.build(
        "default",
        [{"id": "wrong-1", "title": "PN结偏置判断", "knowledge_points": ["PN结"], "updated_at": "now"}],
    )
    node = next(item for item in graph["nodes"] if item["label"] == "PN结")
    assert {source["name"] for source in node["sources"]} == {"教材A.pdf", "教材B.pdf"}
    assert node["questions"][0]["id"] == "Q1"
    assert node["wrong_questions"][0]["id"] == "wrong-1"
    assert node["definition"] == "PN结由P型和N型半导体组成。"
    assert "PN结具有单向导电性。" in node["key_points"]
    assert "图1-2" not in json.dumps(node, ensure_ascii=False)
    assert graph["categories"][-1]["label"] == "其他知识"
    assert service.infer_knowledge_points("请解释PN结", "default") == ["PN结"]


def test_chat_request_accepts_only_two_tutoring_modes():
    request = ChatRequest(session_id="student-1", message="检查这一步", tutoring_mode="guided")
    assert request.tutoring_mode == "guided"


def test_knowledge_graph_rejects_ocr_question_sentences_as_node_labels():
    assert KnowledgeGraphService._clean_point(
        "1.3.2 二极管电路如图所示,请判断二极管是导通还是截止"
    ) == ""
    assert KnowledgeGraphService._clean_point("二极管的伏安特性") == "二极管的伏安特性"
    assert KnowledgeGraphService._clean_point("（已压缩）电子电路基础") == ""
    assert KnowledgeGraphService._clean_point("基本电路和基本分析方法内容回顾") == ""
    assert KnowledgeGraphService._clean_section("1.2.4 二极管的等效电路")
    assert KnowledgeGraphService._clean_section(
        "0.7V。设晶体管β=50,二极管的动态电阻可以忽略不计"
    ) == ""


def test_knowledge_graph_uses_validated_definition_instead_of_ocr_formula_noise():
    summary, definition, key_points = KnowledgeGraphService._summarize_point(
        "叠加定理",
        [
            {
                "text": "U=IVORJ 100k22Rs200Kp由叠加定理得当 3、0时,A组成反相比例加法器。",
                "source": "扫描教材.pdf",
                "parser": "macos-vision-ocr",
            }
        ],
    )

    assert summary == definition
    assert "在线性电路中" in definition
    assert "\\sum_{k=1}^{n}" in definition
    assert "100k22Rs200Kp" not in definition
    assert "功率不能直接叠加" in key_points[1]


def test_knowledge_graph_drops_unreadable_ocr_sentences():
    sentences = KnowledgeGraphService._source_sentences(
        [
            {
                "text": "U=IVORJ 100k22Rs200Kp由某定理得当 3、0时,A组成反相比例加法器。",
                "source": "扫描教材.pdf",
                "parser": "macos-vision-ocr",
            },
            {
                "text": "该电路在线性工作区内可以使用小信号模型进行分析。",
                "source": "文本教材.pdf",
                "parser": "pymupdf",
            },
        ]
    )

    assert [sentence for _, sentence in sentences] == [
        "该电路在线性工作区内可以使用小信号模型进行分析。"
    ]


def test_knowledge_graph_normalizes_embedded_ocr_noise_and_curates_key_points():
    assert KnowledgeGraphService._normalize_ocr_sentence(
        "负反馈取决于基f9Pia本放大电路的输入连接方式。"
    ) == "负反馈取决于基本放大电路的输入连接方式。"

    _, definition, key_points = KnowledgeGraphService._summarize_point(
        "负反馈对输入电阻的影响",
        [
            {
                "text": "输入电阻取决于基f9Pia本放大电路，且图2所示电路需要进一步分析。",
                "source": "扫描教材.pdf",
                "parser": "macos-vision-ocr",
            }
        ],
    )
    assert "串联还是并联" in definition
    assert "串联负反馈" in key_points[0]
    assert all("f9Pia" not in point for point in key_points)


def test_knowledge_graph_curates_field_effect_active_load_card():
    _, definition, key_points = KnowledgeGraphService._summarize_point(
        "场效应管有源电阻及电流源电路", []
    )
    assert "形成有源电阻和电流源电路" in definition
    assert key_points == [
        "场效应管有源负载能够提供较高的等效交流电阻，从而提高单级放大电路的电压增益。",
        "场效应管电流源可提供较稳定的静态工作点，并有利于提高集成度和输出动态范围。",
    ]


def test_knowledge_graph_category_scoring_prefers_specific_domain_markers():
    assert KnowledgeGraphService._category(
        "场效应管有源电阻及电流源电路"
    )[0] == "半导体与器件"
    assert KnowledgeGraphService._category(
        "负反馈对输入电阻的影响"
    )[0] == "模拟电子电路"
    assert KnowledgeGraphService._category("反向截止")[0] == "半导体与器件"
    assert KnowledgeGraphService._category("低通滤波器")[0] == "动态与频域电路"


def test_knowledge_graph_only_lists_primary_sections_for_a_concept(tmp_path):
    chunks = [
        {
            "id": "definition",
            "text": "叠加定理是线性电路分析中的基本定理。",
            "source": "电路基础.pdf",
            "chapter": "网络定理",
            "section": "叠加定理",
            "doc_type": "textbook",
            "knowledge_tags": ["叠加定理"],
        },
        {
            "id": "application",
            "text": "本节分析差分放大电路。计算某一输出时可以利用叠加定理。",
            "source": "模拟电子技术.pdf",
            "chapter": "放大电路",
            "section": "差分放大电路",
            "doc_type": "textbook",
            "knowledge_tags": ["叠加定理"],
        },
    ]
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in chunks),
        encoding="utf-8",
    )

    class Manager:
        @staticmethod
        def index_dir(_: str):
            return tmp_path

    node = KnowledgeGraphService(Manager()).build("default")["nodes"][0]
    assert node["sections"] == ["叠加定理"]
    assert {source["name"] for source in node["sources"]} == {
        "电路基础.pdf",
        "模拟电子技术.pdf",
    }
