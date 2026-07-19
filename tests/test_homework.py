import io
import json

import fitz
import pytest
from PIL import Image, ImageDraw

from backend.app.services.homework import (
    HomeworkStore,
    _choice_recovery_prompt,
    _consolidate_question_keys,
    _grading_reference,
    _merge_prompt_parts,
    _native_inline_answer_bboxes,
    _normalized_page_items,
    _page_prompt,
    _split_labeled_text,
    grade_submission,
    process_homework,
)


class FakeLayoutAdapter:
    def detect(self, _image):
        return []


class FakeVisionClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, prompt, *, image_bytes=None, image_mime="image/png"):
        self.calls.append((prompt, image_bytes, image_mime))
        return self.responses.pop(0)


def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (1000, 1000), "#eef2f1")
    draw = ImageDraw.Draw(image)
    draw.rectangle((450, 150, 650, 250), fill="black")
    draw.rectangle((120, 320, 430, 520), outline="#183c39", width=8)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def extracted_homework(store: HomeworkStore) -> tuple[str, str]:
    created = store.create_homework(
        title="二极管基础练习",
        instructions="请写清计算过程",
        due_at="2026-07-25T18:00",
        filename="练习册.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    extraction = FakeVisionClient(
        {
            "items": [
                {
                    "question_key": "一-1",
                    "section_key": "一",
                    "section_title": "一、计算题（共 10 分）",
                    "number": "1",
                    "question_type": "calculation",
                    "question_text": "计算图示电路中的电流 $I$。",
                    "options": [],
                    "option_columns": 1,
                    "figure_position": "after_question",
                    "points": 10,
                    "question_bboxes": [[50, 50, 950, 600]],
                    "figure_bboxes": [[120, 320, 430, 520]],
                    "answer_bboxes": [[450, 150, 650, 250]],
                    "answer_text": "I = 2 mA",
                    "rubric": "公式 4 分，结果 6 分",
                }
            ],
            "warnings": [],
        }
    )
    process_homework(
        store,
        created["id"],
        client=extraction,
        layout_adapter=FakeLayoutAdapter(),
    )
    raw = store.get_raw_homework(created["id"])
    return created["id"], raw["questions"][0]["id"]


def test_extraction_reflows_text_and_keeps_only_independent_question_figures(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, _ = extracted_homework(store)

    teacher = store.get_homework(homework_id, role="teacher")
    assert teacher["status"] == "draft"
    assert teacher["questions"][0]["answer"] == "I = 2 mA"
    assert teacher["questions"][0]["rubric"] == "公式 4 分，结果 6 分"
    assert teacher["questions"][0]["section_title"] == "一、计算题（共 10 分）"
    assert teacher["questions"][0]["prompt"] == "计算图示电路中的电流 $I$。"
    assert teacher["questions"][0]["options"] == []
    assert teacher["questions"][0]["figure_position"] == "after_question"
    assert teacher["questions"][0]["layout_images"] == []
    assert teacher["questions"][0]["figures"]
    assert store.list_homeworks(role="student", student_id="learner-test") == []

    figure = store.asset_file(homework_id, teacher["questions"][0]["figures"][0]["file"])
    with Image.open(figure) as image:
        assert image.width > 100
        assert image.height > 100

    with pytest.raises(FileNotFoundError):
        store.asset_file(homework_id, "page-001.png")
    assert not (store.root / homework_id / "processing").exists()

    store.publish(homework_id)
    student = store.get_homework(
        homework_id, role="student", student_id="learner-test"
    )
    assert student["status"] == "published"
    assert student["questions"][0]["layout_images"] == []
    assert "answer" not in student["questions"][0]
    assert "rubric" not in student["questions"][0]
    assert "source_url" not in student


def test_submission_is_graded_then_independently_reviewed(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, question_id = extracted_homework(store)
    store.publish(homework_id)
    submission = store.create_submission(
        homework_id=homework_id,
        student_id="learner-test",
        files=[("answer.png", "image/png", sample_image_bytes())],
    )
    grader = FakeVisionClient(
        {
            "extracted_answer": "第 1 题：I = 2 mA",
            "items": [
                {
                    "question_id": question_id,
                    "student_answer": "I = 2 mA",
                    "score": 10,
                    "is_correct": True,
                    "feedback": "答案正确",
                    "evidence": "结果与标准答案一致",
                }
            ],
            "summary": "作答正确",
        }
    )
    reviewer = FakeVisionClient(
        {
            "passed": True,
            "confidence": 0.98,
            "issues": [],
            "recommendation": "无需调整",
        }
    )

    grade_submission(
        store,
        submission["id"],
        grading_client=grader,
        review_client=reviewer,
    )

    graded = store.get_raw_submission(submission["id"])
    assert graded["status"] == "graded"
    assert graded["grading"]["total_score"] == 10
    assert graded["grading"]["max_score"] == 10
    assert graded["review"]["passed"] is True
    assert grader.calls[0][2] == "image/jpeg"
    assert reviewer.calls[0][2] == "image/jpeg"


def test_interrupted_processing_becomes_retryable_after_restart(tmp_path):
    root = tmp_path / "homework"
    store = HomeworkStore(root)
    created = store.create_homework(
        title="重启恢复测试",
        instructions="",
        due_at="",
        filename="exercise.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )

    recovered = HomeworkStore(root).get_homework(created["id"], role="teacher")

    assert recovered["status"] == "error"
    assert "重新识别" in recovered["processing_error"]


def test_whole_document_pass_separates_new_cross_page_question_numbers():
    items = [
        {
            "question_key": "二-1",
            "number": "1",
            "question_type": "calculation",
            "question_text": "第 1 题题干",
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [],
            "page": 3,
        },
        {
            "question_key": "二-1",
            "number": "1",
            "question_type": "calculation",
            "question_text": "第 1 题解题过程续页",
            "question_bboxes": [],
            "answer_bboxes": [[0, 0, 500, 500]],
            "page": 4,
        },
        {
            "question_key": "二-1",
            "number": "2",
            "question_type": "calculation",
            "question_text": "第 2 题新题干",
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [],
            "page": 4,
        },
    ]
    client = FakeVisionClient({
        "assignments": [
            {"segment_index": 0, "canonical_key": "二-1"},
            {"segment_index": 1, "canonical_key": "二-1"},
            {"segment_index": 2, "canonical_key": "二-2"},
        ]
    })

    consolidated, warnings = _consolidate_question_keys(client, items, page_count=4)

    assert [item["question_key"] for item in consolidated] == ["二-1", "二-1", "二-2"]
    assert warnings == []


def test_whole_document_pass_filters_non_question_book_content():
    items = [
        {
            "question_key": "chapter-note",
            "number": "1.2",
            "question_type": "other",
            "question_text": "半导体基础知识与教学要求说明。",
            "subquestions": [],
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [],
            "page": 10,
        },
        {
            "question_key": "1.4-1.2.1",
            "number": "1.2.1",
            "question_type": "calculation",
            "question_text": "对于一个锗 PN 结，试求：",
            "subquestions": [{"label": "1", "text": "反向电压。"}],
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [[0, 500, 500, 900]],
            "page": 21,
        },
    ]
    client = FakeVisionClient({
        "assignments": [
            {
                "segment_index": 0,
                "canonical_key": "chapter-note",
                "keep": False,
                "reason": "普通知识讲解",
            },
            {
                "segment_index": 1,
                "canonical_key": "1.4-1.2.1",
                "keep": True,
                "reason": "有完整题号与作答要求",
            },
        ],
    })

    consolidated, warnings = _consolidate_question_keys(client, items, page_count=21)

    assert [item["question_key"] for item in consolidated] == ["1.4-1.2.1"]
    assert warnings == []


def test_whole_document_pass_cannot_invent_points_for_unscored_workbook():
    items = [{
        "question_key": "1.4-1.2.1",
        "number": "1.2.1",
        "question_type": "calculation",
        "question_text": "对于一个锗 PN 结，试求反向电压。",
        "subquestions": [],
        "points": 0,
        "question_bboxes": [[0, 0, 500, 500]],
        "answer_bboxes": [[0, 500, 500, 900]],
        "page": 21,
    }]
    client = FakeVisionClient({
        "assignments": [{
            "segment_index": 0,
            "canonical_key": "1.4-1.2.1",
            "points": 2,
            "keep": True,
        }],
    })

    consolidated, _ = _consolidate_question_keys(client, items, page_count=21)

    assert consolidated[0]["points"] == 0


def test_choice_questions_recover_complete_options_instead_of_becoming_blanks(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="选择题保真测试",
        instructions="",
        due_at="",
        filename="choice.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    client = FakeVisionClient(
        {
            "items": [{
                "question_key": "一-1",
                "section_key": "一",
                "section_title": "一、选择题（每题 2 分）",
                "number": "1",
                "question_type": "choice",
                "question_text": "二极管的直流电阻和交流电阻分别为 ______。",
                "options": [],
                "points": 2,
                "question_bboxes": [[50, 50, 950, 300]],
                "answer_bboxes": [[400, 100, 430, 130]],
                "answer_text": "A",
            }],
        },
        {
            "recoveries": [{
                "question_key": "一-1",
                "number": "1",
                "options": [
                    {"label": "A", "text": "$700\\,\\Omega$，$26\\,\\Omega$"},
                    {"label": "B", "text": "$700\\,\\Omega$，$16\\,\\Omega$"},
                    {"label": "C", "text": "$0.7\\,\\Omega$，$26\\,\\Omega$"},
                    {"label": "D", "text": "$0.7\\,\\Omega$，$16\\,\\Omega$"},
                ],
                "option_columns": 4,
            }],
        },
    )

    process_homework(
        store,
        created["id"],
        client=client,
        layout_adapter=FakeLayoutAdapter(),
    )

    question = store.get_raw_homework(created["id"])["questions"][0]
    assert question["question_type"] == "choice"
    assert [option["label"] for option in question["options"]] == ["A", "B", "C", "D"]
    assert question["option_columns"] == 4
    assert len(client.calls) == 2
    store.publish(created["id"])


def test_choice_recovery_prompt_contains_valid_json_example():
    prompt = _choice_recovery_prompt(
        {"page": 2, "text": "A. 1kΩ B. 2kΩ C. 4kΩ D. 5kΩ"},
        [{
            "question_key": "一-3",
            "number": "3",
            "page": 1,
            "question_text": "输出电阻为 ______。",
        }],
    )
    example = prompt.rsplit("仅返回 JSON：\n", 1)[1].removesuffix("。")

    parsed = json.loads(example)

    assert parsed["recoveries"][0]["options"][0]["text"] == "$1\\,\\mathrm{k}\\Omega$"


def test_publish_blocks_choice_questions_without_options(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="不完整选择题",
        instructions="",
        due_at="",
        filename="choice.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    store.update_homework(
        created["id"],
        status="draft",
        questions=[{
            "id": "q1",
            "number": "1",
            "question_type": "choice",
            "prompt": "请选择正确答案。",
            "options": [],
        }],
    )

    with pytest.raises(RuntimeError, match="缺少完整选项"):
        store.publish(created["id"])


def test_repeated_cross_page_stem_is_not_printed_twice():
    first = """如图4.1 所示共发射极放大电路，β = 150，V_T = 26mV，V_BE(on) = 0.7V。
(1) 求静态工作点电流 I_CQ。
(2) 使用微变等效电路法求电压增益 A_v1。
(3) 若电容 C_e 开路，求电压增益 A_v2。"""
    hallucinated_repeat = """如图4.1 所示共发射极放大电路，β = 150，V_T = 26mV，V_BE(on) = 0.7V。
(1) 求静态工作点电流 I_CQ。
(2) 使用微变等效电路求中频电压增益 A_v1。
(3) 若在 R_E1 两端并联电容 C_E，求闭环增益 A_v2。"""

    merged = _merge_prompt_parts([first, hallucinated_repeat])

    assert merged == first.strip()
    assert merged.count("如图4.1") == 1


def test_inline_subquestions_are_split_and_backward_references_stay_in_their_part():
    stem, parts = _split_labeled_text(
        "如图所示，完成下列问题：(1) 求静态电流。 (2) 求电压增益。 "
        "(3) 根据第(2)问结果求输出电阻。"
    )

    assert stem == "如图所示，完成下列问题："
    assert [part["label"] for part in parts] == ["1", "2", "3"]
    assert "第(2)问" in parts[2]["text"]


def test_page_prompt_filters_book_explanations_and_requires_structured_subquestions():
    prompt = _page_prompt(
        {"page": 21, "text": "1.2 基本知识点 1.4 习题解答"},
        [],
        [],
    )

    assert "试卷、课后习题、习题册、学习指导书" in prompt
    assert "教学要求、基本知识点、概念讲解" in prompt
    assert "返回空 items" in prompt
    assert "question_text 与 subquestions" in prompt
    assert "answer_text 与 answer_subquestions" in prompt
    assert "题目卷" not in prompt


def test_guidance_book_question_and_answer_parts_are_normalized_for_layout_and_grading():
    items = _normalized_page_items(
        {
            "items": [{
                "question_key": "1.4-1.2.1",
                "section_key": "1.4",
                "section_title": "1.4 习题解答",
                "number": "1.2.1",
                "question_type": "calculation",
                "question_text": (
                    "对于一个锗 PN 结，在 $T=290\\,\\mathrm{K}$ 时，试求： "
                    "(1) 反向电流达到饱和电流的 90% 时的反向电压。 "
                    "(2) 正向电压和反向电压均为 $0.05\\,\\mathrm{V}$ 时的电流比。"
                ),
                "subquestions": [
                    {"label": "(1)", "text": "反向电流达到饱和电流的 90% 时的反向电压。"},
                    {"label": "2", "text": "正向电压和反向电压均为 $0.05\\,\\mathrm{V}$ 时的电流比。"},
                ],
                "answer_text": (
                    "由二极管方程计算。 (1) $v_D=-0.0576\\,\\mathrm{V}$。 "
                    "(2) 电流比为 $-7.389$。"
                ),
                "answer_subquestions": [
                    {"label": "1", "text": "$v_D=-0.0576\\,\\mathrm{V}$。"},
                    {"label": "2", "text": "电流比为 $-7.389$。"},
                ],
            }],
        },
        21,
    )

    assert len(items) == 1
    assert items[0]["number"] == "1.2.1"
    assert items[0]["question_text"].endswith("试求：")
    assert items[0]["answer_text"] == "由二极管方程计算。"
    assert [part["label"] for part in items[0]["subquestions"]] == ["1", "2"]
    assert [part["label"] for part in items[0]["answer_subquestions"]] == ["1", "2"]

    reference = _grading_reference({
        "questions": [{
            "id": "q1",
            "number": "1.2.1",
            "prompt": items[0]["question_text"],
            "subquestions": items[0]["subquestions"],
            "points": 0,
            "answer": items[0]["answer_text"],
            "answer_subquestions": items[0]["answer_subquestions"],
            "rubric": "",
        }],
    })
    assert "(1) 反向电流" in reference[0]["question"]
    assert "(2) 电流比" in reference[0]["standard_answer"]


def test_pdf_glyph_boxes_find_answer_filled_between_underlines():
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.insert_text((40, 100), "Question: ___A____.")

    boxes = _native_inline_answer_bboxes(page)

    document.close()
    assert len(boxes) == 1
    assert 0 < boxes[0][2] - boxes[0][0] < 80
