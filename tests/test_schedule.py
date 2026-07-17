import asyncio

from backend.app.services.schedule import StudentSchedule


def test_schedule_is_durable_ordered_and_student_scoped(tmp_path):
    path = tmp_path / "schedule.json"
    schedule = StudentSchedule(path)
    later = asyncio.run(schedule.add(
        student_id="learner-a",
        title="社团活动",
        date="2026-07-20",
        time="19:00",
        category="activity",
        note="教学楼门口集合",
    ))
    earlier = asyncio.run(schedule.add(
        student_id="learner-a",
        title="模拟电路考试",
        date="2026-07-20",
        time="09:00",
        category="exam",
        note="",
    ))
    asyncio.run(schedule.add(
        student_id="learner-b",
        title="另一位学生的安排",
        date="2026-07-19",
        time="",
        category="other",
        note="",
    ))

    restored = asyncio.run(StudentSchedule(path).list("learner-a"))
    assert [item["id"] for item in restored] == [earlier["id"], later["id"]]
    assert restored[0]["completed"] is False
    assert len(asyncio.run(StudentSchedule(path).list("learner-b"))) == 1


def test_schedule_completion_and_delete_are_student_scoped(tmp_path):
    schedule = StudentSchedule(tmp_path / "schedule.json")
    item = asyncio.run(schedule.add(
        student_id="learner-a",
        title="复习二极管",
        date="2026-07-17",
        time="20:00",
        category="study",
        note="完成课后题",
    ))

    assert asyncio.run(schedule.set_completed("learner-b", item["id"], True)) is None
    completed = asyncio.run(schedule.set_completed("learner-a", item["id"], True))
    assert completed is not None
    assert completed["completed"] is True
    assert asyncio.run(schedule.delete("learner-b", item["id"])) is False
    assert asyncio.run(schedule.delete("learner-a", item["id"])) is True
    assert asyncio.run(schedule.list("learner-a")) == []
