"""Tests for `.RUD` task storage and worktrees."""

from __future__ import annotations

import subprocess
from pathlib import Path

import claudeloop.rud_task as rud_task
from claudeloop.rud_task import (
    PLAN,
    SUCCESS_CONDITION,
    TASK_PROMPT,
    create_task,
    delete_task,
    list_work_repo_keys,
    path_under_task,
    prepare_worktree,
    read_meta,
    read_template,
    repo_name_for_work_dir,
    slugify,
    task_root,
    write_template,
)


def test_slugify() -> None:
    assert slugify("Hello World!") == "hello-world"
    assert slugify("---") == "task"


def test_path_under_task(tmp_path: Path) -> None:
    td = tmp_path / "t"
    td.mkdir()
    assert path_under_task(td, "TASK_PROMPT.md") == td / "TASK_PROMPT.md"
    assert path_under_task(td, "../etc/passwd") is None


def test_create_task_and_templates(tmp_path: Path) -> None:
    skills = tmp_path / "skills.md"
    skills.write_text("# skills", encoding="utf-8")
    meta = create_task(
        tmp_path,
        "My Task",
        "这是我们一个general goal：build X",
        skills_path=skills,
        work_dirs=None,
    )
    assert meta.slug.startswith("my-task")
    assert (tmp_path / ".RUD" / meta.slug / TASK_PROMPT).is_file()
    root = task_root(tmp_path, meta.slug)
    assert (root / SUCCESS_CONDITION).is_file()
    assert (root / PLAN).is_file()
    m2 = read_meta(tmp_path, meta.slug)
    assert m2 is not None
    assert m2.skills_path == str(skills.resolve())
    assert write_template(tmp_path, meta.slug, TASK_PROMPT, "hello")
    assert read_template(tmp_path, meta.slug, TASK_PROMPT) == "hello"


def _git_init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "--no-gpg-sign"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_prepare_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "githubA"
    _git_init_repo(repo)
    task_dir = tmp_path / ".RUD" / "slug1"
    task_dir.mkdir(parents=True)
    (task_dir / "work").mkdir(parents=True, exist_ok=True)
    ok, reason, wt = prepare_worktree(task_dir, repo, "slug1")
    assert ok, reason
    assert wt is not None
    assert wt == task_dir / "work" / "githubA"
    assert repo_name_for_work_dir(repo) == "githubA"
    ok2, reason2, wt2 = prepare_worktree(task_dir, repo, "slug1")
    assert ok2 and wt2 == wt and reason2 == "existing"


def test_auto_nested_worktrees_on_create(tmp_path: Path) -> None:
    g1 = tmp_path / "G1"
    _git_init_repo(g1)
    g2 = g1 / "G2"
    _git_init_repo(g2)
    meta = create_task(
        tmp_path,
        "xorl1",
        "这是我们一个general goal：nested",
        skills_path=None,
        work_dirs=[g1],
    )
    assert meta.slug == "xorl1"
    root = task_root(tmp_path, meta.slug)
    assert (root / "work" / "G1").is_dir()
    assert (root / "work" / "G1" / "G2").is_dir()
    keys = list_work_repo_keys(root)
    assert "G1" in keys
    assert "G1/G2" in keys


def test_write_template_rejects_bad_name(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "t", "goal", skills_path=None, work_dirs=None, auto_worktrees=False)
    assert write_template(tmp_path, meta.slug, "evil.md", "x") is False


def test_delete_task_falls_back_to_sudo(monkeypatch, tmp_path: Path) -> None:
    meta = create_task(tmp_path, "sudo delete", "goal", skills_path=None, work_dirs=None, auto_worktrees=False)
    calls: list[list[str]] = []

    def fake_rmtree(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path / "work" / "__pycache__"))

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(rud_task.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(rud_task.subprocess, "run", fake_run)

    ok, err = delete_task(tmp_path, meta.slug)

    assert ok, err
    assert calls == [["sudo", "-n", "rm", "-rf", "--", str(task_root(tmp_path, meta.slug))]]
