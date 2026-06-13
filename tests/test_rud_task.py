"""Tests for the slim `.RUD` task storage layer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import claudeloop.rud_task as rud_task
from claudeloop.rud_task import (
    PLAN,
    add_claude_session,
    claude_project_dir,
    create_task,
    delete_task,
    list_session_files,
    list_task_markdown_files,
    path_under_task,
    project_notes_path,
    read_meta,
    read_project_notes,
    read_task_markdown_file,
    read_task_monitor,
    read_template,
    session_id_from_path,
    slugify,
    task_root,
    task_worktree_path,
    worktree_diff,
    write_project_notes,
    write_task_monitor,
    write_template,
)


def test_slugify() -> None:
    assert slugify("Hello World!") == "hello-world"
    assert slugify("---") == "task"


def test_path_under_task(tmp_path: Path) -> None:
    td = tmp_path / "t"
    td.mkdir()
    assert path_under_task(td, "PLAN.md") == td / "PLAN.md"
    assert path_under_task(td, "../etc/passwd") is None


def test_create_task_seeds_plan_only(tmp_path: Path) -> None:
    skills = tmp_path / "skills.md"
    skills.write_text("# skills", encoding="utf-8")
    meta = create_task(
        tmp_path,
        "My Task",
        "build the thing",
        skills_path=skills,
        auto_worktree=False,
    )
    assert meta.slug.startswith("my-task")
    root = task_root(tmp_path, meta.slug)
    assert (root / PLAN).is_file()
    # PLAN.md is the only per-task markdown file. NOTES.md is project-scoped
    # (lives at <project>/.RUD/NOTES.md), and INTERVIEW.md / TASK_PROMPT.md /
    # SUCCESS_CONDITION.md are no longer seeded.
    assert not (root / "NOTES.md").exists()
    assert not (root / "INTERVIEW.md").exists()
    assert not (root / "interview.md").exists()
    assert not (root / "TASK_PROMPT.md").exists()
    assert not (root / "SUCCESS_CONDITION.md").exists()
    m2 = read_meta(tmp_path, meta.slug)
    assert m2 is not None
    assert m2.skills_path == str(skills.resolve())
    assert m2.claude_session_ids == []


def test_write_and_read_plan_template(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "edit me", "goal", skills_path=None, auto_worktree=False)
    assert write_template(tmp_path, meta.slug, PLAN, "hello plan")
    assert read_template(tmp_path, meta.slug, PLAN) == "hello plan"


def test_write_template_rejects_disallowed_names(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "t", "goal", skills_path=None, auto_worktree=False)
    # PLAN.md is the only allowed per-task template now.  NOTES.md moved
    # to the project root and TASK_PROMPT.md / SUCCESS_CONDITION.md are gone.
    assert write_template(tmp_path, meta.slug, "NOTES.md", "x") is False
    assert write_template(tmp_path, meta.slug, "TASK_PROMPT.md", "x") is False
    assert write_template(tmp_path, meta.slug, "evil.md", "x") is False


def test_list_task_markdown_files_returns_plan_first(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "mds", "goal", skills_path=None, auto_worktree=False)
    root = task_root(tmp_path, meta.slug)
    (root / "review.md").write_text("# review", encoding="utf-8")
    (root / "Notes.md").write_text("# notes", encoding="utf-8")
    # Non-markdown files are excluded; nested markdown is surfaced by
    # relative path so worktree docs can be previewed from the UI.
    (root / "stuff.txt").write_text("ignore me", encoding="utf-8")
    nested = root / "work" / "subrepo"
    nested.mkdir(parents=True)
    (nested / "DEEP.md").write_text("# deep", encoding="utf-8")

    names = list_task_markdown_files(tmp_path, meta.slug)
    # PLAN.md must be first; the rest are sorted case-insensitively.
    assert names[0] == "PLAN.md"
    assert names[1:] == ["Notes.md", "review.md", "work/subrepo/DEEP.md"]
    assert "stuff.txt" not in names


def test_list_task_markdown_files_without_plan(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "noplan", "goal", skills_path=None, auto_worktree=False)
    root = task_root(tmp_path, meta.slug)
    (root / PLAN).unlink()
    (root / "alpha.md").write_text("a", encoding="utf-8")
    (root / "beta.md").write_text("b", encoding="utf-8")
    # PLAN.md isn't on disk anymore; we should just get the other two
    # in case-insensitive sorted order with no special PLAN.md slot.
    assert list_task_markdown_files(tmp_path, meta.slug) == ["alpha.md", "beta.md"]


def test_read_task_markdown_file_round_trip(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "rt", "goal", skills_path=None, auto_worktree=False)
    root = task_root(tmp_path, meta.slug)
    (root / "review.md").write_text("hello review", encoding="utf-8")
    assert read_task_markdown_file(tmp_path, meta.slug, "review.md") == "hello review"
    # PLAN.md was seeded; readable too.
    assert isinstance(read_task_markdown_file(tmp_path, meta.slug, PLAN), str)


def test_read_task_markdown_file_rejects_unsafe_names(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "safe", "goal", skills_path=None, auto_worktree=False)
    # Path traversal must be blocked even if the resolved file exists.
    outside = tmp_path / "secret.md"
    outside.write_text("nope", encoding="utf-8")
    assert read_task_markdown_file(tmp_path, meta.slug, "../secret.md") is None
    assert read_task_markdown_file(tmp_path, meta.slug, "../../etc/passwd") is None
    assert read_task_markdown_file(tmp_path, meta.slug, "work/inner.md") is None
    # Non-markdown extensions are refused.
    (task_root(tmp_path, meta.slug) / "task.json").write_text("{}", encoding="utf-8")
    assert read_task_markdown_file(tmp_path, meta.slug, "task.json") is None
    # Missing files return None instead of raising.
    assert read_task_markdown_file(tmp_path, meta.slug, "ghost.md") is None
    # Empty / dot names rejected before any filesystem lookup.
    assert read_task_markdown_file(tmp_path, meta.slug, "") is None
    assert read_task_markdown_file(tmp_path, meta.slug, ".") is None
    assert read_task_markdown_file(tmp_path, meta.slug, "..") is None


def test_project_notes_round_trip(tmp_path: Path) -> None:
    # Reads return empty string when the file doesn't exist yet.
    assert read_project_notes(tmp_path) == ""
    # Write places the file at <project>/.RUD/NOTES.md and not inside a task.
    assert write_project_notes(tmp_path, "future ideas")
    assert project_notes_path(tmp_path) == tmp_path / ".RUD" / "NOTES.md"
    assert read_project_notes(tmp_path) == "future ideas"
    # Creating a task afterwards must not stomp the project notes.
    create_task(tmp_path, "t", "goal", skills_path=None, auto_worktree=False)
    assert read_project_notes(tmp_path) == "future ideas"


def test_add_claude_session_dedup_and_order(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "sess", "goal", skills_path=None, auto_worktree=False)
    add_claude_session(tmp_path, meta.slug, "uuid-A")
    add_claude_session(tmp_path, meta.slug, "uuid-B")
    add_claude_session(tmp_path, meta.slug, "uuid-A")  # already there - move to end
    m = read_meta(tmp_path, meta.slug)
    assert m is not None
    assert m.claude_session_ids == ["uuid-B", "uuid-A"]


def test_meta_ignores_legacy_fields(tmp_path: Path) -> None:
    """Older task.json files have extra fields - reading them must not crash."""
    meta = create_task(tmp_path, "legacy", "goal", skills_path=None, auto_worktree=False)
    meta_path = task_root(tmp_path, meta.slug) / "task.json"
    raw = meta_path.read_text(encoding="utf-8").rstrip().rstrip("}")
    raw += (
        ', "work_dirs": ["/old"], "tmux_runner_target": "x:0.0",'
        ' "tmux_ask_target": "y:0.0", "interview_backend": "cli"}'
    )
    meta_path.write_text(raw, encoding="utf-8")
    reloaded = read_meta(tmp_path, meta.slug)
    assert reloaded is not None
    assert reloaded.slug == meta.slug
    assert reloaded.title == "legacy"


def test_delete_task_falls_back_to_sudo(monkeypatch, tmp_path: Path) -> None:
    meta = create_task(tmp_path, "sudo delete", "goal", skills_path=None, auto_worktree=False)
    calls: list[list[str]] = []

    def fake_rmtree(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path / "__pycache__"))

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(rud_task.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(rud_task.subprocess, "run", fake_run)

    ok, err = delete_task(tmp_path, meta.slug)

    assert ok, err
    # First call may be `git worktree remove --force` (fails silently if no
    # worktree was created) and the second call is the sudo rmtree.
    assert any(c == ["sudo", "-n", "rm", "-rf", "--", str(task_root(tmp_path, meta.slug))] for c in calls)


# --- session lookup helpers -------------------------------------------------


def test_claude_project_dir_encodes_path() -> None:
    path = Path("/home/u/proj/.RUD/foo/work/r")
    expected_suffix = "-home-u-proj--RUD-foo-work-r"
    assert claude_project_dir(path).name == expected_suffix


def test_list_session_files_returns_jsonl_sorted(tmp_path: Path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(rud_task.Path, "home", classmethod(lambda cls: fake_home))
    cwd = tmp_path / "a" / "b"
    cwd.mkdir(parents=True)
    enc = claude_project_dir(cwd)
    enc.mkdir(parents=True)
    older = enc / "aaaa-bbbb.jsonl"
    newer = enc / "cccc-dddd.jsonl"
    older.write_text("{}\n", encoding="utf-8")
    newer.write_text("{}\n", encoding="utf-8")
    import os
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    enc.joinpath("ignored.txt").write_text("nope", encoding="utf-8")
    files = list_session_files(cwd)
    assert [p.name for p in files] == ["aaaa-bbbb.jsonl", "cccc-dddd.jsonl"]
    assert session_id_from_path(files[-1]) == "cccc-dddd"


# --- worktree helpers (real git) --------------------------------------------


def _git_init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.local",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.local",
    }
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env={**env, "PATH": "/usr/bin:/bin"})
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "README.md").write_text("# hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, env={**env, "PATH": "/usr/bin:/bin"})
    subprocess.run(
        ["git", "commit", "-m", "init", "--no-gpg-sign", "-q"],
        cwd=path,
        check=True,
        env={**env, "PATH": "/usr/bin:/bin"},
    )


def test_create_task_auto_creates_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "myrepo"
    _git_init_repo(repo)
    meta = create_task(repo, "wt task", "goal", skills_path=None)
    wt = task_worktree_path(repo, meta.slug)
    assert wt is not None
    assert wt == task_root(repo, meta.slug) / "work" / "myrepo"
    assert (wt / "README.md").is_file()
    # Branch follows the zhongzhu/<slug> convention.
    reloaded = read_meta(repo, meta.slug)
    assert reloaded is not None
    assert reloaded.branch == f"zhongzhu/{meta.slug}"
    assert reloaded.worktree_path == str(wt)


def test_delete_task_removes_worktree_registration(tmp_path: Path) -> None:
    """Deleting a task must also unregister its git worktree(s), not just rmtree
    the dir - otherwise the source repo keeps a stale `git worktree list` entry."""
    repo = tmp_path / "myrepo"
    _git_init_repo(repo)
    meta = create_task(repo, "wt del", "goal", skills_path=None)
    wt = task_worktree_path(repo, meta.slug)
    assert wt is not None and wt.is_dir()

    def worktree_list() -> str:
        r = subprocess.run(
            ["git", "worktree", "list"], cwd=repo, capture_output=True, text=True,
        )
        return r.stdout

    assert str(wt) in worktree_list()
    reg = repo / ".git" / "worktrees"
    assert reg.is_dir() and any(reg.iterdir())

    ok, msg = delete_task(repo, meta.slug)
    assert ok, msg
    assert not task_root(repo, meta.slug).exists()
    # No dangling registration left behind.
    assert str(wt) not in worktree_list()
    assert not (reg.is_dir() and any(reg.iterdir()))


def test_create_task_skips_worktree_in_non_git_root(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "no-git", "goal", skills_path=None)
    assert task_worktree_path(tmp_path, meta.slug) is None
    reloaded = read_meta(tmp_path, meta.slug)
    assert reloaded is not None
    assert reloaded.worktree_path == ""
    assert reloaded.branch == ""


def test_detect_worktree_backfills_old_meta(tmp_path: Path) -> None:
    """Tasks created before auto-worktree existed should pick up the worktree
    on next read."""
    from claudeloop.rud_task import detect_and_persist_worktree

    repo = tmp_path / "myrepo"
    _git_init_repo(repo)

    # Step 1: create a task with auto_worktree disabled, simulating a task
    # whose task.json was written before today's changes.
    meta = create_task(repo, "old style", "goal", skills_path=None, auto_worktree=False)
    assert read_meta(repo, meta.slug).worktree_path == ""

    # Step 2: user manually adds a worktree the way they used to.
    td = task_root(repo, meta.slug)
    wt_dir = td / "work" / "myrepo"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "-b", "zhongzhu/legacy", str(wt_dir), "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    # Step 3: the next read should detect it and persist the path.
    updated = detect_and_persist_worktree(repo, meta.slug)
    assert updated is not None
    assert updated.worktree_path == str(wt_dir.resolve())
    assert updated.branch == "zhongzhu/legacy"
    # Persisted to disk too.
    assert read_meta(repo, meta.slug).worktree_path == str(wt_dir.resolve())


def test_list_candidates_returns_self_when_project_root_is_git(tmp_path: Path) -> None:
    from claudeloop.rud_task import list_worktree_candidates

    repo = tmp_path / "repo"
    _git_init_repo(repo)
    cands = list_worktree_candidates(repo)
    assert len(cands) == 1
    assert cands[0]["kind"] == "self"
    assert cands[0]["path"] == str(repo.resolve())


def test_list_candidates_returns_children_when_root_is_container(tmp_path: Path) -> None:
    """Container project root - xorl-style folder holding multiple git clones."""
    from claudeloop.rud_task import list_worktree_candidates

    container = tmp_path / "xorl"
    container.mkdir()
    for name in ("xorl-internal", "xorl-eval"):
        _git_init_repo(container / name)
    # Add a non-git directory and a hidden dir to make sure they're skipped.
    (container / "scratch").mkdir()
    (container / ".cache").mkdir()
    cands = list_worktree_candidates(container)
    names = sorted(c["name"] for c in cands)
    assert names == ["xorl-eval", "xorl-internal"]
    assert all(c["kind"] == "child" for c in cands)


def test_prepare_task_worktree_from_child_repo(tmp_path: Path) -> None:
    """Verify manual selection of a child repo creates a worktree under the
    *task*, not under the container."""
    from claudeloop.rud_task import prepare_task_worktree_from

    container = tmp_path / "xorl"
    container.mkdir()
    child = container / "xorl-internal"
    _git_init_repo(child)
    meta = create_task(container, "review309", "goal", skills_path=None, auto_worktree=False)
    # Container project root means create_task's auto-worktree skipped.
    assert read_meta(container, meta.slug).worktree_path == ""
    wt, branch, msg = prepare_task_worktree_from(container, meta.slug, child)
    assert wt is not None, msg
    assert wt == task_root(container, meta.slug) / "work" / "xorl-internal"
    assert branch == f"zhongzhu/{meta.slug}"


def test_multiple_worktrees_per_task(tmp_path: Path) -> None:
    """Adding a second worktree appends to meta.worktrees instead of replacing."""
    from claudeloop.rud_task import (
        detect_and_persist_worktree,
        list_task_worktrees,
        prepare_task_worktree_from,
    )

    container = tmp_path / "xorl"
    container.mkdir()
    _git_init_repo(container / "xorl-internal")
    _git_init_repo(container / "xorl-sglang")
    meta = create_task(container, "rev309", "goal", skills_path=None, auto_worktree=False)

    wt1, _, _ = prepare_task_worktree_from(container, meta.slug, container / "xorl-internal")
    assert wt1 is not None
    detect_and_persist_worktree(container, meta.slug)
    after1 = read_meta(container, meta.slug)
    assert after1.worktrees == [str(wt1)]
    assert after1.worktree_path == str(wt1)
    assert after1.branch == f"zhongzhu/{meta.slug}"
    assert len(after1.branches) == 1

    wt2, _, _ = prepare_task_worktree_from(container, meta.slug, container / "xorl-sglang")
    assert wt2 is not None
    detect_and_persist_worktree(container, meta.slug)
    after2 = read_meta(container, meta.slug)
    assert sorted(after2.worktrees) == sorted([str(wt1), str(wt2)])
    assert len(after2.branches) == len(after2.worktrees)
    # Every worktree gets the SAME branch name (zhongzhu/<slug>) - branches
    # are scoped per repo so there's no collision.
    expected_branch = f"zhongzhu/{meta.slug}"
    assert after2.branches == [expected_branch, expected_branch]
    # Primary stays as the first one (xorl-internal was added first).
    assert after2.worktree_path == str(wt1)
    assert list_task_worktrees(container, meta.slug)  # populated
    # Verify each git repo really has its own independent zhongzhu/<slug>
    # branch (proves the "same name across repos" guarantee).
    for repo_root in (container / "xorl-internal", container / "xorl-sglang"):
        result = subprocess.run(
            ["git", "branch", "--list", expected_branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert expected_branch in result.stdout, (
            f"{repo_root} missing branch {expected_branch!r}; got: {result.stdout!r}"
        )


def test_remove_task_worktree(tmp_path: Path) -> None:
    from claudeloop.rud_task import (
        detect_and_persist_worktree,
        prepare_task_worktree_from,
        remove_task_worktree,
    )

    container = tmp_path / "c"
    container.mkdir()
    _git_init_repo(container / "a")
    _git_init_repo(container / "b")
    meta = create_task(container, "t", "g", skills_path=None, auto_worktree=False)
    wt_a, _, _ = prepare_task_worktree_from(container, meta.slug, container / "a")
    wt_b, _, _ = prepare_task_worktree_from(container, meta.slug, container / "b")
    detect_and_persist_worktree(container, meta.slug)
    assert len(read_meta(container, meta.slug).worktrees) == 2

    ok, msg = remove_task_worktree(container, meta.slug, wt_a)
    assert ok, msg
    after = read_meta(container, meta.slug)
    assert after.worktrees == [str(wt_b)]
    assert after.worktree_path == str(wt_b)
    assert not wt_a.exists()


def test_legacy_meta_single_worktree_migrates_to_list(tmp_path: Path) -> None:
    """Pre-list-era task.json with only worktree_path should appear as a 1-item list."""
    repo = tmp_path / "r"
    _git_init_repo(repo)
    meta = create_task(repo, "legacy", "g", skills_path=None)
    # Tamper to look pre-migration: drop the lists field, keep singular.
    meta_path = task_root(repo, meta.slug) / "task.json"
    import json
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw.pop("worktrees", None)
    raw.pop("branches", None)
    meta_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    reloaded = read_meta(repo, meta.slug)
    assert reloaded is not None
    assert reloaded.worktrees == [reloaded.worktree_path]
    assert reloaded.branches == [reloaded.branch]


def test_worktree_status_reports_clean_then_dirty(tmp_path: Path) -> None:
    from claudeloop.rud_task import worktree_status

    repo = tmp_path / "r"
    _git_init_repo(repo)
    status = worktree_status(repo)
    assert status is not None
    assert status["clean"] is True
    assert status["dirty_count"] == 0
    assert status["branch"]  # e.g. "master" or "main"
    assert status["has_remote"] is False
    assert status["ahead"] == 0 and status["behind"] == 0

    (repo / "new.txt").write_text("hi", encoding="utf-8")
    (repo / "README.md").write_text("changed", encoding="utf-8")
    dirty = worktree_status(repo)
    assert dirty is not None
    assert dirty["clean"] is False
    assert dirty["untracked"] == 1
    assert dirty["unstaged"] == 1
    assert dirty["dirty_count"] == 2


def test_worktree_status_handles_ahead_behind(tmp_path: Path) -> None:
    """Set up an `origin` remote and verify ahead/behind parsing."""
    from claudeloop.rud_task import worktree_status

    upstream = tmp_path / "u"
    _git_init_repo(upstream)
    # Make `upstream` a bare-style "remote" by using a worktree clone.
    clone = tmp_path / "c"
    subprocess.run(["git", "clone", "-q", str(upstream), str(clone)], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=clone, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=clone, check=True)
    (clone / "x.txt").write_text("a", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    subprocess.run(
        ["git", "commit", "-m", "one", "--no-gpg-sign", "-q"],
        cwd=clone,
        check=True,
    )
    s = worktree_status(clone)
    assert s is not None
    assert s["has_remote"] is True
    assert s["ahead"] == 1
    assert s["behind"] == 0


def test_list_task_worktree_statuses(tmp_path: Path) -> None:
    from claudeloop.rud_task import (
        detect_and_persist_worktree,
        list_task_worktree_statuses,
        prepare_task_worktree_from,
    )

    container = tmp_path / "c"
    container.mkdir()
    _git_init_repo(container / "a")
    _git_init_repo(container / "b")
    meta = create_task(container, "wts", "g", skills_path=None, auto_worktree=False)
    prepare_task_worktree_from(container, meta.slug, container / "a")
    prepare_task_worktree_from(container, meta.slug, container / "b")
    detect_and_persist_worktree(container, meta.slug)
    statuses = list_task_worktree_statuses(container, meta.slug)
    assert len(statuses) == 2
    for s in statuses:
        assert s["branch"] == f"zhongzhu/{meta.slug}"
        assert s["clean"] is True


def test_create_task_defaults_to_claude_agent(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "ag1", "g", skills_path=None, auto_worktree=False)
    assert meta.agent == "claude"
    assert read_meta(tmp_path, meta.slug).agent == "claude"


def test_create_task_with_codex_agent(tmp_path: Path) -> None:
    meta = create_task(
        tmp_path,
        "ag2",
        "g",
        skills_path=None,
        agent="codex",
        auto_worktree=False,
    )
    assert meta.agent == "codex"
    assert read_meta(tmp_path, meta.slug).agent == "codex"


def test_normalize_and_label_agent() -> None:
    from claudeloop.rud_task import agent_label, normalize_agent

    assert normalize_agent("claude") == "claude"
    assert normalize_agent("CODEX") == "codex"
    assert normalize_agent("") == "claude"
    assert normalize_agent("gpt-4") == "claude"  # unknown -> default
    assert agent_label("claude") == "Claude"
    assert agent_label("codex") == "Codex"
    assert agent_label("nonsense") == "Claude"


def test_build_agent_command_claude() -> None:
    from claudeloop.rud_task import build_agent_command

    cmd = build_agent_command("claude")
    assert cmd[0] == "claude"
    assert "--model" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--effort" in cmd and "max" in cmd
    assert "--resume" not in cmd

    cmd_resume = build_agent_command("claude", model="m1", resume_session_id="abc-123")
    assert cmd_resume[cmd_resume.index("--model") + 1] == "m1"
    assert cmd_resume[cmd_resume.index("--resume") + 1] == "abc-123"


def test_build_agent_command_codex() -> None:
    from claudeloop.rud_task import build_agent_command

    cmd = build_agent_command("codex")
    assert cmd == ["codex"]

    cmd_model = build_agent_command("codex", model="o3")
    assert cmd_model == ["codex", "-c", "model=o3"]

    cmd_resume = build_agent_command("codex", resume_session_id="019e296e-…")
    assert cmd_resume[:3] == ["codex", "resume", "019e296e-…"]

    cmd_both = build_agent_command("codex", model="o3", resume_session_id="abc")
    assert cmd_both == ["codex", "resume", "abc", "-c", "model=o3"]


def test_meta_migrates_old_task_json_without_agent(tmp_path: Path) -> None:
    """task.json from before the agent field should default agent to claude."""
    meta = create_task(tmp_path, "lg", "g", skills_path=None, auto_worktree=False)
    meta_path = task_root(tmp_path, meta.slug) / "task.json"
    import json
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw.pop("agent", None)
    meta_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    reloaded = read_meta(tmp_path, meta.slug)
    assert reloaded is not None
    assert reloaded.agent == "claude"


def test_list_session_files_dispatches_to_codex(tmp_path: Path, monkeypatch) -> None:
    """Codex sessions are matched by ``payload.cwd`` in the rollout file."""
    from claudeloop.rud_task import (
        list_session_files,
        session_id_from_path,
    )

    fake_home = tmp_path / "home"
    monkeypatch.setattr(rud_task.Path, "home", classmethod(lambda cls: fake_home))
    cwd = tmp_path / "wt"
    cwd.mkdir()
    sessions_dir = fake_home / ".codex" / "sessions" / "2026" / "05" / "28"
    sessions_dir.mkdir(parents=True)
    # One session matching our cwd
    match_path = sessions_dir / "rollout-2026-05-28T01-23-45-aaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
    import json as _json
    match_path.write_text(
        _json.dumps({
            "type": "session_meta",
            "payload": {"id": "aaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "cwd": str(cwd.resolve())},
        }) + "\n",
        encoding="utf-8",
    )
    # One session from a different cwd
    other_path = sessions_dir / "rollout-2026-05-28T02-00-00-zzzz.jsonl"
    other_path.write_text(
        _json.dumps({
            "type": "session_meta",
            "payload": {"id": "zzzz", "cwd": "/somewhere/else"},
        }) + "\n",
        encoding="utf-8",
    )
    files = list_session_files(cwd, "codex")
    assert [p.name for p in files] == [match_path.name]
    assert session_id_from_path(files[0], "codex") == "aaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    # Claude lookup for the same cwd returns nothing (no ~/.claude/projects).
    assert list_session_files(cwd, "claude") == []


def test_detect_worktree_clears_stale_path(tmp_path: Path) -> None:
    """A recorded worktree_path that no longer exists on disk should be
    re-detected (and replaced with the on-disk reality, or cleared)."""
    from claudeloop.rud_task import detect_and_persist_worktree, update_meta

    repo = tmp_path / "r"
    _git_init_repo(repo)
    meta = create_task(repo, "wt", "goal", skills_path=None)
    # Pretend metadata pointed somewhere now missing.
    update_meta(repo, meta.slug, worktree_path="/no/such/path")
    refreshed = detect_and_persist_worktree(repo, meta.slug)
    assert refreshed is not None
    # On-disk worktree still exists, so detection picks it back up.
    assert Path(refreshed.worktree_path).is_dir()


def test_task_monitor_roundtrip(tmp_path: Path) -> None:
    meta = create_task(tmp_path, "mon", "g", skills_path=None, auto_worktree=False)
    # Default for a brand-new task: disabled, empty pattern.
    cfg = read_task_monitor(tmp_path, meta.slug)
    assert cfg == {"enabled": False, "pattern": "", "last_fired": "", "last_match": ""}

    assert write_task_monitor(
        tmp_path, meta.slug, enabled=True, pattern=r"done|blocked"
    )
    cfg = read_task_monitor(tmp_path, meta.slug)
    assert cfg["enabled"] is True
    assert cfg["pattern"] == r"done|blocked"

    # Omitting last_fired/last_match preserves prior values.
    write_task_monitor(
        tmp_path, meta.slug, enabled=True, pattern=r"done|blocked",
        last_fired="2026-01-01T00:00:00+00:00", last_match="done",
    )
    write_task_monitor(tmp_path, meta.slug, enabled=False, pattern=r"x")
    cfg = read_task_monitor(tmp_path, meta.slug)
    assert cfg["enabled"] is False
    assert cfg["pattern"] == "x"
    assert cfg["last_fired"] == "2026-01-01T00:00:00+00:00"
    assert cfg["last_match"] == "done"


def test_worktree_diff_captures_uncommitted_and_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "drepo"
    _git_init_repo(repo)
    meta = create_task(repo, "diff task", "goal", skills_path=None)
    wt = task_worktree_path(repo, meta.slug)
    assert wt is not None and wt.is_dir()

    # Modify a tracked file + add an untracked one inside the worktree.
    (wt / "README.md").write_text("# hi\nmore\n", encoding="utf-8")
    (wt / "new_file.txt").write_text("brand new\n", encoding="utf-8")

    d = worktree_diff(wt)
    assert d["path"] == str(wt)
    files = {f["path"]: f for f in d["files"]}
    assert "README.md" in files
    assert files["README.md"]["scope"] == "uncommitted"
    assert files["README.md"]["status"] == "modified"
    assert files["README.md"]["additions"] >= 1
    assert "new_file.txt" in files
    assert files["new_file.txt"]["status"] == "added"


def test_parse_patch_files_splits_per_file() -> None:
    from claudeloop.rud_task import _parse_patch_files

    patch = (
        "diff --git a/foo.py b/foo.py\n"
        "index 111..222 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
        " context\n"
        "diff --git a/bar.txt b/bar.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/bar.txt\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
    )
    files = _parse_patch_files(patch, "uncommitted")
    assert [f["path"] for f in files] == ["foo.py", "bar.txt"]
    assert files[0]["status"] == "modified"
    assert files[0]["additions"] == 1 and files[0]["deletions"] == 1
    assert files[1]["status"] == "added"
    assert files[1]["additions"] == 1
