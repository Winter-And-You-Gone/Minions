"""测试命令补全。"""

from prompt_toolkit.document import Document

from voice_agent.cli.command_completer import MinionsCommandCompleter


def collect(text: str) -> list:
    completer = MinionsCommandCompleter()
    return list(completer.get_completions(Document(text, cursor_position=len(text)), None))


def test_slash_command_completion():
    comps = collect("/sta")
    assert any(c.text == "/status" for c in comps)


def test_completion_has_description():
    comps = collect("/sta")
    c = next(c for c in comps if c.text == "/status")
    assert c.display_meta_text


def test_name_subcommand_completion():
    comps = collect("/name ")
    assert any(c.text == "set" for c in comps)
    assert any(c.text == "alias" for c in comps)
    assert any(c.text == "save" for c in comps)


def test_mic_subcommand_completion():
    comps = collect("/mic ")
    assert any(c.text == "list" for c in comps)
    assert any(c.text == "select" for c in comps)
    assert any(c.text == "info" for c in comps)
    assert any(c.text == "monitor" for c in comps)
    assert any(c.text == "autodetect" for c in comps)


def test_no_completion_without_slash():
    comps = collect("status")
    assert len(comps) == 0


def test_alias_resolves_subcommands():
    comps = collect("/名字 ")
    assert any(c.text == "set" for c in comps)
    assert any(c.text == "alias" for c in comps)


# ── 新命令：wakeup / sleep / judge ─────────────────────────────────────────

def test_wakeup_completion():
    comps = collect("/wake")
    assert any(c.text == "/wakeup" for c in comps)
    assert any(c.text == "/wake" for c in comps)


def test_sleep_completion():
    comps = collect("/sl")
    assert any(c.text == "/sleep" for c in comps)


def test_standby_completion():
    comps = collect("/st")
    assert any(c.text == "/standby" for c in comps)


def test_judge_subcommand_completion():
    comps = collect("/judge ")
    assert any(c.text == "rule" for c in comps)
    assert any(c.text == "local" for c in comps)
    assert any(c.text == "llm" for c in comps)
