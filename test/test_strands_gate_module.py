"""The Strands gate module's own behaviour, run against stand-in ``strands`` modules.

The gate module is loaded by the Strands CLI's Python worker, never by Kiro Crew, and
the Strands SDK is not a Kiro Crew dependency. So these tests load the module from its
packaged path with small stand-ins for exactly the ``strands`` names it imports, and
drive the one behaviour that lives in its own code: what happens to the next prompt
when a cancel left an approval interrupt open. On Strands CLI 0.2.3 that state is
saved with the session, and Strands refuses every later prompt and reload with "must
resume from interrupt"; the gate denies the open interrupts first, so the tool never
runs and the session keeps working. The same behaviour was checked on the real SDK
(``strands-agents`` 1.63.0) with a scripted model: red without the fix, green with it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

GATE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "kiro_crew"
    / "agent_sdk"
    / "gate_extensions"
    / "strands"
    / "kiro_crew_strands_gate.py"
)


class _Interrupt:
    def __init__(self, response: Any = None) -> None:
        self.response = response


class _InterruptState:
    def __init__(self) -> None:
        self.activated = False
        self.interrupts: dict[str, _Interrupt] = {}
        self.deactivated = False

    def deactivate(self) -> None:
        self.activated = False
        self.interrupts = {}
        self.deactivated = True


class _Hooks:
    def add_hook(self, _hook: Any) -> None:
        pass


class _LocalSandbox:
    pass


class _MCPAgentTool:
    pass


def _stand_in_strands(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install stand-ins for every ``strands`` name the gate imports; return ``Agent``."""

    class Agent:
        #: What the real ``Agent.stream_async`` does with interrupt state, reduced to
        #: the rule the fix answers: while an interrupt is open, only a list of
        #: interrupt responses is accepted.
        keep_interrupt_open = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.hooks = _Hooks()
            self.sandbox = _LocalSandbox()
            self._interrupt_state = _InterruptState()
            self.prompts: list[Any] = []

        async def stream_async(self, prompt: Any = None, **kwargs: Any):
            state = self._interrupt_state
            if state.activated:
                if not (isinstance(prompt, list) and all("interruptResponse" in p for p in prompt)):
                    raise TypeError("must resume from interrupt with list of interruptResponse's")
                for item in prompt:
                    answer = item["interruptResponse"]
                    state.interrupts[answer["interruptId"]].response = answer["response"]
                if not type(self).keep_interrupt_open:
                    state.activated = False
                    state.interrupts = {}
            self.prompts.append(prompt)
            yield {"result": "ok"}

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    local = module(
        "strands.sandbox.not_a_sandbox_local_environment",
        NotASandboxLocalEnvironment=_LocalSandbox,
        _stream_process=lambda *a, **k: None,
    )
    module("strands", Agent=Agent)
    module("strands.hooks", HookProvider=object, HookRegistry=object)
    module("strands.hooks.events", BeforeToolCallEvent=object)
    module("strands.sandbox", not_a_sandbox_local_environment=local)
    module("strands.sandbox.types", ExecutionResult=object)
    module("strands.session", SnapshotSessionManager=object)
    module("strands.storage", LocalFileStorage=object)
    module("strands.tools", mcp=None)
    module("strands.tools.mcp", MCPAgentTool=_MCPAgentTool)
    slots = (
        "shell",
        "read",
        "write",
        "edit",
        "web_fetch",
        "web_search",
        "programmatic_tool_caller",
        "subagent",
    )

    def builtin_tools(*_a: Any, **_k: Any) -> dict:
        return {slot: types.SimpleNamespace(tool_name=slot) for slot in slots}

    module("strands_harness.agent", _builtin_tools=builtin_tools)
    module("strands_harness", create_harness=lambda *a, **k: Agent(), BUILTIN_TOOL_NAMES=slots)
    return Agent


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> type:
    for name in (
        "KIROCREW_STRANDS_PROJECT",
        "KIROCREW_STRANDS_GATE_MARKER",
        "KIROCREW_STRANDS_GATE_SESSION",
        "KIROCREW_STRANDS_WITHHELD",
    ):
        monkeypatch.delenv(name, raising=False)
    agent_cls = _stand_in_strands(monkeypatch)
    spec = importlib.util.spec_from_file_location("kiro_crew_strands_gate_under_test", GATE)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    agent_cls.gate = gate
    return agent_cls


def _run(agent: Any, prompt: Any) -> None:
    async def drive() -> None:
        async for _event in agent.stream_async(prompt):
            pass

    asyncio.run(drive())


def test_an_interrupt_a_cancel_left_open_is_denied_before_the_next_prompt(gated) -> None:
    agent = gated()
    agent._interrupt_state.activated = True
    agent._interrupt_state.interrupts = {"v1:before_tool_call:tooluse_1:x": _Interrupt()}

    _run(agent, "hello again")

    # The open interrupt was answered with a DENY first, then the prompt ran.
    assert agent.prompts == [
        [
            {
                "interruptResponse": {
                    "interruptId": "v1:before_tool_call:tooluse_1:x",
                    "response": False,
                }
            }
        ],
        "hello again",
    ]
    assert agent._interrupt_state.activated is False


def test_a_prompt_with_no_open_interrupt_runs_unchanged(gated) -> None:
    agent = gated()
    _run(agent, "hello")
    assert agent.prompts == ["hello"]


def test_the_harness_own_interrupt_answers_pass_straight_through(gated) -> None:
    """The CLI answering an approval is not a new prompt and is never pre-denied."""
    agent = gated()
    agent._interrupt_state.activated = True
    agent._interrupt_state.interrupts = {"i1": _Interrupt()}
    answer = [{"interruptResponse": {"interruptId": "i1", "response": True}}]

    _run(agent, answer)

    assert agent.prompts == [answer]


def test_a_deny_pass_that_stays_paused_never_blocks_the_prompt(gated) -> None:
    """Bounded: if denying keeps the agent paused, the state is cleared and the prompt runs."""
    gated.keep_interrupt_open = True
    agent = gated()
    agent._interrupt_state.activated = True
    agent._interrupt_state.interrupts = {"i1": _Interrupt()}

    _run(agent, "hello again")

    # One deny was sent; the stand-in kept the agent paused, so the gate cleared the
    # state rather than leave the session refusing every prompt.
    assert agent.prompts[0] == [{"interruptResponse": {"interruptId": "i1", "response": False}}]
    assert agent.prompts[-1] == "hello again"
    assert agent._interrupt_state.deactivated is True


def _event(name: str, tool: Any) -> Any:
    return types.SimpleNamespace(tool_use={"name": name}, selected_tool=tool)


def _harness_builtins(gate: Any) -> dict:
    """The built-ins as the harness's own factory builds them, through the gate's hook."""
    return sys.modules["strands_harness.agent"]._builtin_tools({})


@pytest.mark.parametrize("name", ["shell", "read", "write", "edit", "web_fetch"])
def test_a_harness_builtin_is_owned_by_the_harness(gated, name) -> None:
    tool = _harness_builtins(gated.gate)[name]
    assert gated.gate._tool_identity(_event(name, tool)) == (name, "builtin")


@pytest.mark.parametrize("name", ["read", "shell", "web_fetch"])
def test_a_project_tool_wearing_a_builtin_name_is_custom(gated, name) -> None:
    """A ``read`` built-in is auto-approved by the read-only tier, so a project tool
    named ``read`` (one that marks mail as read, say) is ``custom``, not the built-in."""
    _harness_builtins(gated.gate)
    own = types.SimpleNamespace(tool_name=name)
    assert gated.gate._tool_identity(_event(name, own)) == (name, "custom")


def test_a_tool_that_is_not_an_object_the_harness_built_is_custom(gated) -> None:
    assert gated.gate._tool_identity(_event("read", object())) == ("read", "custom")
    assert gated.gate._tool_identity(_event("read", None)) == ("read", "custom")


def test_a_factory_result_of_another_shape_records_no_builtin(gated) -> None:
    """A renamed or missing slot breaks the one-per-slot shape: nothing is recorded."""
    factory = sys.modules["strands_harness.agent"]
    renamed = {"shell": types.SimpleNamespace(tool_name="read")}
    original = gated.gate._original_builtin_tools
    try:
        gated.gate._original_builtin_tools = lambda *a, **k: renamed
        tools = factory._builtin_tools({})
    finally:
        gated.gate._original_builtin_tools = original
    assert gated.gate._tool_identity(_event("read", tools["shell"])) == ("read", "custom")
