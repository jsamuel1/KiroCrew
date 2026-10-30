"""An ssh-family option value that names a LOCAL command is judged as a command.

``ssh``/``scp``/``sftp`` run the value of a ``*Command`` option through ``$SHELL -c``
on this host, ``rsync`` runs its ``-e``/``--rsh`` value, and a leading
``GIT_SSH_COMMAND=``/``RSYNC_RSH=`` assignment is a command line run from here. The
self-protection floors (credential mint, restart, kill) must see each of those bodies
exactly as they see a ``bash -c`` body. ``shell_normalizer_ssh_options`` feeds them to
the nested-command walk; this module pins that every allowed row from the filed report
is now refused, and that an ordinary ssh ProxyCommand and other legitimate option
values stay allowed.
"""

from __future__ import annotations

import pytest

from kiro_crew import security
from kiro_crew.security import argv_floor as _argv_floor
from kiro_crew.security import is_denied
from kiro_crew.security.shell_normalizer import _argv_programs
from kiro_crew.security.shell_normalizer_ssh_options import ssh_command_option_payloads


def _payloads(tokens):
    return ssh_command_option_payloads(tokens, _argv_programs(tokens))


# Match the ssh-self family: start with the macOS libc/libproc handle caches unbuilt.
pytestmark = pytest.mark.usefixtures("fresh_darwin_libc_handles")


# Bodies that must be REFUSED when carried as an ssh-family option value. Each is a
# self-protection command the gate already refuses when wrapped in ``bash -c``.
_REFUSED = [
    "ssh -oProxyCommand='kirocrew token' x",
    "ssh -o ProxyCommand='kirocrew token' x",
    "ssh -o 'ProxyCommand kirocrew token' x",
    "ssh -oProxyCommand='bash -c \"kirocrew token\"' x",
    "ssh -oPermitLocalCommand=yes -oLocalCommand='kirocrew token' x",
    "ssh -o PermitLocalCommand=yes -o 'LocalCommand kirocrew token' x",
    "ssh -oKnownHostsCommand='kirocrew token' x",
    "ssh -voProxyCommand='kirocrew token' x",
    "SSH -OPROXYCOMMAND='kirocrew token' x",
    "scp -oProxyCommand='kirocrew token' f x:/tmp/",
    "sftp -oProxyCommand='kirocrew token' x",
    "rsync -e \"ssh -oProxyCommand='kirocrew token'\" f x:/tmp/",
    "rsync -e 'kirocrew token' f x:/tmp/",
    "rsync --rsh='kirocrew token' f x:/tmp/",
    "rsync -ave 'kirocrew token' f x:/tmp/",
    "GIT_SSH_COMMAND='kirocrew token' git fetch x",
    "RSYNC_RSH='kirocrew token' rsync f x:/tmp/",
    "ssh -oProxyCommand='kirocrew restart' x",
    "ssh -oProxyCommand='pkill -f kirocrew' x",
    "scp -oProxyCommand='pkill -f kirocrew' f x:/tmp/",
    "sftp -oKnownHostsCommand='kirocrew restart' x",
    # A '=' inside the value must not break the whitespace-form keyword split:
    # OpenSSH ends the keyword at the first whitespace or '=', whichever is first.
    "ssh -o 'ProxyCommand env A=1 kirocrew token' x",
    "ssh '-oProxyCommand env A=1 kirocrew restart' x",
    # A whitespace-then-'=' delimiter must be consumed, not kept on the value.
    "ssh -o 'ProxyCommand =kirocrew token' x",
    "ssh -o 'ProxyCommand = kirocrew token' x",
    # The case-folded deny view turns scp's valueless -O into a second -o and
    # rsync's valueless -E into -e; the spurious one must not eat the real option.
    "scp -O -o ProxyCommand='kirocrew token' f x:/tmp/",
    "scp -O -o 'ProxyCommand kirocrew restart' f x:/tmp/",
    "scp -roProxyCommand='kirocrew token' f x:/tmp/",
    "rsync -aE -e 'kirocrew restart' f x:/tmp/",
    "rsync -avE --rsh='kirocrew token' f x:/tmp/",
    # A GIT_SSH_COMMAND= that leads a LATER statement after '&&' chaining.
    "cd . && GIT_SSH_COMMAND='kirocrew token' git ls-remote x",
]


# Legitimate commands that merely NAME an ssh-family program as an argument must
# keep their own option parsing -- a decoy word must not trigger ssh/rsync rules.
_ALLOWED_DECOY = [
    "grep -e rsync -e 'kirocrew token' notes.txt",
    "git log -S rsync -e 'kirocrew token'",
    "grep -c ssh -o 'LocalCommand kirocrew restart' notes.txt",
    # RemoteCommand runs on the far side, not here.
    "ssh -o RemoteCommand='pkill -f kirocrew' devbox",
]


# Legitimate option values that must STAY allowed: an ordinary ProxyCommand, a token
# substitution, an rsync remote-shell with a port, an ``-i`` identity, and a ``-e``
# that only belongs to a non-ssh program (grep).
_ALLOWED = [
    "ssh -oProxyCommand='ssh -W %h:%p bastion' host",
    "ssh -o ProxyCommand='nc %h %p' remotebox",
    "rsync -e 'ssh -p 2222' f x:/tmp/",
    "GIT_SSH_COMMAND='ssh -i ~/.ssh/k' git fetch origin",
    "grep -e 'kirocrew token' notes.txt",
    # Common invocations whose valueless flags share a letter with -o/-e: the
    # scan must not read their file operands as an option value.
    "rsync -aE f x:/tmp/",
    "scp -O f x:/tmp/",
    "cd . && ls",
]


@pytest.fixture(autouse=True)
# nosemgrep: kirocrew.test-autouse-shared-monkeypatch -- same own-host pin the ssh-self suite uses, to keep the DNS daemon thread and network out of the test
def _pin_own_host_cache(monkeypatch):
    # Pin the own-host cache with the resolved-once latch set so ``_own_host_names``
    # never seeds the module globals or spawns the real resolver thread (a
    # no-test-side-effects violation), the kernel address table reads as published,
    # and unresolved names answer "not self" so the remote-host allow cases stay
    # allowed. Mirrors ``TestSandboxEscapeSshSelf._pin_own_host_cache``.
    own = security.socket.gethostname().strip().lower()
    pinned = frozenset(name for name in {own, own.split(".", 1)[0]} if name)
    monkeypatch.setattr(_argv_floor, "_OWN_HOST_NAMES_CACHE", pinned)
    monkeypatch.setattr(_argv_floor, "_OWN_HOST_RESOLVE_DONE", True)
    monkeypatch.setattr(_argv_floor, "_NETLINK_ADDRS_PUBLISHED", True)
    monkeypatch.setattr(_argv_floor, "_OWN_HOST_RESOLVE_NEXT_TRY", float("inf"))
    monkeypatch.setattr(_argv_floor, "_resolved_host_verdict", lambda host, **_kw: False)


@pytest.mark.parametrize("command", _REFUSED)
def test_ssh_command_option_body_is_refused(command):
    assert is_denied(command) is not None, command


@pytest.mark.parametrize("command", _ALLOWED + _ALLOWED_DECOY)
def test_legitimate_option_value_stays_allowed(command):
    assert is_denied(command) is None, command


class TestExtractor:
    """``ssh_command_option_payloads`` yields the local-command bodies verbatim."""

    def test_glued_and_separated_and_whitespace_spellings(self):
        assert "kirocrew token" in _payloads(["ssh", "-oproxycommand=kirocrew token", "x"])
        assert "kirocrew token" in _payloads(["ssh", "-o", "proxycommand=kirocrew token", "x"])
        assert "kirocrew token" in _payloads(["ssh", "-o", "proxycommand kirocrew token", "x"])

    def test_value_may_contain_equals_in_the_whitespace_form(self):
        # The keyword ends at the first whitespace; the '=' belongs to the value.
        assert "env a=1 kirocrew token" in _payloads(
            ["ssh", "-o", "proxycommand env a=1 kirocrew token", "x"]
        )

    def test_remotecommand_runs_on_the_far_side_so_is_not_a_local_body(self):
        assert _payloads(["ssh", "-oremotecommand=kirocrew token", "x"]) == []

    def test_rsync_remote_shell_and_leading_env_assignment(self):
        assert "kirocrew token" in _payloads(["rsync", "-e", "kirocrew token", "f", "x:/tmp/"])
        assert "kirocrew token" in _payloads(
            ["git_ssh_command=kirocrew token", "git", "fetch", "x"]
        )

    def test_verb_only_parsed_in_command_position(self):
        # ``grep -e`` is not rsync's remote-shell flag, and a ``rsync`` word that is
        # merely grep's argument must not be read as the program.
        assert _payloads(["grep", "-e", "kirocrew token", "notes.txt"]) == []
        assert _payloads(["grep", "-e", "rsync", "-e", "kirocrew token", "notes.txt"]) == []

    def test_delimiter_whitespace_and_equals_are_consumed(self):
        # OpenSSH drops the keyword delimiter; the value must not keep a leading '='.
        assert "kirocrew token" in _payloads(["ssh", "-o", "proxycommand =kirocrew token", "x"])
        assert "kirocrew token" in _payloads(["ssh", "-o", "proxycommand = kirocrew token", "x"])

    def test_spurious_o_from_case_fold_does_not_eat_the_real_option(self):
        # scp's valueless -O folds to -o in the deny view; the spurious -o must not
        # consume the following real -o spec (which starts with '-').
        assert "kirocrew token" in _payloads(
            ["scp", "-o", "-o", "proxycommand=kirocrew token", "f", "x:/tmp/"]
        )
        # rsync's valueless -E folds to -e; a trailing bare -e must not swallow the
        # following real -e option.
        assert "kirocrew token" in _payloads(
            ["rsync", "-ae", "-e", "kirocrew token", "f", "x:/tmp/"]
        )

    def test_leading_assignment_of_a_later_statement_is_inspected(self):
        toks = ["cd", ".", "&&", "git_ssh_command=kirocrew token", "git", "ls-remote", "x"]
        assert "kirocrew token" in _payloads(toks)

    def test_repeated_verb_scan_is_linear_and_not_duplicated(self):
        # A line of repeated ssh words must be scanned once (not restarted at each
        # verb token), so neither the payload list nor the time is quadratic.
        assert _payloads(["ssh"] * 4000) == []
        out = _payloads(["ssh", "-oproxycommand=a b"] * 500)
        assert len(out) == 500
