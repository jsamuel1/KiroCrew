"""Shell text reader: command bodies carried inside ssh-family options.

An ssh-family program runs several of its argument values as LOCAL shell commands
on this host, not as host operands:

* ``ssh``/``scp``/``sftp`` run the value of any ``*Command`` option
  (``ProxyCommand``, ``LocalCommand``, ``KnownHostsCommand``) through
  ``$SHELL -c`` -- ProxyCommand and KnownHostsCommand even with no connection.
* ``rsync`` runs the value of ``-e``/``--rsh`` (and a leading ``RSYNC_RSH=`` /
  ``RSYNC_CONNECT_PROG=`` environment assignment) as its remote-shell command.
* ``git`` runs a leading ``GIT_SSH_COMMAND=`` assignment as the transport it
  shells out to.

Each of those values is a command line this host executes, exactly as a
``bash -c`` body is, so this reader hands each one to the nested-command walk
that already judges a ``-c`` body. The ssh floor's own self-dial check
(``argv_floor_ssh``) is unaffected: it reads the same values for a different
question (does the value DIAL this host) and stays in place.

``RemoteCommand`` is the one ``*Command`` option excluded: its value runs on the
REMOTE host, not here, so feeding it to the local-command walk would refuse a
legitimate remote invocation.

The verb is read only in COMMAND position (the caller's ``programs`` attribution),
so a word that merely NAMES an ssh-family program as an argument -- ``grep -e rsync
-e 'x'`` -- does not get ssh/rsync option parsing. A verb reached only through a
wrapper (``env -u rsync ssh …``, ``sudo ssh …``) is attributed to the wrapper, not
to ssh, so its option values are not read here; that is the same wrapper residual
the rest of the ssh floor carries, not a gap this reader introduces.

Data and a single pass only; no tier decides anything here. The module imports
nothing from the floors above it, so the reader stays below them.
"""

from __future__ import annotations

# ssh-family program basenames. A path- or ``.exe``-qualified spelling is reduced
# to the bare verb by ``_family_verb`` before this is applied.
_SSH_FAMILY_VERBS: frozenset[str] = frozenset({"ssh", "scp", "sftp", "rsync"})


# Verbs that accept ssh ``-o Key=value`` options whose value is a local command.
_DASH_O_VERBS: frozenset[str] = frozenset({"ssh", "scp", "sftp"})


# Leading environment assignments whose value is a command line run FROM HERE.
_ENV_COMMAND_VARS: frozenset[str] = frozenset(
    {"git_ssh_command", "rsync_rsh", "rsync_connect_prog"}
)


def _family_verb(token: str) -> "str | None":
    """The ssh-family verb *token* names, or None.

    Tokens reaching this reader are already shlex-split (quotes removed), so only
    a path prefix and a ``.exe`` suffix need stripping to reach the bare verb. The
    result is lowercased, so a case-preserved caller's ``SSH`` still resolves.
    """
    base = token.strip().strip("\"'")
    base = base.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base.lower().endswith(".exe"):
        base = base[: -len(".exe")]
    base = base.lower()
    return base if base in _SSH_FAMILY_VERBS else None


def _is_local_command_key(key: str) -> bool:
    """True if ssh option *key* runs its value as a LOCAL command.

    Every current OpenSSH option whose value this host executes ends in
    ``Command`` (``ProxyCommand``, ``LocalCommand``, ``KnownHostsCommand``), so a
    suffix test future-proofs the set. ``RemoteCommand`` is the sole ``*Command``
    option run on the far side, so it is excluded -- feeding it here would refuse
    a legitimate remote command.
    """
    k = key.strip().lower()
    return k.endswith("command") and k != "remotecommand"


def _strip_dash_o_prefix(token: str) -> "str | None":
    """For a single-dash cluster, the text after an ``-o`` option letter, or None.

    An ``-o`` option can follow any run of other short flags getopt bundles in
    front of it (``-vo…``, scp/sftp ``-ro…``), and the deny tiers case-fold the
    whole command before this runs, so the OPTION LETTER's own case carries no
    information here -- ``-O`` (scp legacy protocol) and ``-o`` are the same byte
    by the time a payload reaches this reader. So the leading run is skipped
    without a per-verb valueless table: any alphanumeric that is not ``o`` is a
    flag letter to step over, and the first ``o`` starts the option. Stepping over
    a value-taking letter here can only OVER-read a later ``o`` as an option, the
    floor's documented safe direction. Returns the glued remainder after the
    ``o`` (possibly empty, meaning the keyword is the NEXT token).
    """
    if not token.startswith("-") or token.startswith("--"):
        return None
    body = token[1:]
    index = 0
    while index < len(body) and body[index].isalnum() and body[index].lower() != "o":
        index += 1
    if index < len(body) and body[index].lower() == "o":
        return body[index + 1 :]
    return None


def _split_key_value(spec: str) -> "tuple[str, str]":
    """Split an ssh option spec into ``(key, value)`` as OpenSSH's reader does.

    OpenSSH ends the keyword at the first whitespace OR ``=``, whichever comes
    first, then consumes that delimiter -- a single optional ``=`` with any
    surrounding whitespace -- before the value. So ``ProxyCommand =cmd`` and
    ``ProxyCommand = cmd`` both have the value ``cmd``, while an ``=`` INSIDE the
    value (``ProxyCommand env A=1 cmd``) is preserved. A leading ``=`` left on
    the value would make the deny tiers read ``=cmd`` instead of the ``cmd``
    OpenSSH runs.
    """
    stripped = spec.strip()
    for index, ch in enumerate(stripped):
        if ch == "=" or ch.isspace():
            key = stripped[:index].strip()
            rest = stripped[index:].lstrip()
            if rest.startswith("="):
                rest = rest[1:].lstrip()
            return key, rest
    return stripped, ""


def _rsync_rsh_glued_value(token: str) -> "str | None":
    """For an rsync single-dash cluster carrying ``-e``, its remote-shell value.

    ``-e`` selects the remote shell and consumes the rest of the cluster as its
    value, so ``-e``/``-ave``/``-essh …``/``-avessh …`` all resolve here. When
    ``e`` is the last flag the value is the NEXT token (signalled by returning an
    empty string); when text follows ``e`` in the same token that glued text IS
    the value. Returned in the token's ORIGINAL case.
    """
    if not token.startswith("-") or token.startswith("--"):
        return None
    cluster = token[1:]
    pos = cluster.lower().find("e")
    if pos == -1:
        return None
    return cluster[pos + 1 :]


def ssh_command_option_payloads(tokens: "list[str]", programs: "list[str]") -> "list[str]":
    """Local-command bodies carried inside an ssh-family invocation in *tokens*.

    *programs* is the per-token command-position attribution (``_argv_programs``):
    an ssh-family verb is parsed only where it is the program the argv runs, so a
    decoy word naming an ssh-family program as an argument is inert. Returns every
    value this host would execute: ssh ``*Command`` option values
    (``ssh``/``scp``/``sftp``), the rsync ``-e``/``--rsh`` remote-shell value, and
    a leading ``GIT_SSH_COMMAND=`` / ``RSYNC_RSH=`` / ``RSYNC_CONNECT_PROG=``
    assignment value. Each is a command line the nested-command walk then
    re-tokenizes and judges, so an ``-e "ssh -oProxyCommand=…"`` value descends on
    the next pass. Over-approximate by design: a value naming a self-dial or a
    harmless command is emitted too, and the deny tiers decide.
    """
    payloads: list[str] = []
    limit = len(tokens)

    # A command-leading ``NAME=value`` assignment that names a command run from
    # here. Attribution marks a leading-assignment position (before its
    # statement's program) with an empty program name and resets at every
    # statement boundary (``&&``, ``;``, ``|`` -- see ``_ends_argv``), so scanning
    # every empty-program position catches a ``GIT_SSH_COMMAND=`` leading a LATER
    # statement (``cd . && GIT_SSH_COMMAND=… git …``), not only the first.
    for index, token in enumerate(tokens):
        if index < len(programs) and programs[index]:
            continue
        eq = token.find("=")
        if eq <= 0 or token.startswith("-"):
            continue
        name = token[:eq]
        if all(ch.isalnum() or ch == "_" for ch in name) and name.lower() in _ENV_COMMAND_VARS:
            payloads.append(token[eq + 1 :])

    # One forward pass over the token list. A matched command's whole argv span
    # is scanned once, then the outer index jumps PAST it (``i = j``), so a line
    # of repeated ssh words -- each attributed to ``ssh`` and so each otherwise a
    # fresh command-position hit -- is linear, not quadratic, inside this
    # synchronous permission gate.
    i = 0
    while i < limit:
        verb = _family_verb(tokens[i])
        # Only parse where this family verb is the command the argv actually runs.
        if verb is None or i >= len(programs) or programs[i].lower() != verb:
            i += 1
            continue
        j = i + 1
        # Scan this command's own argv only: attribution marks every token of it
        # with this verb, so the span ends where the program name changes (the next
        # pipeline stage or statement), which bounds the scan without a separator set.
        while j < limit and j < len(programs) and programs[j].lower() == verb:
            token = tokens[j]
            if verb in _DASH_O_VERBS:
                glued = _strip_dash_o_prefix(token)
                if glued is not None:
                    if glued:  # ``-oKey=value`` / ``-o'Key value'``
                        key, value = _split_key_value(glued)
                        if _is_local_command_key(key) and value:
                            payloads.append(value)
                    # ``-o Key=value`` / ``-o 'Key value'``. A next token that
                    # itself starts with ``-`` is another option, not this one's
                    # argument -- consuming it would skip the real ``-o`` spec that
                    # follows (the case-folded deny view turns scp's valueless
                    # ``-O`` into a second ``-o``, so the spurious one must not eat
                    # the real spec).
                    elif j + 1 < limit and not tokens[j + 1].startswith("-"):
                        key, value = _split_key_value(tokens[j + 1])
                        if _is_local_command_key(key) and value:
                            payloads.append(value)
                        j += 1
            elif verb == "rsync":
                # Same guard: a next token starting with ``-`` is a flag, not the
                # remote-shell value, so a case-folded ``-E`` (``--executability``,
                # valueless) cannot make a trailing bare ``-e`` swallow the real
                # ``-e`` option that follows.
                if token in ("-e", "--rsh") and j + 1 < limit and not tokens[j + 1].startswith("-"):
                    payloads.append(tokens[j + 1])
                    j += 1
                elif token.lower().startswith("--rsh="):
                    payloads.append(token[len("--rsh=") :])
                else:
                    glued = _rsync_rsh_glued_value(token)
                    if glued is not None:
                        if glued:
                            payloads.append(glued)
                        elif j + 1 < limit and not tokens[j + 1].startswith("-"):
                            payloads.append(tokens[j + 1])
                            j += 1
            j += 1
        # The whole span was scanned; continue PAST it so no token is re-scanned.
        i = j

    return [p for p in payloads if p.strip()]
