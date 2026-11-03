---
name: todo-dispatch
description: Work one section of the README TODO list, fixing items inline or dispatching parallel agents only when the items are distinct enough that agents are cheaper, then report status before anything is ticked off. Use when asked to "work through", "dispatch agents for", or "fix" a named section of the TODO list in README.md.
argument-hint: "<TODO section heading, e.g. Documentation>"
---

# Dispatch agents on a README TODO section

Target section: `$ARGUMENTS`. If no section is given, list the `### ` headings under
`## TODO` in `README.md` and ask which one. Touch only the named section; leave the
others alone even if an agent spots something related.

## 1. Gather context (do this yourself, before spawning anything)

- Read the target section of `README.md` and list its items.
- Run `git status --short` and `git diff`. The user often has uncommitted
  work-in-progress on these items. Note which items it touches and anything that
  already looks wrong, so agents can be told to review it rather than overwrite it.
- Check the test env imports the working tree, not a stale install:
  `conda run -n gifba_test python -c "import gifba; print(gifba.__file__)"`.
  It should point into `package/gifba/`. If it points into `site-packages`, tell the
  user and suggest `conda run -n gifba_test pip install -e "./package[test]"`. Until
  that is done, agents must run tests with `PYTHONPATH=package` prefixed.

## 2. Decide whether to dispatch at all

Every agent starts cold: it re-reads `CLAUDE.md`, its context set and the WIP, and its
report has to be read and checked afterwards. That overhead only pays off when the work
is genuinely split. Before spawning anything, group the items by similarity. Items are
**similar** when they share a file or module, the same kind of change (e.g. docstrings
across several modules, one rename applied in several places, the same bug pattern in
sibling functions), or the same context set. A group of similar items counts as one
task.

- **Work inline (no agents)** when there are fewer than three distinct tasks after
  grouping, when the items are all similar, or when the whole section is Light-tier work
  you can finish with the context you already gathered in step 1. Do the work yourself,
  following the same rules as the agent prompt checklist in step 4, then go to step 5.
- **Dispatch** only when there are several (three or more) distinct tasks, each needs
  its own substantial context set that overlaps little with the others, and splitting
  them clearly costs fewer tokens than loading all of that context into this session.
  One agent per distinct task, never one per similar item.
- **Mixed sections**: do the small or similar tasks inline and dispatch only the
  distinct, context-heavy ones, if there are still at least two of those. Otherwise do
  everything inline.

State the decision and a one-line reason (task count after grouping, and why agents
are or are not cheaper) before doing any work. If unsure, work inline.

## 3. Partition by file ownership (dispatch only)

Group the tasks from step 2 by the source file they touch, so no two agents edit the
same file. Similar items stay together in one partition. Each agent gets:

- the source file(s) it owns, and a **new** test file of its own
  (`package/tests/test_<area>.py`), or an existing test file no other agent touches;
- an explicit list of files it must not edit (the other agents' files, `README.md`,
  `tests/golden_values.py`, `tests/conftest.py`).

If an item needs a change in a file another agent owns, the agent reports it instead of
editing. Relay it to the owning agent with SendMessage while that agent is still
running.

### Context set per agent

Build each agent's context set from the graphify code graph (repository id: the
owner/name from `git remote get-url origin`), not from a directory sweep:

- `graphify_find` for each symbol the items name, to locate its definition.
- `graphify_file_neighbors` on each owned file for its direct imports and dependents.
- `graphify_callers` for any function whose signature or behaviour may change.
- `graphify_tests_for` on the owned files and changed symbols for the relevant tests.

The context set is the owned files, plus those graph hits, plus any file in
`git status`/`git diff` that touches the items (the index is a snapshot and can miss
uncommitted work). Keep it to direct (one-hop) connections; drop files the graph links
only through broad utilities unless an item is about them.

If graphify is unavailable or the repo is not indexed, say so and build the set by
hand with targeted grep for the item's symbols instead.

## 4. Agent prompt checklist

Spawn the agents in one message, with `run_in_background: true`. Pick `model` and
`effort` per agent from its partition's hardest item, using what step 1 showed you:

| Tier | Use when the partition is | `model` | `effort` |
|---|---|---|---|
| Light | Docs, docstrings, comments, README, type hints, renames, config or path fixes; no behaviour change | `haiku` | `low` |
| Standard | A contained bug fix or small feature in one module, with a test; signature changes with few callers | `sonnet` | `medium` |
| Heavy | Numerical or solver behaviour, anything that could move golden values, cross-module refactors, unclear root cause, or reviewing substantial user WIP | `opus` | `high` |

When unsure between two tiers, take the higher one. State the tier and a one-line reason
for each agent before spawning. If a Light or Standard agent reports that the item is
harder than expected (unclear root cause, golden values move, needs a design decision),
tell the user what it found and recommend a tier. Never stop an agent or respawn a
partition on a higher tier without the user's explicit approval; until they answer,
leave the agent running or finished as it is.

Each prompt says:

- Read `CLAUDE.md` first and follow its code, docstring and test rules.
- The exact TODO items, quoted, and any WIP problems you noticed in step 1.
- Its context set from step 3, as a file list. Read only those files (plus
  `CLAUDE.md`); do not explore the package or examples directories broadly. If it
  needs more, use the graphify tools (`graphify_find`, `graphify_callers`,
  `graphify_file_neighbors`, `graphify_tests_for`) or a targeted grep to find the one
  file it needs, and name any file added this way in its report.
- Confirm each bug is real before fixing it, and report the evidence (model files,
  notebooks, golden values, a reproduction). "Not a bug" or "WIP was already right" is
  a valid outcome.
- Review the user's WIP critically: keep what is right, fix what is wrong, and do not
  revert their chosen direction without saying why.
- Grep for callers of anything whose signature changes, and report callers outside the
  agent's files.
- Do not change golden values without re-measuring and flagging it. Do not add `xfail`.
- The exact test command, with `PYTHONPATH=package` if step 1 found a stale install.
- Do not commit and do not modify the environment.
- Final report: per item, the status (fixed / already fixed by WIP / not a bug /
  partial), what changed, evidence, test results, and decisions for the user.

## 5. Integrate

- As each agent reports, relay any cross-file fixes it needs, and pass shared findings
  (such as a stale environment) to the agents still running.
- When all agents are done (or you finished the inline work), run the full suite yourself and check `git status` for
  files nobody was supposed to touch.

## 6. Report before ticking anything off

Give the user one table: item, status, tier used (`inline` for items you did
yourself; note any escalation), and a one-line
summary of what changed. Follow
it with a numbered list of decisions that need their input, and any related defects
found but left out of scope. **Do not edit the TODO list until the user confirms.**
Then remove the finished items (or the whole section, if it is empty).

Do not commit or push anything. Instead, provide the user copy/paste snippets of
the code you recommend for commit. When committing, recommend stage changes with 
`git add -A` and use a commit message that lists the TODO items fixed, e.g.:
```
git add -A
git commit -m "Fix README TODO: <item 1>, <item 2>, <item 3>"
``` 
