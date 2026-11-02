---
name: todo-dispatch
description: Work one section of the README TODO list by dispatching parallel agents that review and fix each item, then report status before anything is ticked off. Use when asked to "work through", "dispatch agents for", or "fix" a named section of the TODO list in README.md.
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

## 2. Partition by file ownership

Group items by the source file they touch, so no two agents edit the same file.
Each agent gets:

- the source file(s) it owns, and a **new** test file of its own
  (`package/tests/test_<area>.py`), or an existing test file no other agent touches;
- an explicit list of files it must not edit (the other agents' files, `README.md`,
  `tests/golden_values.py`, `tests/conftest.py`).

If an item needs a change in a file another agent owns, the agent reports it instead of
editing. Relay it to the owning agent with SendMessage while that agent is still
running.

## 3. Agent prompt checklist

Spawn the agents in one message, with `run_in_background: true`. Each prompt says:

- Read `CLAUDE.md` first and follow its code, docstring and test rules.
- The exact TODO items, quoted, and any WIP problems you noticed in step 1.
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

## 4. Integrate

- As each agent reports, relay any cross-file fixes it needs, and pass shared findings
  (such as a stale environment) to the agents still running.
- When all agents are done, run the full suite yourself and check `git status` for
  files nobody was supposed to touch.

## 5. Report before ticking anything off

Give the user one table: item, status, and a one-line summary of what changed. Follow
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
