# Automatic Agent Delegation

## MANDATORY: Auto-Trigger Rules

Claude MUST automatically delegate to agents without waiting for the user to ask. These are NOT optional.

### 1. PLAN before implementing

**Trigger**: User asks for a new feature, refactoring, or any task touching 3+ files.
**Action**: Launch the **planner** agent FIRST. Present the plan. Wait for user approval before writing code.

### 2. TDD for new code and bug fixes

**Trigger**: User asks to write a new function, fix a bug, add a feature, or change behavior.
**Action**: Launch the **tdd-guide** agent. Write a failing test FIRST, then implement.

### 3. Review after writing code

**Trigger**: After ANY code modification is complete (implementation done, tests passing).
**Action**: Launch **code-reviewer** + **security-reviewer** agents in parallel.

### 4. Python review for Python files

**Trigger**: After modifying any `.py` file.
**Action**: Launch **python-reviewer** agent (runs in parallel with code-reviewer).

### 5. Build fix when things break

**Trigger**: A test run fails, import error occurs, or the app won't start.
**Action**: Launch **build-error-resolver** agent automatically.
**Do NOT**: Retry the same command hoping it works. Diagnose first.

## Parallel Execution

Always run independent agents in parallel:

```
GOOD: After implementation is done
  -> Launch code-reviewer + security-reviewer + python-reviewer simultaneously

BAD: Run code-reviewer, wait, then security-reviewer, wait, then python-reviewer
```

### 6. Database changes

**Trigger**: User asks to change schema, add tables, modify queries, or optimize DB performance.
**Action**: Launch **database-reviewer** agent.

### 7. Architecture decisions

**Trigger**: User asks about system design, service boundaries, data flow, or major structural changes.
**Action**: Launch **architect** agent.

### 8. Code cleanup

**Trigger**: User asks to clean up, remove dead code, or reduce file sizes.
**Action**: Launch **refactor-cleaner** agent.

### 9. Documentation

**Trigger**: After significant feature work is complete, or user asks to update docs.
**Action**: Launch **doc-updater** agent.

### 10. Autonomous loops

**Trigger**: User asks to set up monitoring, polling, or recurring tasks.
**Action**: Launch **loop-operator** agent.

## Agent Selection Quick Reference

| User says... | Agent(s) to launch |
|---|---|
| "add/build/create/implement..." | planner -> tdd-guide -> code-reviewer |
| "fix/debug/repair..." | tdd-guide -> code-reviewer |
| "refactor/clean/reorganize..." | refactor-cleaner -> code-reviewer |
| "why is X broken / X doesn't work" | build-error-resolver |
| "review this / is this safe" | code-reviewer + security-reviewer |
| "change the schema / add a table / slow query" | database-reviewer |
| "how should we structure / design / architect" | architect |
| "clean up / remove dead code / shrink this file" | refactor-cleaner |
| "update docs / add documentation" | doc-updater |
| "monitor / poll / keep checking" | loop-operator |
