"""Per-domain supervised loops (v-parallel-loops-2026-05-12).

Phase 1 of the parallel-loops refactor (see
`.claude/PLAN_parallel_loops.md`). Each loop owns one data domain,
runs on its own cadence, and has a per-iteration try/except so a
crash in one loop cannot wedge a sibling. Loops are registered with
`core.task_supervisor.TaskSupervisor` in `engine._build_supervisor`.
"""
