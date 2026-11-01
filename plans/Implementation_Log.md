# Implementation ledger
Baseline: plans/From_Digits_to_Decisions_Project_Plan.md, explicitly selected by user.
Ruling: use a feature branch in the supplied checkout to keep results immediately accessible; no separate worktree or commits needed.
Ruling: Python 3.13 is the installed interpreter; validate this rather than install another interpreter.
Ruling: retain CPU algorithms; measure actual durations before reconsidering GPU complexity.
Sequence: environment -> regression tests -> learning modules -> UI interaction tests -> UI -> measured results, README and eight-slide presentation -> full verification.
Shared interfaces: learning modules return results consumed by app; normalized feature vectors used independently by classification and clustering. RL evaluation must not mutate training table.
Verification: pytest full suite, compileall, pip check, actual local server health check. No preexisting lint/security configuration.

Learning tests: observed four expected failures before implementation, then four passed. UI test: observed missing app failure before implementation; corrected AppTest relative-path setup to absolute path; complete suite five passed.
Measured CPU: classifier 0.074s, clustering 0.162s, agent 0.797s; no CUDA needed. Independent reviewer found no correctness blockers. Browser tested all three training flows and saved screenshots. Fixed plot stretching after inspecting screenshot (presentation-only change; verify browser and suite).
Docs: README, DEMO, results.json, eight offline HTML slides. pip check and compileall passed; local health endpoint returned ok. Pinned requirements reinstall matched installed versions. Original environment was created fresh for this implementation; no second clean environment installation performed.

Final browser capture: zero console errors, two chart extent warnings during chart mounting; verify chart visibly at rehearsal. All eight presentation sections loaded in browser. Screenshots saved in screenshots/. No network-disabled test; modules use bundled data and local environment. Human tasks: member names, rehearsal, submission channel/cutoff and submission.

November 1: confusion-cell inspection approved by user. Added regression test first; observed missing-selectors assertion fail, implemented UI filtering and matching-image selector; six full-suite tests passed. Updated README, demo and baseline. Preserved preexisting reinforcement.py edits. Dataset file confirmed inside sklearn package: data/digits.csv.gz.
