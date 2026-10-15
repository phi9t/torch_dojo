# Recover the original evaluation after a frontend disconnect
Status: resolved
Blocked by: none

Implement the first slice of ../spec.md. Add a regression at the real Emacs evaluation/session seam: delayed output completes while disconnected, reconnection recovers the original request and verifies the mutation happened once. Add disconnected-submit and restarted-kernel behavior as subsequent test slices. Preserve existing GPU objects and explicit shutdown ownership. Record commands/results and limitations.

## Verification

Implemented `insula/emacs/torch-dojo-session.el`, scoped to Dojo clients; existing package files and stock clients are unchanged. `insula/emacs/run.sh` loads the module. The capsule maps the build-time Doom source abbreviation `~/doom` without replacing any existing runtime file/symlink.

- `bash insula/emacs/verify_session.sh`: eight live ERT tests pass against owned GPU kernels; logs in `target/insula/session-final.log`.
- `bash insula/emacs/verify_startup.sh`: real terminal Doom startup and transport loading pass; `target/insula/session/startup.log`.
- `python3 -m unittest discover -s tests -p test_insula_plan.py`: existing role/workspace plan test passes.
- Shell syntax and `git diff --check` pass.

The original offline-completion regression failed before adding the transport (`target/insula/session-red.log`) and passed afterward. Review identified inherited lifecycle/restart and kernelspec-launch defects; additional regressions reproduced the lifecycle defects and all pass after correction. Standards and Spec review found no remaining material findings. An injected send failure after real delivery confirms no automatic replay. Terminal outcome history is bounded while incomplete outcomes remain retained.

Known boundary: generation checks use shell requests and may wait behind running/paused execution. Identities and outstanding outcomes are currently in-process only. Interactive responsiveness, SSH failures, debugger UI, production cache helpers, and additional language integrations remain in subsequent tickets. Usage is documented in `insula/emacs/session.md`.

## Landing verification

Before local landing, terminal startup and all eight live Emacs/GPU regressions passed again (`target/insula/session-land.log`). The full unittest suite passed in the Torch insula, with seven optional-environment skips (`target/insula/session-land-suite-insula.log`). The initial host `make test` attempt lacked Torch; no dependencies were installed on the host.
