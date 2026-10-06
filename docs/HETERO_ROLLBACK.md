# Rollback and service restoration

The original Strata checkout and `F:\Strata-data` model are kept intact. The initial Hetero base is `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`; the first pushed plan checkpoint is `be9ecba3491309a8fe67721ab313477e4e96d836`. Keep each tested feature on a phase branch, and inspect `git status` before changing branches. Preserve uncommitted work. Revert an accepted feature with an explicit revert commit rather than erasing published history.

Experimental backends default off until their measurement and quality gates pass. An unavailable or slower Arc/NPU route falls back to the existing CUDA/CPU route. A failed RAM lock is reported and must not be described as a resident table. Restore the previous loopback run configuration to roll back a placement experiment; do not rewrite the historical config or remove the original GGUF.

## Temporary WSL/ComfyUI hold

The user explicitly allowed temporary shutdown of WSL and ComfyUI on 2026-10-06. Before the change, Ubuntu-24.04 was running, `comfyui.service` was enabled/active with MainPID 229, and that process held `/dev/dxg`. State is recorded in `bench/hetero/20261006-00-inventory/wsl-original-state.json`.

After a successful stop and WSL shutdown, a later observation found Ubuntu/ComfyUI running again; the source of the restart is unresolved. The admission observer is being changed to use only WSL's running-distro list and never `wsl -d`, because an observation must not boot a stopped distro. To hold ComfyUI off during the authorized experiments, its startup is temporarily disabled and its service stopped. It is not removed or reconfigured. The hold and restoration command are recorded in `wsl-autostart-hold.json`. The subsequent check returned `MainPID=0`, `ActiveState=inactive`, `SubState=dead`, `UnitFileState=disabled`, and no `/dev/dxg` client before WSL shutdown.

When the GPU experiments finish or the user asks to restore the original state, run:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl enable --now comfyui.service
wsl -d Ubuntu-24.04 -- systemctl show comfyui.service -p ActiveState -p SubState -p MainPID -p UnitFileState
```

Verify enabled/active, the expected process identity and the application's health endpoint; record a new restoration receipt. Starting Ubuntu to restore the originally running service is intentional. Do not restore a different service or assume that the old boot ID/PID still applies. The old Strata process belongs to another active chat and is preserved; its GPU measurements are not interrupted by this project.
