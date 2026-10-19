# Thermal safety — SOLVED (machine has marginal cooling; thermally shut down before)

## The working solution: a fixed kernel CPU cap on the user app slice
Heat comes from BOTH a pegged CPU core (boost) AND GPU inference. Throttling the CPU
also PACES GPU jobs (their CPU feeder side is capped -> GPU util/power drop too), so a
single CPU cap controls the whole system's heat.

Applied two ways (belt + suspenders), value = 90% of one core:
  systemctl --user set-property app.slice CPUQuota=90%     # durable, needs D-Bus (interactive shell)
  echo "90000 100000" > /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/cpu.max  # direct, no D-Bus

Measured on this box (with a GPU generation job running):
  cap 50% -> 63-67C / GPU 112W ;  90% -> ~76C ;  100% -> 79C ;  200% -> 91C (danger).
Single boosting core is the hottest case; 90% bounds ALL job types at ~76C.

## What did NOT work / not available
- GPU power limit (`nvidia-smi -pl`): needs root (Insufficient Permissions). Min 400W anyway.
- `systemctl --user set-property` from a DETACHED script: silently fails (no D-Bus session).
  -> use the DIRECT cgroup write for any background governor.
- Dynamic governor via STOP/CONT or name-matching: unreliable (missed multiprocessing kids).

## Re-apply after reboot/logout (the cap is runtime state):
  systemctl --user set-property app.slice CPUQuota=90%
Raising it requires re-measuring temps. Do NOT exceed ~100% without servicing cooling.
