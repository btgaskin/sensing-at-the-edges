# Friction and Stiction in the Ledge Tasks

A methodology note on the contact model the cliff and ledge tasks actually
exercise. It is deliberately logic-first rather than paper-style: it states the
force laws, their limiting behavior, and which of them governs the runs in this
repository. The laws themselves live in the `jax-evogym` simulator; the
parameter values are set here, in `config/sim/`.

## Scope

- Run family: the `sensing_edges` bilateral goal-cliff and ledge tasks.
- Current sim values under review:
  - `spring_stiffness_scale = 0.8`
  - `floor_static_friction_const = 0.4`
  - `floor_stiction_stiffness = 20000`
  - `floor_stiction_switch_speed = 0.05`
- Reward terms (`sensing_edges/fitness.py`):
  - `OUTWARD_PROGRESS_SCALE = 10`
  - `REVERSAL_PROGRESS_SCALE = 20`
  - `TARGET_PROGRESS_SCALE = 30`
  - `GOAL_TOUCH_BONUS = 40`
  - `CLIFF_FALL_PENALTY = 20`
  - `LOWER_LEDGE_PENALTY_CAP = 10`

## Current Contact Laws

There are two dynamic-friction saturation laws in the simulator.

### 1. Ground-plane contact

This is the legacy law used by `compute_forces()` when a point penetrates the
global ground plane (`y < 0`):

`F_dyn_ground = -c_g * mu_d * N * tanh(v_t * N * mu_d)`

With stiction enabled for ground contact:

`F_stick_ground = clip(L_t - k_stick * v_t, -mu_s * N, mu_s * N)`

`alpha = tanh(|v_t| / v_switch)`

`F_ground = (1 - alpha) * F_stick_ground + alpha * F_dyn_ground`

For the default no-load case on the ground plane, `L_t = 0`, so the stick term
reduces to `clip(-k_stick * v_t, +/- mu_s * N)`.

### 2. Static-terrain contact

This is the law used for fixed floor cells, ledges, and slope geometry:

`F_dyn_static = -c_f * mu_d * N * tanh(v_t / (mu_d * N))`

`F_stick_static = clip(L_t - k_stick * v_t, -mu_s * N, mu_s * N)`

`alpha = tanh(|v_t| / v_switch)`

`F_static = (1 - alpha) * F_stick_static + alpha * F_dyn_static`

The important difference is the argument to `tanh`.

## Limiting-Case Analysis

### Low-speed slope

For `|x| << 1`, `tanh(x) ~= x`.

Ground-plane dynamic friction becomes:

`F_dyn_ground ~= -c_g * v_t * (mu_d * N)^2`

So its low-speed slope depends on `N^2`.

Static-terrain dynamic friction becomes:

`F_dyn_static ~= -c_f * v_t`

So its low-speed slope is independent of `mu_d` and `N`.

This is the main logic distinction in the current simulator.

### High-speed saturation

As `|v_t| -> inf`, both laws saturate at a Coulomb-like cap:

`|F_dyn| -> c * mu_d * N`

### Zero-velocity behavior

At `v_t = 0`:

- `alpha = 0`
- the model uses the stick branch if stiction is enabled
- the stick branch can oppose an external tangential load up to `mu_s * N`

This is the main reason floor stiction can stabilize contact without requiring
large slip velocities.

### Vanishing normal load

As `N -> 0`, both dynamic branches go to `0`, and the stick cap also goes to
`0`. Tangential support disappears smoothly as contact vanishes.

## What Matters for the Ledge Runs

For the recent ledge tasks, the robot mostly interacts with fixed terrain cells,
not the global ground plane:

- the main runway is fixed terrain
- the warning ledge is fixed terrain
- the death plane sits just below the lowest occupied terrain row

That means episodes usually terminate before any meaningful recovery on the
global ground plane. The relevant contact law for the current runs is therefore
the static-terrain path plus floor stiction, not the legacy ground-plane law.

## Reward-Coupling Review

The bilateral task score is:

`score = 10 * min(incorrect_progress, cap)`
`      + 20 * reversal_progress * gate(incorrect_progress)`
`      + 30 * toward_goal_frontier`
`      + 40 * goal_touch`
`      - 20 * cliff_fall`
`      - lower_ledge_penalty`

Interpretation:

- falls are already penalized materially
- lower-ledge exposure adds another soft penalty up to `10`
- if fall rates are suddenly high, the first suspect should be contact stability,
  not weak punishment

This does not prove reward shaping is never a factor. It means the first tuning
lever should remain floor friction and stiction.

## Tuning Order

Where contact behavior needs adjusting, this is the order that follows from the
analysis above:

1. `floor_static_friction_const`
2. `floor_stiction_stiffness`
3. `floor_stiction_switch_speed`
4. fall penalty only if the contact model looks internally coherent and stable

## Current Conclusion

- The recent ledge runs are dominated by the static-terrain friction path.
- The most important logic question is not "is the fall penalty too small?"
  but "are the floor stiction parameters giving stable support on softened
  springs?"
- The legacy ground-plane law is worth documenting and testing, but it is not
  the first-order explanation for high ledge fall rates.
