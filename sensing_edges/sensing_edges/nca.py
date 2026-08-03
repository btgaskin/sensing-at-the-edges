"""NCA-CTRNN controller for distributed per-axis actuation.

Simplified from gradient_ascent/nca.py (commit 93baec2, 2026-03-27).
Stripped: MLP mode, single_channel MaCE.
Kept: CTRNN mode, signed deformation (4ch), mechanistic post-run deformation variants,
      signaling/perception ablations, GasNet (off/single/dual), MaCE cross_channel,
      OU noise (configurable), configurable communication filter bank, hybrid update.

Signed deformation channel layout:
  0:   abs_h_delta  (absolute horizontal deformation - 1.0)
  1:   abs_v_delta  (absolute vertical deformation - 1.0)
  2:   comp_h_delta (compliance horizontal deformation - 1.0)
  3:   comp_v_delta (compliance vertical deformation - 1.0)
  4:   h_contraction (output, read as horizontal action)
  5:   v_contraction (output, read as vertical action)
  6:   OU noise (persistent, optional — if ou_enabled=False, channel is absent)
  7+:  grid_hidden channels (visible in grid, written by CTRNN output)
  ...: optional gas channels (mode dependent)
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp

from .deformation_probe import signed_deformation_deltas


# --- Constants ---

N_DEFORM_CHANNELS = 4  # signed mode: 2 absolute + 2 compliance
ACTION_H_IDX = 4
ACTION_V_IDX = 5

GASNET_MODES = ("off", "single", "dual")
GASNET_TYPES = ("signal", "gain")
UPDATE_MODES = ("tanh", "clip", "hybrid")
MACE_MODES = ("off", "cross_channel")
COMM_FILTERS = ("sobel_x", "sobel_y", "laplacian", "identity")

# Sobel and Laplacian kernels for perception.
SOBEL_X = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
SOBEL_Y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)
LAPLACIAN = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=jnp.float32)

_COMM_FILTER_SET = frozenset(COMM_FILTERS)
_SIGNALING_VARIANT_REQUIRED_FILTERS = {
    "intact": (),
    "sobel_x_off": ("sobel_x",),
    "sobel_y_off": ("sobel_y",),
    "laplacian_off": ("laplacian",),
    "identity_comm_off": ("identity",),
    "all_comm_off": (),
    "sobel_x_sign_flip": ("sobel_x",),
}


def normalize_active_comm_filters(
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Normalize active communication filters into canonical order."""
    if active_comm_filters is None:
        return COMM_FILTERS

    requested = tuple(str(name).strip().lower() for name in active_comm_filters)
    if not requested:
        raise ValueError("active_comm_filters must be non-empty")

    invalid = tuple(name for name in requested if name not in _COMM_FILTER_SET)
    if invalid:
        raise ValueError(
            f"Unsupported active_comm_filters={invalid!r}; expected a subset of {COMM_FILTERS}."
        )

    if len(set(requested)) != len(requested):
        raise ValueError("active_comm_filters must not contain duplicates")

    return tuple(name for name in COMM_FILTERS if name in requested)


def signaling_variant_required_filters(signaling_variant: str) -> tuple[str, ...]:
    """Return filters required for a signaling variant to be meaningful."""
    variant = str(signaling_variant).strip().lower()
    required = _SIGNALING_VARIANT_REQUIRED_FILTERS.get(variant)
    if required is None:
        raise ValueError(f"Unsupported signaling_variant={signaling_variant!r}")
    return required


def signaling_variant_is_compatible(
    signaling_variant: str,
    active_comm_filters: Iterable[str] | None = None,
) -> bool:
    """Return whether a signaling variant can run with the active filter bank."""
    active = frozenset(normalize_active_comm_filters(active_comm_filters))
    required = signaling_variant_required_filters(signaling_variant)
    return all(name in active for name in required)


def validate_signaling_variant_compatibility(
    signaling_variant: str,
    active_comm_filters: Iterable[str] | None = None,
    *,
    field_name: str = "signaling_variant",
) -> str:
    """Normalize and validate signaling variant against active communication filters."""
    variant = str(signaling_variant).strip().lower()
    active = normalize_active_comm_filters(active_comm_filters)
    required = signaling_variant_required_filters(variant)
    missing = tuple(name for name in required if name not in active)
    if missing:
        raise ValueError(
            f"{field_name}={variant!r} requires active communication filter(s) {missing}, "
            f"but active_comm_filters={active}."
        )
    return variant


def compatible_signaling_variants(
    signaling_variants: Iterable[str],
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return signaling variants that are compatible with the active filter bank."""
    active = normalize_active_comm_filters(active_comm_filters)
    compatible: list[str] = []
    for variant in signaling_variants:
        normalized = str(variant).strip().lower()
        if signaling_variant_is_compatible(normalized, active):
            compatible.append(normalized)
    return tuple(compatible)


# --- Dimension helpers ---


def gas_channel_count(gasnet_mode: str) -> int:
    if gasnet_mode == "off":
        return 0
    if gasnet_mode == "single":
        return 2  # emission + concentration
    if gasnet_mode == "dual":
        return 4  # e_emit, i_emit, e_conc, i_conc
    raise ValueError(f"Invalid gasnet_mode={gasnet_mode!r}. Expected one of {GASNET_MODES}.")


def gas_emission_count(gasnet_mode: str) -> int:
    if gasnet_mode == "off":
        return 0
    if gasnet_mode == "single":
        return 1
    if gasnet_mode == "dual":
        return 2
    raise ValueError(f"Invalid gasnet_mode={gasnet_mode!r}. Expected one of {GASNET_MODES}.")


def gas_concentration_count(gasnet_mode: str) -> int:
    return gas_channel_count(gasnet_mode) - gas_emission_count(gasnet_mode)


def base_channel_count(*, ou_enabled: bool = True) -> int:
    """Deformation (4) + action (2) + optional noise (1)."""
    return N_DEFORM_CHANNELS + 2 + (1 if ou_enabled else 0)


def growth_identity_channel_count() -> int:
    """Alive-mode growth uses a single frozen identity channel."""
    return 1


def growth_enabled_extra_channels(growth_enabled: bool) -> int:
    return growth_identity_channel_count() + 1 if bool(growth_enabled) else 0


def noise_channel_index(*, ou_enabled: bool = True) -> int | None:
    """Index of the OU noise channel, or None if disabled."""
    if not ou_enabled:
        return None
    return N_DEFORM_CHANNELS + 2  # = 6


def grid_channel_count(
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    growth_enabled: bool = False,
) -> int:
    """Total channels in the visible NCA grid."""
    return (
        base_channel_count(ou_enabled=ou_enabled)
        + int(ctrnn_grid_hidden)
        + gas_channel_count(gasnet_mode)
        + growth_enabled_extra_channels(growth_enabled)
    )


class PerceptionLayout(NamedTuple):
    private_channel_indices: tuple[int, ...]
    comm_channel_indices: tuple[int, ...]
    n_private: int
    n_comm: int


def perception_layout(
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    growth_enabled: bool = False,
) -> PerceptionLayout:
    """Split state channels into identity-only private and spatial communication sets."""
    base_count = base_channel_count(ou_enabled=ou_enabled)
    private_indices = list(range(base_count))
    comm_indices = list(range(base_count, base_count + int(ctrnn_grid_hidden)))
    gas_start = base_count + int(ctrnn_grid_hidden)

    if gasnet_mode == "off":
        pass
    elif gasnet_mode == "single":
        private_indices.append(gas_start)
        comm_indices.append(gas_start + 1)
    elif gasnet_mode == "dual":
        private_indices.extend((gas_start, gas_start + 1))
        comm_indices.extend((gas_start + 2, gas_start + 3))
    else:
        raise ValueError(f"Invalid gasnet_mode={gasnet_mode!r}. Expected one of {GASNET_MODES}.")

    if bool(growth_enabled):
        identity_idx, growth_signal_idx = _growth_channel_indices(
            ctrnn_grid_hidden,
            gasnet_mode,
            ou_enabled=ou_enabled,
        )
        private_indices.append(growth_signal_idx)
        comm_indices.append(identity_idx)

    return PerceptionLayout(
        private_channel_indices=tuple(private_indices),
        comm_channel_indices=tuple(comm_indices),
        n_private=len(private_indices),
        n_comm=len(comm_indices),
    )


def perception_dim(
    n_private: int,
    n_comm: int,
    active_comm_filters: Iterable[str] | None = None,
) -> int:
    """Identity for private channels and a configurable bank for communication channels."""
    return int(n_private) + int(n_comm) * len(normalize_active_comm_filters(active_comm_filters))


def ctrnn_output_dim(
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    *,
    growth_enabled: bool = False,
) -> int:
    """CTRNN output: runtime writes actions/grid_hidden/gas, growth adds identity."""
    return (
        2
        + int(ctrnn_grid_hidden)
        + gas_emission_count(gasnet_mode)
        + (growth_identity_channel_count() if bool(growth_enabled) else 0)
    )


def controller_flat_dim(
    ctrnn_hidden: int,
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> int:
    """Total learnable parameters in the flat genotype."""
    layout = perception_layout(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=growth_enabled,
    )
    pd = perception_dim(
        layout.n_private,
        layout.n_comm,
        active_comm_filters=active_comm_filters,
    )
    od = ctrnn_output_dim(
        ctrnn_grid_hidden,
        gasnet_mode,
        growth_enabled=growth_enabled,
    )
    h = int(ctrnn_hidden)
    # w_input + w_rec + w_output + bias + log_tau
    return pd * h + h * h + h * od + h + h


# --- Parameters ---


class CTRNNParams(NamedTuple):
    """CTRNN controller parameters."""
    w_input: jnp.ndarray   # (perception_dim, ctrnn_hidden)
    w_rec: jnp.ndarray     # (ctrnn_hidden, ctrnn_hidden)
    w_output: jnp.ndarray  # (ctrnn_hidden, output_dim)
    bias: jnp.ndarray      # (ctrnn_hidden,)
    log_tau: jnp.ndarray   # (ctrnn_hidden,)


class NCAStepDiagnostics(NamedTuple):
    """Lightweight alignment diagnostics from one NCA step."""

    controller_hidden_output_max_abs: jnp.ndarray
    action_output_max_abs: jnp.ndarray


def init_ctrnn_params(
    key: jax.Array,
    ctrnn_hidden: int,
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> CTRNNParams:
    """Initialize CTRNN parameters with small random weights."""
    layout = perception_layout(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=growth_enabled,
    )
    pd = perception_dim(
        layout.n_private,
        layout.n_comm,
        active_comm_filters=active_comm_filters,
    )
    od = ctrnn_output_dim(
        ctrnn_grid_hidden,
        gasnet_mode,
        growth_enabled=growth_enabled,
    )
    h = int(ctrnn_hidden)
    k_input, k_rec = jax.random.split(key)
    return CTRNNParams(
        w_input=jax.random.normal(k_input, (pd, h)) * 0.1,
        w_rec=jax.random.normal(k_rec, (h, h)) * 0.05,
        w_output=jnp.zeros((h, od), dtype=jnp.float32),
        bias=jnp.zeros((h,), dtype=jnp.float32),
        log_tau=jnp.zeros((h,), dtype=jnp.float32),
    )


def ctrnn_params_to_flat(params: CTRNNParams) -> jnp.ndarray:
    return jnp.concatenate([
        params.w_input.flatten(),
        params.w_rec.flatten(),
        params.w_output.flatten(),
        params.bias.flatten(),
        params.log_tau.flatten(),
    ])


def ctrnn_flat_to_params(
    flat: jnp.ndarray,
    ctrnn_hidden: int,
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> CTRNNParams:
    """Unflatten a genotype vector into CTRNNParams."""
    layout = perception_layout(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=growth_enabled,
    )
    pd = perception_dim(
        layout.n_private,
        layout.n_comm,
        active_comm_filters=active_comm_filters,
    )
    od = ctrnn_output_dim(
        ctrnn_grid_hidden,
        gasnet_mode,
        growth_enabled=growth_enabled,
    )
    h = int(ctrnn_hidden)

    expected = controller_flat_dim(
        ctrnn_hidden,
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=growth_enabled,
        active_comm_filters=active_comm_filters,
    )
    flat = jnp.ravel(flat)
    if flat.shape[0] != expected:
        raise ValueError(f"Flat length mismatch: expected {expected}, got {flat.shape[0]}.")

    idx = 0
    w_input = flat[idx:idx + pd * h].reshape(pd, h); idx += pd * h
    w_rec = flat[idx:idx + h * h].reshape(h, h); idx += h * h
    w_output = flat[idx:idx + h * od].reshape(h, od); idx += h * od
    bias = flat[idx:idx + h]; idx += h
    log_tau = flat[idx:idx + h]
    return CTRNNParams(w_input=w_input, w_rec=w_rec, w_output=w_output, bias=bias, log_tau=log_tau)


# --- Perception ---


def _apply_filter_bank(
    x: jnp.ndarray,
    *,
    signaling_variant: str = "intact",
    active_comm_filters: Iterable[str] | None = None,
) -> jnp.ndarray:
    """Apply the configured communication filter bank to communication channels."""
    active_filters = normalize_active_comm_filters(active_comm_filters)
    variant = validate_signaling_variant_compatibility(
        signaling_variant,
        active_filters,
        field_name="signaling_variant",
    )
    h, w, c = x.shape
    if c == 0:
        empty_outputs = [x for _ in active_filters]
        return empty_outputs[0] if len(empty_outputs) == 1 else jnp.concatenate(empty_outputs, axis=-1)

    base_kernels = jnp.stack([SOBEL_X, SOBEL_Y, LAPLACIAN], axis=-1)
    rhs = jnp.tile(base_kernels[:, :, None, :], (1, 1, 1, c))
    filtered = jax.lax.conv_general_dilated(
        lhs=x[None, :, :, :],
        rhs=rhs,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=c,
    )[0]
    filtered = filtered.reshape(h, w, c, 3)
    sobel_x = filtered[..., 0]
    sobel_y = filtered[..., 1]
    laplacian = filtered[..., 2]
    identity = x
    if variant == "sobel_x_off":
        sobel_x = jnp.zeros_like(sobel_x)
    elif variant == "sobel_y_off":
        sobel_y = jnp.zeros_like(sobel_y)
    elif variant == "laplacian_off":
        laplacian = jnp.zeros_like(laplacian)
    elif variant == "identity_comm_off":
        identity = jnp.zeros_like(identity)
    elif variant == "all_comm_off":
        sobel_x = jnp.zeros_like(sobel_x)
        sobel_y = jnp.zeros_like(sobel_y)
        laplacian = jnp.zeros_like(laplacian)
        identity = jnp.zeros_like(identity)
    elif variant == "sobel_x_sign_flip":
        sobel_x = -sobel_x
    filter_outputs = {
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "laplacian": laplacian,
        "identity": identity,
    }
    ordered_outputs = [filter_outputs[name] for name in active_filters]
    return ordered_outputs[0] if len(ordered_outputs) == 1 else jnp.concatenate(ordered_outputs, axis=-1)


def perceive(
    state: jnp.ndarray,
    alive_mask: jnp.ndarray,
    *,
    private_channel_indices: tuple[int, ...],
    comm_channel_indices: tuple[int, ...],
    signaling_variant: str = "intact",
    active_comm_filters: Iterable[str] | None = None,
) -> jnp.ndarray:
    """Split perception: identity for private channels, filter bank for communication."""
    masked = state * alive_mask[:, :, None]
    private_idx = jnp.asarray(private_channel_indices, dtype=jnp.int32)
    comm_idx = jnp.asarray(comm_channel_indices, dtype=jnp.int32)
    private = jnp.take(masked, private_idx, axis=-1)
    comm = jnp.take(masked, comm_idx, axis=-1)
    return jnp.concatenate(
        [
            private,
            _apply_filter_bank(
                comm,
                signaling_variant=signaling_variant,
                active_comm_filters=active_comm_filters,
            ),
        ],
        axis=-1,
    )


# --- CTRNN Update ---


def ctrnn_update(
    perception_input: jnp.ndarray,
    params: CTRNNParams,
    hidden_state: jnp.ndarray,
    alive_mask: jnp.ndarray,
    update_key: jax.Array,
    p_update: float = 0.5,
    tau_min: float = 0.1,
    tau_max: float = 10.0,
    dt: float = 1.0,
    gain: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rate-coded CTRNN with exact exponential integration (Dayan & Abbott).

    tau * dy/dt = -y + W @ r + I + bias
    r = tanh(y)
    alpha = exp(-dt / tau)
    y_new = target + (y - target) * alpha
    """
    tau = tau_min + (tau_max - tau_min) * jax.nn.sigmoid(params.log_tau)
    r = jnp.tanh(hidden_state)
    i_ext = perception_input @ params.w_input
    target = r @ params.w_rec + i_ext + params.bias
    if gain is not None:
        target = gain * target
    alpha = jnp.exp(-dt / tau)[jnp.newaxis, jnp.newaxis, :]
    hidden_proposed = target + (hidden_state - target) * alpha
    update_mask = jax.random.bernoulli(update_key, p=p_update, shape=alive_mask.shape)
    mask = (alive_mask & update_mask)[:, :, None]
    hidden_new = jnp.where(mask, hidden_proposed, hidden_state)
    output = jnp.tanh(hidden_new) @ params.w_output
    return hidden_new, output


# --- Gas Dynamics ---


def _gaussian_blur2d(x: jnp.ndarray) -> jnp.ndarray:
    kernel = jnp.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=jnp.float32
    ) / 16.0
    rhs = kernel[:, :, None, None]
    y = jax.lax.conv_general_dilated(
        x[None, :, :, None], rhs,
        window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0, :, :, 0]
    return y


def _diffuse_field_steps(x: jnp.ndarray, n_steps: int | jnp.ndarray) -> jnp.ndarray:
    n_steps = jnp.maximum(jnp.asarray(n_steps, dtype=jnp.int32), jnp.int32(0))
    def _step(_, carry):
        return _gaussian_blur2d(carry)
    return jax.lax.fori_loop(jnp.int32(0), n_steps, _step, x)


def _diffuse_field_with_radius(x: jnp.ndarray, radius: float | jnp.ndarray) -> jnp.ndarray:
    radius = jnp.maximum(jnp.asarray(radius, dtype=jnp.float32), 0.0)
    base = jnp.floor(radius)
    n0 = base.astype(jnp.int32)
    frac = radius - base
    y0 = _diffuse_field_steps(x, n0)
    y1 = _diffuse_field_steps(x, n0 + jnp.int32(1))
    return (1.0 - frac) * y0 + frac * y1


def _gas_channel_indices(
    ctrnn_grid_hidden: int,
    gasnet_mode: str,
    ou_enabled: bool = True,
) -> tuple[int, ...]:
    """Return gas channel indices in the grid state."""
    start = base_channel_count(ou_enabled=ou_enabled) + int(ctrnn_grid_hidden)
    if gasnet_mode == "off":
        return ()
    if gasnet_mode == "single":
        return (start, start + 1)
    if gasnet_mode == "dual":
        return (start, start + 1, start + 2, start + 3)
    raise ValueError(f"Invalid gasnet_mode={gasnet_mode!r}.")


def _growth_channel_indices(
    ctrnn_grid_hidden: int,
    gasnet_mode: str,
    *,
    ou_enabled: bool = True,
) -> tuple[int, int]:
    start = (
        base_channel_count(ou_enabled=ou_enabled)
        + int(ctrnn_grid_hidden)
        + gas_channel_count(gasnet_mode)
    )
    identity_idx = start
    growth_signal_idx = start + growth_identity_channel_count()
    return identity_idx, growth_signal_idx


# --- MaCE ---


def _mace_neighbor_sum_3x3(x_hwc: jnp.ndarray) -> jnp.ndarray:
    """3x3 SAME-padded neighbor sum."""
    h, w, _ = x_hwc.shape
    x_pad = jnp.pad(x_hwc, ((1, 1), (1, 1), (0, 0)))
    return (
        x_pad[0:h, 0:w, :] + x_pad[0:h, 1:w+1, :] + x_pad[0:h, 2:w+2, :]
        + x_pad[1:h+1, 0:w, :] + x_pad[1:h+1, 1:w+1, :] + x_pad[1:h+1, 2:w+2, :]
        + x_pad[2:h+2, 0:w, :] + x_pad[2:h+2, 1:w+1, :] + x_pad[2:h+2, 2:w+2, :]
    )


def _mace_cross_channel_step(
    mass_hw_n: jnp.ndarray,
    affinity_hw_n: jnp.ndarray,
    alive_mask: jnp.ndarray,
    beta: float,
) -> jnp.ndarray:
    """Mass-conserving cross-channel update: all channels share one mass pool."""
    alive_f = alive_mask.astype(jnp.float32)
    alive_hw_n = jnp.broadcast_to(alive_f[:, :, None], mass_hw_n.shape)
    logits = jnp.clip(jnp.asarray(beta, dtype=jnp.float32) * affinity_hw_n, -20.0, 20.0)
    exp_a = jnp.exp(logits) * alive_hw_n
    exp_sum = jnp.sum(exp_a, axis=-1, keepdims=True)
    z_shared = _mace_neighbor_sum_3x3(exp_sum)
    z_shared = jnp.maximum(z_shared, jnp.float32(1e-10))
    inv_z = jnp.reciprocal(z_shared)
    mass_total = jnp.sum(mass_hw_n * alive_hw_n, axis=-1, keepdims=True)
    mass_over_z = mass_total * inv_z
    neighbor_sum = _mace_neighbor_sum_3x3(mass_over_z)
    return exp_a * neighbor_sum * alive_hw_n


# --- Deformation Injection ---


def deformation_to_grid(
    deformation: jnp.ndarray,
    deform_mask: jnp.ndarray,
    robot_offset_y: int,
    robot_offset_x: int,
    canvas_h: int,
    canvas_w: int,
    h: int,
    w: int,
) -> jnp.ndarray:
    """Map padded deformation ratios (H*W, 4) to NCA canvas grid (canvas_h, canvas_w, 4)."""
    deform_2d = deformation.reshape(h, w, 4)  # dual: [abs_h, abs_v, comp_h, comp_v]
    deform_robot = jax.lax.dynamic_slice(
        deform_2d, (robot_offset_y, robot_offset_x, 0), (canvas_h, canvas_w, 4)
    )
    mask_2d = deform_mask.reshape(h, w)
    mask_robot = jax.lax.dynamic_slice(mask_2d, (robot_offset_y, robot_offset_x), (canvas_h, canvas_w))
    return jnp.where(mask_robot[:, :, None], deform_robot, 1.0)


def inject_deformation_channels(
    nca_state: jnp.ndarray,
    deformation: jnp.ndarray,
    deform_mask: jnp.ndarray,
    alive_mask: jnp.ndarray,
    *,
    robot_offset_y: int,
    robot_offset_x: int,
    h: int,
    w: int,
    ablate_deformation: bool = False,
    deformation_gain_absolute: float = 0.0,
    deformation_gain_compliance: float = 0.0,
) -> jnp.ndarray:
    """Write signed deformation inputs into the controller state channels."""
    canvas_h, canvas_w, _ = nca_state.shape
    deform_grid = deformation_to_grid(
        deformation,
        deform_mask,
        robot_offset_y,
        robot_offset_x,
        canvas_h,
        canvas_w,
        h,
        w,
    )

    def _scale(raw: jnp.ndarray, gain: float) -> jnp.ndarray:
        if gain > 0.0:
            return jnp.sign(raw) * jnp.log1p(gain * jnp.abs(raw))
        return raw

    alpha_a = float(deformation_gain_absolute)
    alpha_c = float(deformation_gain_compliance)
    signed_delta = signed_deformation_deltas(deform_grid)
    nca_state = nca_state.at[:, :, 0].set(_scale(signed_delta[:, :, 0], alpha_a) * alive_mask)
    nca_state = nca_state.at[:, :, 1].set(_scale(signed_delta[:, :, 1], alpha_a) * alive_mask)
    nca_state = nca_state.at[:, :, 2].set(_scale(signed_delta[:, :, 2], alpha_c) * alive_mask)
    nca_state = nca_state.at[:, :, 3].set(_scale(signed_delta[:, :, 3], alpha_c) * alive_mask)
    if ablate_deformation:
        nca_state = nca_state.at[:, :, :N_DEFORM_CHANNELS].set(0.0)
    return nca_state


def controller_perception_input(
    nca_state: jnp.ndarray,
    alive_mask: jnp.ndarray,
    *,
    ctrnn_grid_hidden: int = 6,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    signaling_variant: str = "intact",
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> jnp.ndarray:
    """Return the CTRNN perception tensor for a prepared controller state."""
    layout = perception_layout(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=growth_enabled,
    )
    return perceive(
        nca_state,
        alive_mask,
        private_channel_indices=layout.private_channel_indices,
        comm_channel_indices=layout.comm_channel_indices,
        signaling_variant=signaling_variant,
        active_comm_filters=active_comm_filters,
    )


# --- State Update ---


def _apply_state_update(
    nca_state: jnp.ndarray,
    action_values: jnp.ndarray,
    ctrnn_output: jnp.ndarray,
    ctrnn_grid_hidden: int,
    alive_mask: jnp.ndarray,
    update_mode: str,
    state_clip_value: float,
    ou_enabled: bool = True,
) -> jnp.ndarray:
    """Write CTRNN outputs into the NCA grid state."""
    alive_f = alive_mask.astype(jnp.float32)
    new_state = nca_state

    # Write action channels.
    if update_mode == "hybrid":
        new_state = new_state.at[:, :, ACTION_H_IDX].set(jnp.tanh(action_values[:, :, 0]))
        new_state = new_state.at[:, :, ACTION_V_IDX].set(jnp.tanh(action_values[:, :, 1]))
    elif update_mode == "clip":
        new_state = new_state.at[:, :, ACTION_H_IDX].set(
            jnp.clip(action_values[:, :, 0], -state_clip_value, state_clip_value))
        new_state = new_state.at[:, :, ACTION_V_IDX].set(
            jnp.clip(action_values[:, :, 1], -state_clip_value, state_clip_value))
    elif update_mode == "tanh":
        new_state = new_state.at[:, :, ACTION_H_IDX].set(jnp.tanh(action_values[:, :, 0]))
        new_state = new_state.at[:, :, ACTION_V_IDX].set(jnp.tanh(action_values[:, :, 1]))

    # Write grid_hidden channels.
    gh_start = base_channel_count(ou_enabled=ou_enabled)
    if ctrnn_grid_hidden > 0:
        new_state = new_state.at[:, :, gh_start:gh_start + ctrnn_grid_hidden].set(
            ctrnn_output[:, :, 2:2 + ctrnn_grid_hidden] * alive_f[:, :, None]
        )

    return new_state


def _apply_growth_state_update(
    nca_state: jnp.ndarray,
    ctrnn_output: jnp.ndarray,
    ctrnn_grid_hidden: int,
    alive_mask: jnp.ndarray,
    update_mode: str,
    state_clip_value: float,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
) -> jnp.ndarray:
    """Write growth-phase outputs into the NCA grid state."""
    action_values = ctrnn_output[:, :, 0:2]
    new_state = _apply_state_update(
        nca_state,
        action_values,
        ctrnn_output,
        ctrnn_grid_hidden,
        alive_mask,
        update_mode,
        state_clip_value,
        ou_enabled,
    )
    alive_f = alive_mask.astype(jnp.float32)
    output_idx = 2 + int(ctrnn_grid_hidden)
    if gasnet_mode == "single":
        emit_idx, _ = _gas_channel_indices(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        new_state = new_state.at[:, :, emit_idx].set(ctrnn_output[:, :, output_idx] * alive_f)
        output_idx += 1
    elif gasnet_mode == "dual":
        e_emit_idx, i_emit_idx, _, _ = _gas_channel_indices(
            ctrnn_grid_hidden,
            gasnet_mode,
            ou_enabled,
        )
        new_state = new_state.at[:, :, e_emit_idx].set(ctrnn_output[:, :, output_idx] * alive_f)
        new_state = new_state.at[:, :, i_emit_idx].set(ctrnn_output[:, :, output_idx + 1] * alive_f)
        output_idx += 2

    identity_idx, _ = _growth_channel_indices(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled=ou_enabled,
    )
    identity_update = ctrnn_output[:, :, output_idx]
    if update_mode == "clip":
        identity_update = jnp.clip(identity_update, -state_clip_value, state_clip_value)
    else:
        identity_update = jnp.tanh(identity_update)
    new_state = new_state.at[:, :, identity_idx].set(identity_update * alive_f)
    return new_state


def _dilate_mask(mask: jnp.ndarray) -> jnp.ndarray:
    up = jnp.zeros_like(mask).at[:-1, :].set(mask[1:, :])
    down = jnp.zeros_like(mask).at[1:, :].set(mask[:-1, :])
    left = jnp.zeros_like(mask).at[:, :-1].set(mask[:, 1:])
    right = jnp.zeros_like(mask).at[:, 1:].set(mask[:, :-1])
    return mask | up | down | left | right


def _connected_to_seed(
    alive_mask: jnp.ndarray,
    seed_y: int,
    seed_x: int,
    max_iters: int = 10,
) -> jnp.ndarray:
    connected = jnp.zeros_like(alive_mask).at[seed_y, seed_x].set(True)

    def _step(_, conn):
        return _dilate_mask(conn) & alive_mask

    return jax.lax.fori_loop(0, int(max_iters), _step, connected)


def clamp_growth_channels(
    nca_state: jnp.ndarray,
    frozen_identity: jnp.ndarray,
    *,
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
) -> jnp.ndarray:
    identity_idx, growth_signal_idx = _growth_channel_indices(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled=ou_enabled,
    )
    clamped = nca_state.at[:, :, growth_signal_idx].set(0.0)
    return clamped.at[:, :, identity_idx].set(frozen_identity)


def nca_growth_step(
    nca_state: jnp.ndarray,
    ctrnn_carry: jnp.ndarray,
    alive_mask: jnp.ndarray,
    params: CTRNNParams,
    key: jax.Array,
    *,
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    seed_y: int,
    seed_x: int,
    alive_threshold: float = 0.5,
    frontier_identity_only: bool = True,
    p_update: float = 0.5,
    update_mode: str = "hybrid",
    state_clip_value: float = 3.0,
    ctrnn_tau_min: float = 0.1,
    ctrnn_tau_max: float = 10.0,
    ctrnn_dt: float = 1.0,
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single pre-simulation growth step."""
    growth_mask = _dilate_mask(alive_mask)
    layout = perception_layout(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=True,
    )
    perception = perceive(
        nca_state,
        growth_mask,
        private_channel_indices=layout.private_channel_indices,
        comm_channel_indices=layout.comm_channel_indices,
        active_comm_filters=active_comm_filters,
    )
    update_key, _ = jax.random.split(key)
    hidden_new, ctrnn_output = ctrnn_update(
        perception,
        params,
        ctrnn_carry,
        growth_mask,
        update_key,
        p_update=p_update,
        tau_min=ctrnn_tau_min,
        tau_max=ctrnn_tau_max,
        dt=ctrnn_dt,
    )
    if bool(frontier_identity_only):
        frontier_only = growth_mask & (~alive_mask)
        identity_output_idx = ctrnn_output.shape[-1] - growth_identity_channel_count()
        output_mask = jnp.zeros((ctrnn_output.shape[-1],), dtype=ctrnn_output.dtype).at[
            identity_output_idx:
        ].set(1.0)
        ctrnn_output = jnp.where(
            frontier_only[:, :, None],
            ctrnn_output * output_mask[None, None, :],
            ctrnn_output,
        )

    new_state = _apply_growth_state_update(
        nca_state,
        ctrnn_output,
        ctrnn_grid_hidden,
        growth_mask,
        update_mode,
        state_clip_value,
        gasnet_mode=gasnet_mode,
        ou_enabled=ou_enabled,
    )
    identity_idx, growth_signal_idx = _growth_channel_indices(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled=ou_enabled,
    )
    new_state = new_state.at[:, :, growth_signal_idx].set(1.0)
    new_state = new_state.at[seed_y, seed_x, identity_idx].set(1.0)

    new_alive = new_state[:, :, identity_idx] > alive_threshold
    new_alive = new_alive.at[seed_y, seed_x].set(True)
    return new_state, new_alive, hidden_new


def run_growth_phase(
    params: CTRNNParams,
    key: jax.Array,
    *,
    growth_steps: int,
    canvas_h: int,
    canvas_w: int,
    ctrnn_hidden: int,
    ctrnn_grid_hidden: int,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    seed_y: int,
    seed_x: int,
    alive_threshold: float = 0.5,
    frontier_identity_only: bool = True,
    p_update: float = 0.5,
    update_mode: str = "hybrid",
    state_clip_value: float = 3.0,
    connectivity_max_iters: int = 10,
    noise_init_scale: float = 0.0,
    ctrnn_tau_min: float = 0.1,
    ctrnn_tau_max: float = 10.0,
    ctrnn_dt: float = 1.0,
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run pre-simulation growth and return morphology plus terminal controller state."""
    state_channels = grid_channel_count(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=True,
    )
    identity_idx, growth_signal_idx = _growth_channel_indices(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled=ou_enabled,
    )
    nca_state = jnp.zeros((canvas_h, canvas_w, state_channels), dtype=jnp.float32)
    nca_state = nca_state.at[:, :, growth_signal_idx].set(1.0)
    nca_state = nca_state.at[seed_y, seed_x, identity_idx].set(1.0)
    alive_mask = jnp.zeros((canvas_h, canvas_w), dtype=jnp.bool_).at[seed_y, seed_x].set(True)
    ctrnn_carry = jnp.zeros((canvas_h, canvas_w, ctrnn_hidden), dtype=jnp.float32)

    if float(noise_init_scale) > 0.0:
        key, key_noise, key_hidden = jax.random.split(key, 3)
        gh_start = base_channel_count(ou_enabled=ou_enabled)
        if int(ctrnn_grid_hidden) > 0:
            hidden_noise = float(noise_init_scale) * jax.random.uniform(
                key_noise,
                (canvas_h, canvas_w, ctrnn_grid_hidden),
                minval=-1.0,
                maxval=1.0,
            )
            nca_state = nca_state.at[:, :, gh_start:gh_start + ctrnn_grid_hidden].set(hidden_noise)
        if int(ctrnn_hidden) > 0:
            ctrnn_carry = float(noise_init_scale) * jax.random.uniform(
                key_hidden,
                ctrnn_carry.shape,
                minval=-1.0,
                maxval=1.0,
            )

    if int(growth_steps) > 0:
        step_keys = jax.random.split(key, int(growth_steps))

        def _scan_fn(carry, step_key):
            state, alive, hidden = carry
            new_state, new_alive, new_hidden = nca_growth_step(
                state,
                hidden,
                alive,
                params,
                step_key,
                ctrnn_grid_hidden=ctrnn_grid_hidden,
                gasnet_mode=gasnet_mode,
                ou_enabled=ou_enabled,
                seed_y=seed_y,
                seed_x=seed_x,
                alive_threshold=alive_threshold,
                frontier_identity_only=frontier_identity_only,
                p_update=p_update,
                update_mode=update_mode,
                state_clip_value=state_clip_value,
                ctrnn_tau_min=ctrnn_tau_min,
                ctrnn_tau_max=ctrnn_tau_max,
                ctrnn_dt=ctrnn_dt,
                active_comm_filters=active_comm_filters,
            )
            return (new_state, new_alive, new_hidden), None

        (final_state, raw_alive, final_ctrnn_carry), _ = jax.lax.scan(
            _scan_fn,
            (nca_state, alive_mask, ctrnn_carry),
            step_keys,
        )
    else:
        final_state = nca_state
        raw_alive = alive_mask
        final_ctrnn_carry = ctrnn_carry

    raw_alive = raw_alive.at[seed_y, seed_x].set(True)
    alive_connected = _connected_to_seed(
        raw_alive,
        seed_y,
        seed_x,
        max_iters=int(connectivity_max_iters),
    )
    frozen_identity = alive_connected.astype(jnp.float32)
    final_state = final_state * alive_connected[:, :, None].astype(final_state.dtype)
    final_state = final_state.at[:, :, identity_idx].set(frozen_identity)
    final_state = final_state.at[:, :, growth_signal_idx].set(alive_connected.astype(jnp.float32))
    final_ctrnn_carry = final_ctrnn_carry * alive_connected[:, :, None].astype(final_ctrnn_carry.dtype)
    return alive_connected, frozen_identity, final_state, final_ctrnn_carry


# --- Action Mapping ---


def grid_to_padded_actions(
    h_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    robot_offset_y: int,
    robot_offset_x: int,
    h: int,
    w: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map action grids to padded (H*W,) arrays."""
    h_full = jnp.ones((h, w), dtype=jnp.float32)
    v_full = jnp.ones((h, w), dtype=jnp.float32)
    h_full = jax.lax.dynamic_update_slice(h_full, h_grid, (robot_offset_y, robot_offset_x))
    v_full = jax.lax.dynamic_update_slice(v_full, v_grid, (robot_offset_y, robot_offset_x))
    h_padded = jnp.where(actuator_mask, h_full.flatten(), 0.0)
    v_padded = jnp.where(actuator_mask, v_full.flatten(), 0.0)
    return h_padded, v_padded


# --- Full NCA Step ---


def _nca_step_impl(
    nca_state: jnp.ndarray,
    ctrnn_carry: jnp.ndarray,
    deformation: jnp.ndarray,
    deform_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    alive_mask: jnp.ndarray,
    params: CTRNNParams,
    key: jax.Array,
    robot_offset_y: int,
    robot_offset_x: int,
    h: int,
    w: int,
    *,
    ctrnn_grid_hidden: int = 6,
    p_update: float = 0.5,
    ou_theta: float = 0.1,
    ou_sigma: float = 0.3,
    ou_enabled: bool = True,
    ctrnn_tau_min: float = 0.1,
    ctrnn_tau_max: float = 10.0,
    ctrnn_dt: float = 1.0,
    gasnet_mode: str = "off",
    gasnet_type: str = "gain",
    gas_decay_rate: float = 0.95,
    gas_diffusion_radius: float = 3.0,
    gas_gain_sensitivity: float = 2.0,
    gas_max_concentration: float = 1.0,
    gas_gain_min: float = -4.0,
    gas_gain_max: float = 4.0,
    ablate_deformation: bool = False,
    update_mode: str = "hybrid",
    state_clip_value: float = 3.0,
    mace_mode: str = "cross_channel",
    mace_beta: float = 1.0,
    deformation_gain_absolute: float = 0.0,
    deformation_gain_compliance: float = 0.0,
    signaling_variant: str = "intact",
    frozen_perception_input: jnp.ndarray | None = None,
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, NCAStepDiagnostics]:
    """One NCA step: inject deformation → noise → perceive → CTRNN → actions.

    Returns:
        (new_nca_state, new_ctrnn_carry, h_action_padded, v_action_padded, diagnostics)
    """
    key_noise, key_update = jax.random.split(key)

    # --- Deformation injection (signed mode: 4 channels) ---
    nca_state = inject_deformation_channels(
        nca_state,
        deformation,
        deform_mask,
        alive_mask,
        robot_offset_y=robot_offset_y,
        robot_offset_x=robot_offset_x,
        h=h,
        w=w,
        ablate_deformation=ablate_deformation,
        deformation_gain_absolute=deformation_gain_absolute,
        deformation_gain_compliance=deformation_gain_compliance,
    )

    # --- OU noise ---
    if ou_enabled:
        noise_idx = noise_channel_index(ou_enabled=True)
        noise_prev = nca_state[:, :, noise_idx]
        noise_new = noise_prev * (1.0 - ou_theta) + ou_sigma * jax.random.normal(
            key_noise, shape=noise_prev.shape
        )
        nca_state = nca_state.at[:, :, noise_idx].set(noise_new * alive_mask)

    # --- Perception ---
    perc = controller_perception_input(
        nca_state,
        alive_mask,
        ctrnn_grid_hidden=ctrnn_grid_hidden,
        gasnet_mode=gasnet_mode,
        ou_enabled=ou_enabled,
        signaling_variant=signaling_variant,
        growth_enabled=growth_enabled,
        active_comm_filters=active_comm_filters,
    )
    if frozen_perception_input is not None:
        perc = jnp.asarray(frozen_perception_input, dtype=perc.dtype)

    # --- GasNet gain modulation ---
    effective_gain_sensitivity = 0.0 if gasnet_type == "signal" else gas_gain_sensitivity
    ctrnn_gain = None
    if gasnet_mode == "single":
        _, conc_idx = _gas_channel_indices(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        conc = nca_state[:, :, conc_idx]
        ctrnn_gain = jnp.clip(
            1.0 + effective_gain_sensitivity * conc, gas_gain_min, gas_gain_max
        )[:, :, jnp.newaxis]
    elif gasnet_mode == "dual":
        _, _, e_conc_idx, i_conc_idx = _gas_channel_indices(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        e_conc = nca_state[:, :, e_conc_idx]
        i_conc = nca_state[:, :, i_conc_idx]
        ctrnn_gain = jnp.clip(
            1.0 + effective_gain_sensitivity * (e_conc - i_conc), gas_gain_min, gas_gain_max
        )[:, :, jnp.newaxis]

    # --- CTRNN update ---
    hidden_new, ctrnn_output = ctrnn_update(
        perc, params, ctrnn_carry, alive_mask, key_update,
        p_update=p_update, tau_min=ctrnn_tau_min, tau_max=ctrnn_tau_max,
        dt=ctrnn_dt, gain=ctrnn_gain,
    )

    # --- Write outputs to grid state ---
    alive_f = alive_mask.astype(jnp.float32)
    action_values = ctrnn_output[:, :, 0:2] * alive_f[:, :, None]
    if ctrnn_grid_hidden > 0:
        controller_hidden_output_max_abs = jnp.max(
            jnp.abs(ctrnn_output[:, :, 2:2 + ctrnn_grid_hidden] * alive_f[:, :, None])
        )
    else:
        controller_hidden_output_max_abs = jnp.float32(0.0)

    new_state = _apply_state_update(
        nca_state, action_values, ctrnn_output, ctrnn_grid_hidden,
        alive_mask, update_mode, state_clip_value, ou_enabled,
    )
    action_output_max_abs = jnp.max(
        jnp.abs(new_state[:, :, ACTION_H_IDX:ACTION_V_IDX + 1])
    )

    # --- Gas emission channels ---
    if gasnet_mode == "single":
        emit_idx, _ = _gas_channel_indices(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        new_state = new_state.at[:, :, emit_idx].set(
            ctrnn_output[:, :, 2 + ctrnn_grid_hidden] * alive_f
        )
    elif gasnet_mode == "dual":
        e_emit_idx, i_emit_idx, _, _ = _gas_channel_indices(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        new_state = new_state.at[:, :, e_emit_idx].set(
            ctrnn_output[:, :, 2 + ctrnn_grid_hidden] * alive_f
        )
        new_state = new_state.at[:, :, i_emit_idx].set(
            ctrnn_output[:, :, 3 + ctrnn_grid_hidden] * alive_f
        )

    # --- Gas diffusion + decay ---
    hidden_affinity = (
        ctrnn_output[:, :, 2:2 + ctrnn_grid_hidden] * alive_f[:, :, None]
        if ctrnn_grid_hidden > 0 else None
    )

    if gasnet_mode == "single":
        emit_idx, conc_idx = _gas_channel_indices(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        emission = jax.nn.relu(new_state[:, :, emit_idx]) * alive_mask
        conc = nca_state[:, :, conc_idx] * gas_decay_rate + emission
        conc = _diffuse_field_with_radius(conc, gas_diffusion_radius)
        conc = jnp.clip(conc, 0.0, gas_max_concentration)
        new_state = new_state.at[:, :, conc_idx].set(conc * alive_mask)
    elif gasnet_mode == "dual":
        e_emit_idx, i_emit_idx, e_conc_idx, i_conc_idx = _gas_channel_indices(
            ctrnn_grid_hidden, gasnet_mode, ou_enabled
        )
        e_emission = jax.nn.relu(new_state[:, :, e_emit_idx]) * alive_mask
        i_emission = jax.nn.relu(new_state[:, :, i_emit_idx]) * alive_mask
        e_conc = nca_state[:, :, e_conc_idx] * gas_decay_rate + e_emission
        i_conc = nca_state[:, :, i_conc_idx] * gas_decay_rate + i_emission
        e_conc = _diffuse_field_with_radius(e_conc, gas_diffusion_radius)
        i_conc = _diffuse_field_with_radius(i_conc, gas_diffusion_radius)
        e_conc = jnp.clip(e_conc, 0.0, gas_max_concentration)
        i_conc = jnp.clip(i_conc, 0.0, gas_max_concentration)
        new_state = new_state.at[:, :, e_conc_idx].set(e_conc * alive_mask)
        new_state = new_state.at[:, :, i_conc_idx].set(i_conc * alive_mask)

    # --- MaCE ---
    if mace_mode == "cross_channel" and ctrnn_grid_hidden > 0:
        gh_start = base_channel_count(ou_enabled=ou_enabled)
        hidden_mass_src = nca_state[:, :, gh_start:gh_start + ctrnn_grid_hidden]
        hidden_mass_next = _mace_cross_channel_step(
            hidden_mass_src, hidden_affinity, alive_mask, mace_beta
        )
        new_state = new_state.at[:, :, gh_start:gh_start + ctrnn_grid_hidden].set(hidden_mass_next)

    # --- Read actions ---
    h_contraction = new_state[:, :, ACTION_H_IDX]
    v_contraction = new_state[:, :, ACTION_V_IDX]
    h_action_grid = jnp.clip(1.0 + 0.5 * h_contraction, 0.5, 1.5)
    v_action_grid = jnp.clip(1.0 + 0.5 * v_contraction, 0.5, 1.5)
    h_action_grid = jnp.where(alive_mask, h_action_grid, 1.0)
    v_action_grid = jnp.where(alive_mask, v_action_grid, 1.0)

    h_action, v_action = grid_to_padded_actions(
        h_action_grid, v_action_grid, actuator_mask, robot_offset_y, robot_offset_x, h, w
    )

    return new_state, hidden_new, h_action, v_action, NCAStepDiagnostics(
        controller_hidden_output_max_abs=controller_hidden_output_max_abs,
        action_output_max_abs=action_output_max_abs,
    )


def nca_step(
    nca_state: jnp.ndarray,
    ctrnn_carry: jnp.ndarray,
    deformation: jnp.ndarray,
    deform_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    alive_mask: jnp.ndarray,
    params: CTRNNParams,
    key: jax.Array,
    robot_offset_y: int,
    robot_offset_x: int,
    h: int,
    w: int,
    *,
    ctrnn_grid_hidden: int = 6,
    p_update: float = 0.5,
    ou_theta: float = 0.1,
    ou_sigma: float = 0.3,
    ou_enabled: bool = True,
    ctrnn_tau_min: float = 0.1,
    ctrnn_tau_max: float = 10.0,
    ctrnn_dt: float = 1.0,
    gasnet_mode: str = "off",
    gasnet_type: str = "gain",
    gas_decay_rate: float = 0.95,
    gas_diffusion_radius: float = 3.0,
    gas_gain_sensitivity: float = 2.0,
    gas_max_concentration: float = 1.0,
    gas_gain_min: float = -4.0,
    gas_gain_max: float = 4.0,
    ablate_deformation: bool = False,
    update_mode: str = "hybrid",
    state_clip_value: float = 3.0,
    mace_mode: str = "cross_channel",
    mace_beta: float = 1.0,
    deformation_gain_absolute: float = 0.0,
    deformation_gain_compliance: float = 0.0,
    signaling_variant: str = "intact",
    frozen_perception_input: jnp.ndarray | None = None,
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One NCA step without diagnostic outputs."""
    new_state, hidden_new, h_action, v_action, _ = _nca_step_impl(
        nca_state,
        ctrnn_carry,
        deformation,
        deform_mask,
        actuator_mask,
        alive_mask,
        params,
        key,
        robot_offset_y,
        robot_offset_x,
        h,
        w,
        ctrnn_grid_hidden=ctrnn_grid_hidden,
        p_update=p_update,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
        ou_enabled=ou_enabled,
        ctrnn_tau_min=ctrnn_tau_min,
        ctrnn_tau_max=ctrnn_tau_max,
        ctrnn_dt=ctrnn_dt,
        gasnet_mode=gasnet_mode,
        gasnet_type=gasnet_type,
        gas_decay_rate=gas_decay_rate,
        gas_diffusion_radius=gas_diffusion_radius,
        gas_gain_sensitivity=gas_gain_sensitivity,
        gas_max_concentration=gas_max_concentration,
        gas_gain_min=gas_gain_min,
        gas_gain_max=gas_gain_max,
        ablate_deformation=ablate_deformation,
        update_mode=update_mode,
        state_clip_value=state_clip_value,
        mace_mode=mace_mode,
        mace_beta=mace_beta,
        deformation_gain_absolute=deformation_gain_absolute,
        deformation_gain_compliance=deformation_gain_compliance,
        signaling_variant=signaling_variant,
        frozen_perception_input=frozen_perception_input,
        growth_enabled=growth_enabled,
        active_comm_filters=active_comm_filters,
    )
    return new_state, hidden_new, h_action, v_action


def nca_step_with_diagnostics(
    nca_state: jnp.ndarray,
    ctrnn_carry: jnp.ndarray,
    deformation: jnp.ndarray,
    deform_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    alive_mask: jnp.ndarray,
    params: CTRNNParams,
    key: jax.Array,
    robot_offset_y: int,
    robot_offset_x: int,
    h: int,
    w: int,
    *,
    ctrnn_grid_hidden: int = 6,
    p_update: float = 0.5,
    ou_theta: float = 0.1,
    ou_sigma: float = 0.3,
    ou_enabled: bool = True,
    ctrnn_tau_min: float = 0.1,
    ctrnn_tau_max: float = 10.0,
    ctrnn_dt: float = 1.0,
    gasnet_mode: str = "off",
    gasnet_type: str = "gain",
    gas_decay_rate: float = 0.95,
    gas_diffusion_radius: float = 3.0,
    gas_gain_sensitivity: float = 2.0,
    gas_max_concentration: float = 1.0,
    gas_gain_min: float = -4.0,
    gas_gain_max: float = 4.0,
    ablate_deformation: bool = False,
    update_mode: str = "hybrid",
    state_clip_value: float = 3.0,
    mace_mode: str = "cross_channel",
    mace_beta: float = 1.0,
    deformation_gain_absolute: float = 0.0,
    deformation_gain_compliance: float = 0.0,
    signaling_variant: str = "intact",
    frozen_perception_input: jnp.ndarray | None = None,
    growth_enabled: bool = False,
    active_comm_filters: Iterable[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, NCAStepDiagnostics]:
    """One NCA step plus lightweight alignment diagnostics."""
    return _nca_step_impl(
        nca_state,
        ctrnn_carry,
        deformation,
        deform_mask,
        actuator_mask,
        alive_mask,
        params,
        key,
        robot_offset_y,
        robot_offset_x,
        h,
        w,
        ctrnn_grid_hidden=ctrnn_grid_hidden,
        p_update=p_update,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
        ou_enabled=ou_enabled,
        ctrnn_tau_min=ctrnn_tau_min,
        ctrnn_tau_max=ctrnn_tau_max,
        ctrnn_dt=ctrnn_dt,
        gasnet_mode=gasnet_mode,
        gasnet_type=gasnet_type,
        gas_decay_rate=gas_decay_rate,
        gas_diffusion_radius=gas_diffusion_radius,
        gas_gain_sensitivity=gas_gain_sensitivity,
        gas_max_concentration=gas_max_concentration,
        gas_gain_min=gas_gain_min,
        gas_gain_max=gas_gain_max,
        ablate_deformation=ablate_deformation,
        update_mode=update_mode,
        state_clip_value=state_clip_value,
        mace_mode=mace_mode,
        mace_beta=mace_beta,
        deformation_gain_absolute=deformation_gain_absolute,
        deformation_gain_compliance=deformation_gain_compliance,
        signaling_variant=signaling_variant,
        frozen_perception_input=frozen_perception_input,
        growth_enabled=growth_enabled,
        active_comm_filters=active_comm_filters,
    )
