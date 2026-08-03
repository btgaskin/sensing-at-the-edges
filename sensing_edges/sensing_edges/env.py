"""Cliff-walk environment with phased sense-act-physics interleaving.

Build pattern uses core jax_evogym library (compile_world_templates, instantiate_world)
via the gradient_ascent env.py compile_core_env_templates + make_core_env_assets helpers.

Episode runner uses nested jax.lax.scan:
  outer: env steps (max_steps)
  inner: sense-act phases (sense_act_phases_per_env_step)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_evogym import constants as C
from jax_evogym.build import (
    compile_world_templates,
    instantiate_world,
    make_static_collider_data,
)
from jax_evogym.cell_geometry import build_sparse_cell_geometry, grid_active_cells
from jax_evogym.collision import StaticCollisionWorklist, build_static_collision_worklist
from jax_evogym.constants import CELL_SIZE, EMPTY
from jax_evogym.deformation import compute_dual_deformation
from jax_evogym.actuators import set_per_axis_goals_compact
from jax_evogym.jax_utils import (
    jax_build_actuator_info,
    jax_build_collision_data,
    jax_build_deformation_info,
    jax_build_per_axis_actuator_info,
    jax_build_sim_state,
    jax_compose_grid,
    precompute_grid,
)
from jax_evogym.sim import (
    PhysicsSubstepProbe,
    physics_substep_dynamic_broadphase,
    physics_substep_dynamic_broadphase_with_probe,
)
from jax_evogym.types import (
    ActuatorInfo,
    CollisionData,
    DeformationInfo,
    GridData,
    PerAxisActuatorInfo,
    PhysicsConstants,
    RenderInfo,
    RenderStepOutput,
    SimState,
    SpringTopology,
    StaticColliderData,
    default_physics_constants,
)
from jax_evogym.world import EvoWorld

from .collision_tuning import CollisionBuildCapacity
from .deformation_probe import (
    update_deformation_channel_max,
    zero_deformation_channel_max,
)
from .nca import (
    ACTION_H_IDX,
    ACTION_V_IDX,
    COMM_FILTERS,
    CTRNNParams,
    base_channel_count,
    clamp_growth_channels,
    controller_perception_input,
    grid_channel_count,
    inject_deformation_channels,
    nca_step_with_diagnostics,
    _growth_channel_indices,
)

REFERENCE_SUBSTEPS = 30
REFERENCE_DT = 0.0001
_DEATH_PLANE_MARGIN = 0.1 * CELL_SIZE
_TASK_MODE_CODE_LEGACY = 0
_TASK_MODE_CODE_BILATERAL_GOAL_CLIFF = 1
_SIDE_CODE_LEFT = -1
_SIDE_CODE_NONE = 0
_SIDE_CODE_RIGHT = 1
_TASK_MODE_BILATERAL_GOAL_CLIFF = "bilateral_goal_cliff"
_GOAL_OBJECT_ID = 2
_TERRAIN_OBJECT_ID = 1
_TURN_DIRECTION_DELTA_THRESHOLD_MIN = 0.1 * CELL_SIZE
_TURN_DIRECTION_DELTA_THRESHOLD_SPAN_FRACTION = 0.02
_STABLE_TURN_DISTANCE_THRESHOLD = 1.0 * CELL_SIZE
_MAX_TURN_COUNT = 6


# ---------------------------------------------------------------------------
# Environment Data
# ---------------------------------------------------------------------------


class CliffEnvData(NamedTuple):
    """Static environment data built once per terrain."""
    sim_state: SimState
    topology: SpringTopology
    constants: PhysicsConstants
    actuator_info: ActuatorInfo
    per_axis_info: PerAxisActuatorInfo
    collision_data: CollisionData
    static_collider_data: StaticColliderData
    static_collision_worklist: StaticCollisionWorklist
    deformation_info: DeformationInfo
    actuator_mask: jnp.ndarray          # (H*W,) bool
    deform_mask: jnp.ndarray            # (H*W,) bool
    robot_point_mask: jnp.ndarray       # (n_points,) bool
    robot_point_count: jnp.ndarray      # () int32
    grid_h: int
    grid_w: int
    spawn_x: int
    spawn_y: int
    robot_h: int
    robot_w: int
    death_plane_y: float                # non-colliding y threshold for death
    init_com_y: float                   # initial robot COM y (for airborne)
    init_com_x: float                   # initial robot COM x
    init_min_x: float                   # initial min robot point x
    init_front_x: float                 # initial max robot point x
    cliff_x: int                        # last FIXED column
    task_mode: int                      # 0=legacy cliff, 1=bilateral goal-cliff
    goal_direction: int                 # -1=left, +1=right, 0=none
    cliff_side: int                     # -1=left, +1=right, 0=none
    goal_distance: int                  # symmetric runway distance for bilateral task
    cliff_distance: int                 # cliff-side runway distance for bilateral task
    goal_block_x: int                   # goal-block cell x, -1 if none
    goal_block_y: int                   # goal-block cell y, -1 if none
    goal_x: float                       # world-space goal-block center x, -1 if none
    goal_object_id: int                 # render/debug object id for goal block
    cliff_probe_ledge_width: int        # incorrect-side warning ledge width, 0 if none
    cliff_probe_ledge_cell_x: int       # incorrect-side warning ledge start x, -1 if none
    cliff_probe_ledge_cell_y: int       # incorrect-side warning ledge y, -1 if none
    render_info: object                 # RenderInfo from instantiate_world (not JIT-traced)


class CliffEvalEnvData(NamedTuple):
    """Array-only environment data safe to stack, vmap, and jit."""
    sim_state: SimState
    topology: SpringTopology
    constants: PhysicsConstants
    actuator_info: ActuatorInfo
    per_axis_info: PerAxisActuatorInfo
    collision_data: CollisionData
    static_collider_data: StaticColliderData
    static_collision_worklist: StaticCollisionWorklist
    deformation_info: DeformationInfo
    actuator_mask: jnp.ndarray
    deform_mask: jnp.ndarray
    robot_point_mask: jnp.ndarray
    robot_point_count: jnp.ndarray
    grid_h: int
    grid_w: int
    spawn_x: int
    spawn_y: int
    robot_h: int
    robot_w: int
    death_plane_y: float
    init_com_y: float
    init_com_x: float
    init_min_x: float
    init_front_x: float
    cliff_x: int
    task_mode: int
    goal_direction: int
    cliff_side: int
    goal_distance: int
    cliff_distance: int
    goal_block_x: int
    goal_block_y: int
    goal_x: float
    goal_object_id: int
    cliff_probe_ledge_width: int
    cliff_probe_ledge_cell_x: int
    cliff_probe_ledge_cell_y: int


class CliffRenderAssets(NamedTuple):
    """Render-only assets kept out of the jitted training/eval path."""
    render_info: RenderInfo | None


class CliffStaticTemplate(NamedTuple):
    """Terrain-only static collider template plus rollout metadata."""

    static_collider_data: StaticColliderData
    cliff_x: int
    death_plane_y: float
    task_mode: int
    goal_direction: int
    cliff_side: int
    goal_distance: int
    cliff_distance: int
    goal_block_x: int
    goal_block_y: int
    goal_x: float
    goal_object_id: int
    cliff_probe_ledge_width: int
    cliff_probe_ledge_cell_x: int
    cliff_probe_ledge_cell_y: int
    grid_h: int
    grid_w: int
    render_info: RenderInfo | None


class CliffStaticTemplateBatch(NamedTuple):
    """Batched terrain-only static collider templates."""

    static_collider_data: StaticColliderData
    cliff_x: jnp.ndarray
    death_plane_y: jnp.ndarray
    task_mode: jnp.ndarray
    goal_direction: jnp.ndarray
    cliff_side: jnp.ndarray
    goal_distance: jnp.ndarray
    cliff_distance: jnp.ndarray
    goal_block_x: jnp.ndarray
    goal_block_y: jnp.ndarray
    goal_x: jnp.ndarray
    goal_object_id: jnp.ndarray
    cliff_probe_ledge_width: jnp.ndarray
    cliff_probe_ledge_cell_x: jnp.ndarray
    cliff_probe_ledge_cell_y: jnp.ndarray


class StaticColliderCapacity(NamedTuple):
    """Fixed static-collider capacities used to keep batched env shapes stable."""

    max_static_points: int
    max_static_cells: int
    max_grid_columns: int = 0
    max_cells_per_column: int = 0


class CliffDynamicCore(NamedTuple):
    """Robot-only dynamic simulation state shared across terrains."""

    sim_state: SimState
    topology: SpringTopology
    constants: PhysicsConstants
    actuator_info: ActuatorInfo
    per_axis_info: PerAxisActuatorInfo
    collision_data: CollisionData
    deformation_info: DeformationInfo
    robot_point_mask: jnp.ndarray
    robot_point_count: jnp.ndarray
    robot_morph: jnp.ndarray
    spawn_x: int
    spawn_y: int
    robot_h: int
    robot_w: int
    init_com_y: float
    init_com_x: float
    init_min_x: float
    init_front_x: float
    render_info: RenderInfo | None


class CliffTerrainAssets(NamedTuple):
    """Terrain-specific static collider and render assets."""

    static_collider_template: StaticColliderData
    grid_h: int
    grid_w: int
    cliff_x: int
    death_plane_y: float
    task_mode: int
    goal_direction: int
    cliff_side: int
    goal_distance: int
    cliff_distance: int
    goal_block_x: int
    goal_block_y: int
    goal_x: float
    goal_object_id: int
    cliff_probe_ledge_width: int
    cliff_probe_ledge_cell_x: int
    cliff_probe_ledge_cell_y: int
    render_info: RenderInfo | None


class CliffEpisodeState(NamedTuple):
    """Mutable state carried through the episode scan."""
    sim_state: SimState
    nca_state: jnp.ndarray              # (robot_h, robot_w, grid_channels)
    ctrnn_carry: jnp.ndarray            # (robot_h, robot_w, ctrnn_hidden)
    step_count: jnp.ndarray             # () int32
    done: jnp.ndarray                   # () bool
    peak_front_x: jnp.ndarray           # () float32
    safe_front_x: jnp.ndarray           # () float32 — last front_x before termination
    touched_death_plane: jnp.ndarray    # () bool
    went_airborne: jnp.ndarray          # () bool
    com_x_went_negative: jnp.ndarray    # () bool
    cumulative_activation: jnp.ndarray  # () float32
    cumulative_h_activation: jnp.ndarray  # () float32
    cumulative_v_activation: jnp.ndarray  # () float32
    cumulative_speed: jnp.ndarray       # () float32
    cumulative_lower_ledge_exposure_steps: jnp.ndarray  # () int32 — env steps overlapping the lowered probe ledge below its top
    cumulative_cliff_vicinity_steps: jnp.ndarray  # () int32 — COM within 3 cells of cliff edge
    cumulative_lofted_steps: jnp.ndarray  # () int32 — env steps with COM above the soft loft threshold
    cumulative_soft_loft_excess_cells: jnp.ndarray  # () float32 — summed COM-height excess above the soft loft threshold
    first_cliff_vicinity_step: jnp.ndarray  # () int32 — 1-based first env step entering cliff band, -1 if never
    peak_any_direction_progress: jnp.ndarray  # () float32 — max |com_x - init_com_x|
    peak_goal_progress: jnp.ndarray      # () float32 — max directed leading-edge progress toward goal
    peak_incorrect_progress: jnp.ndarray  # () float32 — max directed trailing-edge progress away from goal
    peak_toward_goal_frontier: jnp.ndarray  # () float32 — unused legacy bilateral target-progress scratch
    goal_touched: jnp.ndarray            # () bool
    cliff_fell: jnp.ndarray              # () bool
    first_goal_touch_step: jnp.ndarray   # () int32 — 1-based first env step touching goal, -1 if never
    last_significant_direction: jnp.ndarray  # () int32 — last significant COM-x direction in {-1,0,+1}
    turn_count: jnp.ndarray              # () int32 — clipped count of significant COM-x direction flips
    stable_turn_direction: jnp.ndarray   # () int32 — current sustained COM-x segment direction in {-1,0,+1}
    stable_turn_distance: jnp.ndarray    # () float32 — distance accumulated in current direction segment
    stable_turn_eligible: jnp.ndarray    # () bool — whether the current segment can still mature into a stable turn
    stable_turn_count: jnp.ndarray       # () int32 — clipped count of sustained COM-x direction flips
    steps_alive: jnp.ndarray            # () int32
    deformation_channel_max: jnp.ndarray  # (4,) float32 — raw signed deformation abs maxima
    controller_hidden_output_max_abs: jnp.ndarray  # () float32
    action_output_max_abs: jnp.ndarray  # () float32
    peak_com_y_delta_cells: jnp.ndarray  # () float32 — max COM lift above the initial COM in cell units
    first_significant_direction: jnp.ndarray = 0


class EpisodeResult(NamedTuple):
    """Outputs from a completed episode."""
    peak_front_x: jnp.ndarray           # () float32
    safe_front_x: jnp.ndarray           # () float32 — last front_x before termination
    init_front_x: jnp.ndarray           # () float32
    init_com_x: jnp.ndarray             # () float32
    touched_death_plane: jnp.ndarray    # () bool
    went_airborne: jnp.ndarray          # () bool
    com_x_went_negative: jnp.ndarray    # () bool
    mean_activation: jnp.ndarray        # () float32
    mean_h_activation: jnp.ndarray      # () float32
    mean_v_activation: jnp.ndarray      # () float32
    mean_speed: jnp.ndarray             # () float32
    cliff_vicinity_steps: jnp.ndarray   # () int32 — COM within 3 cells of cliff edge
    cliff_vicinity_fraction: jnp.ndarray  # () float32 — cliff_vicinity_steps / max_steps
    time_to_edge: jnp.ndarray           # () float32 — first cliff-band env step / max_steps, 1.0 if never reached
    elapsed_time_fraction: jnp.ndarray  # () float32 — steps_alive / max_steps with a 1/max_steps floor
    steps_alive: jnp.ndarray            # () int32
    cliff_x: jnp.ndarray                # () int32
    task_mode: jnp.ndarray              # () int32
    goal_direction: jnp.ndarray         # () int32
    cliff_side: jnp.ndarray             # () int32
    goal_distance: jnp.ndarray          # () int32
    cliff_distance: jnp.ndarray         # () int32
    goal_block_x: jnp.ndarray           # () int32
    goal_block_y: jnp.ndarray           # () int32
    goal_x: jnp.ndarray                 # () float32
    any_progress: jnp.ndarray           # () float32 — normalized any-direction COM progress
    goal_progress: jnp.ndarray          # () float32 — normalized peak goal-facing edge progress
    incorrect_progress: jnp.ndarray     # () float32 — normalized peak wrong-facing edge progress
    toward_goal_frontier: jnp.ndarray   # () float32 — normalized final goal-facing edge progress toward target
    goal_touched: jnp.ndarray           # () bool
    cliff_fell: jnp.ndarray             # () bool
    time_to_goal: jnp.ndarray           # () float32 — first goal-touch step / max_steps, 1.0 if never touched
    turn_count: jnp.ndarray             # () int32 — clipped count of significant COM-x direction flips
    deformation_channel_max: jnp.ndarray  # (4,) float32 — raw signed deformation abs maxima
    controller_hidden_output_max_abs: jnp.ndarray  # () float32
    action_output_max_abs: jnp.ndarray  # () float32
    lower_ledge_exposure_steps: jnp.ndarray = 0
    lower_ledge_exposure_fraction: jnp.ndarray = 0.0
    reversal_progress: jnp.ndarray = 0.0
    stable_turn_count: jnp.ndarray = 0
    turn_motion_threshold: jnp.ndarray = 0.0
    first_significant_direction: jnp.ndarray = 0
    final_com_x: jnp.ndarray = 0.0
    final_front_x: jnp.ndarray = 0.0
    final_back_x: jnp.ndarray = 0.0
    lofted_fraction: jnp.ndarray = 0.0
    mean_lofted_excess_cells: jnp.ndarray = 0.0
    peak_com_y_delta_cells: jnp.ndarray = 0.0


class EpisodePhaseProbeStep(NamedTuple):
    """Per-phase diagnostic output for one environment step."""

    pre_positions: jnp.ndarray
    pre_velocities: jnp.ndarray
    pre_spring_rest_length: jnp.ndarray
    h_compact: jnp.ndarray
    v_compact: jnp.ndarray
    controller_hidden_output_max_abs: jnp.ndarray
    action_output_max_abs: jnp.ndarray
    post_positions: jnp.ndarray
    post_velocities: jnp.ndarray
    post_velocities_true: jnp.ndarray
    post_spring_rest_length: jnp.ndarray
    last_substep_post_integrator_velocities: jnp.ndarray
    last_substep_final_velocities: jnp.ndarray
    last_substep_final_velocities_true: jnp.ndarray
    done_before: jnp.ndarray
    done_after: jnp.ndarray


class EpisodePhaseProbeTrace(NamedTuple):
    """Phase-level rollout diagnostics for repeated same-input episode probes."""

    initial_positions: jnp.ndarray
    initial_velocities: jnp.ndarray
    initial_spring_rest_length: jnp.ndarray
    initial_nca_state: jnp.ndarray
    initial_ctrnn_carry: jnp.ndarray
    phase_pre_positions: jnp.ndarray
    phase_pre_velocities: jnp.ndarray
    phase_pre_spring_rest_length: jnp.ndarray
    phase_h_compact: jnp.ndarray
    phase_v_compact: jnp.ndarray
    phase_controller_hidden_output_max_abs: jnp.ndarray
    phase_action_output_max_abs: jnp.ndarray
    phase_post_positions: jnp.ndarray
    phase_post_velocities: jnp.ndarray
    phase_post_velocities_true: jnp.ndarray
    phase_post_spring_rest_length: jnp.ndarray
    phase_last_substep_post_integrator_velocities: jnp.ndarray
    phase_last_substep_final_velocities: jnp.ndarray
    phase_last_substep_final_velocities_true: jnp.ndarray
    phase_done_before: jnp.ndarray
    phase_done_after: jnp.ndarray


class EpisodeStateTrace(NamedTuple):
    """Env-step robot-local state trace for PCA and offline assay metrics."""

    env_step: jnp.ndarray
    nca_state: jnp.ndarray
    ctrnn_carry: jnp.ndarray
    action_outputs: jnp.ndarray
    com_x: jnp.ndarray
    min_x: jnp.ndarray
    front_x: jnp.ndarray
    goal_touched: jnp.ndarray
    cliff_fell: jnp.ndarray
    done: jnp.ndarray


class EpisodeComTrace(NamedTuple):
    """Env-step COM-x trace for lightweight gait spectral metrics."""

    com_x: jnp.ndarray


class EpisodeControllerTrace(NamedTuple):
    """Controller-step robot-local trace for targeted PCA assays."""

    nca_state: jnp.ndarray
    ctrnn_carry: jnp.ndarray
    action_outputs: jnp.ndarray
    com_x: jnp.ndarray
    back_x: jnp.ndarray
    front_x: jnp.ndarray
    active: jnp.ndarray


class EpisodeStateSpacePhaseStep(NamedTuple):
    """Per sense-act phase diagnostics for dense env-step state traces."""

    active: jnp.ndarray
    deformation_inputs: jnp.ndarray
    h_compact: jnp.ndarray
    v_compact: jnp.ndarray
    controller_hidden_output_max_abs: jnp.ndarray
    action_output_max_abs: jnp.ndarray
    nca_state: jnp.ndarray
    ctrnn_carry: jnp.ndarray
    action_outputs: jnp.ndarray
    post_positions: jnp.ndarray
    post_velocities: jnp.ndarray
    post_velocities_true: jnp.ndarray
    post_spring_rest_length: jnp.ndarray


class EpisodeStateSpaceTrace(NamedTuple):
    """Env-step plant/controller trace for state-space analyses."""

    env_step: jnp.ndarray
    active: jnp.ndarray
    done_before: jnp.ndarray
    done_after: jnp.ndarray
    pre_positions: jnp.ndarray
    pre_velocities: jnp.ndarray
    pre_spring_rest_length: jnp.ndarray
    post_positions: jnp.ndarray
    post_velocities: jnp.ndarray
    post_velocities_true: jnp.ndarray
    post_spring_rest_length: jnp.ndarray
    nca_state: jnp.ndarray
    ctrnn_carry: jnp.ndarray
    action_outputs: jnp.ndarray
    phase: EpisodeStateSpacePhaseStep
    com_x: jnp.ndarray
    com_y: jnp.ndarray
    min_x: jnp.ndarray
    max_x: jnp.ndarray
    min_y: jnp.ndarray
    max_y: jnp.ndarray
    front_x: jnp.ndarray
    back_x: jnp.ndarray
    goal_touched: jnp.ndarray
    cliff_fell: jnp.ndarray


class EpisodeSubstepProbeTrace(NamedTuple):
    """Selected-phase substep diagnostics for repeated same-input probes."""

    step_index: jnp.ndarray
    phase_index: jnp.ndarray
    phase_pre_positions: jnp.ndarray
    phase_pre_velocities: jnp.ndarray
    phase_pre_spring_rest_length: jnp.ndarray
    phase_h_compact: jnp.ndarray
    phase_v_compact: jnp.ndarray
    phase_controller_hidden_output_max_abs: jnp.ndarray
    phase_action_output_max_abs: jnp.ndarray
    substep_dynamic_selected_pair_mask: jnp.ndarray
    substep_dynamic_overflow: jnp.ndarray
    substep_dynamic_selected_triple_active: jnp.ndarray
    substep_dynamic_triple_active: jnp.ndarray
    substep_dynamic_forces: jnp.ndarray
    substep_static_forces: jnp.ndarray
    substep_static_contact_mask: jnp.ndarray
    substep_static_memory_contact_mask: jnp.ndarray
    substep_tangential_deformation: jnp.ndarray
    substep_static_manifold_active: jnp.ndarray
    substep_post_update_actuator_spring_rest_length: jnp.ndarray
    substep_post_integrator_positions: jnp.ndarray
    substep_post_integrator_velocities: jnp.ndarray
    substep_post_constraint_positions: jnp.ndarray
    substep_post_anchor_positions: jnp.ndarray
    substep_final_positions: jnp.ndarray
    substep_velocity_true_positions_last: jnp.ndarray
    substep_velocity_true_displacement: jnp.ndarray
    substep_final_velocities: jnp.ndarray
    substep_final_velocities_true: jnp.ndarray


class EpisodeSubstepProbeCarry(NamedTuple):
    """Episode state plus the captured selected-phase substep trace."""

    episode_state: CliffEpisodeState
    substep_trace: EpisodeSubstepProbeTrace


def _side_code(side: str | None) -> int:
    normalized = str(side).strip().lower() if side is not None else ""
    if normalized == "left":
        return _SIDE_CODE_LEFT
    if normalized == "right":
        return _SIDE_CODE_RIGHT
    return _SIDE_CODE_NONE


def _task_mode_code(task_metadata: dict[str, object] | None) -> int:
    if str((task_metadata or {}).get("task_mode", "")).strip().lower() == _TASK_MODE_BILATERAL_GOAL_CLIFF:
        return _TASK_MODE_CODE_BILATERAL_GOAL_CLIFF
    return _TASK_MODE_CODE_LEGACY


def _goal_world_x(goal_block_x: int) -> float:
    if int(goal_block_x) < 0:
        return -1.0
    return (float(int(goal_block_x)) + 0.5) * float(CELL_SIZE)


def _cliff_frontier_world_x(cliff_x: jnp.ndarray, cliff_side: jnp.ndarray) -> jnp.ndarray:
    """Return the world-space cliff/gap frontier boundary for the incorrect side."""
    cliff_x = cliff_x.astype(jnp.float32)
    cliff_side = cliff_side.astype(jnp.int32)
    return jnp.where(
        cliff_side == jnp.int32(_SIDE_CODE_LEFT),
        cliff_x * jnp.float32(CELL_SIZE),
        (cliff_x + jnp.float32(1.0)) * jnp.float32(CELL_SIZE),
    )


def _goal_facing_edge_x(
    *,
    min_x: jnp.ndarray,
    front_x: jnp.ndarray,
    goal_direction: jnp.ndarray,
) -> jnp.ndarray:
    """Return the robot edge that leads in the goal direction."""
    return jnp.where(
        goal_direction.astype(jnp.int32) < 0,
        min_x.astype(jnp.float32),
        front_x.astype(jnp.float32),
    )


def _wrong_facing_edge_x(
    *,
    min_x: jnp.ndarray,
    front_x: jnp.ndarray,
    goal_direction: jnp.ndarray,
) -> jnp.ndarray:
    """Return the robot edge that leads in the non-goal direction."""
    return jnp.where(
        goal_direction.astype(jnp.int32) < 0,
        front_x.astype(jnp.float32),
        min_x.astype(jnp.float32),
    )


def _bilateral_progress_metrics(
    *,
    bilateral_task: jnp.ndarray,
    init_com_x: jnp.ndarray,
    init_min_x: jnp.ndarray,
    init_front_x: jnp.ndarray,
    goal_x: jnp.ndarray,
    goal_direction: jnp.ndarray,
    cliff_x: jnp.ndarray,
    cliff_side: jnp.ndarray,
    goal_touched: jnp.ndarray,
    final_min_x: jnp.ndarray,
    final_front_x: jnp.ndarray,
    peak_any_direction_progress: jnp.ndarray,
    peak_goal_progress: jnp.ndarray,
    peak_incorrect_progress: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    init_goal_edge_x = _goal_facing_edge_x(
        min_x=init_min_x,
        front_x=init_front_x,
        goal_direction=goal_direction,
    )
    goal_route = jnp.maximum(
        jnp.abs(goal_x.astype(jnp.float32) - init_goal_edge_x),
        jnp.float32(1e-6),
    )
    any_progress = jnp.where(
        bilateral_task,
        jnp.clip(peak_any_direction_progress.astype(jnp.float32) / goal_route, 0.0, 1.0),
        0.0,
    )
    goal_progress = jnp.where(
        bilateral_task,
        jnp.clip(peak_goal_progress.astype(jnp.float32) / goal_route, 0.0, 1.0),
        0.0,
    )
    init_wrong_edge_x = _wrong_facing_edge_x(
        min_x=init_min_x,
        front_x=init_front_x,
        goal_direction=goal_direction,
    )
    incorrect_route = jnp.maximum(
        jnp.abs(
            _cliff_frontier_world_x(
                cliff_x.astype(jnp.int32),
                cliff_side.astype(jnp.int32),
            )
            - init_wrong_edge_x
        ),
        jnp.float32(1e-6),
    )
    incorrect_progress = jnp.where(
        bilateral_task,
        jnp.clip(
            peak_incorrect_progress.astype(jnp.float32) / incorrect_route,
            0.0,
            1.0,
        ),
        0.0,
    )
    final_wrong_edge_x = _wrong_facing_edge_x(
        min_x=final_min_x,
        front_x=final_front_x,
        goal_direction=goal_direction,
    )
    final_incorrect_displacement = jnp.maximum(
        -goal_direction.astype(jnp.float32) * (final_wrong_edge_x - init_wrong_edge_x),
        0.0,
    )
    reversal_progress = jnp.where(
        bilateral_task,
        jnp.clip(
            (peak_incorrect_progress.astype(jnp.float32) - final_incorrect_displacement)
            / incorrect_route,
            0.0,
            1.0,
        ),
        0.0,
    )
    final_goal_edge_x = _goal_facing_edge_x(
        min_x=final_min_x,
        front_x=final_front_x,
        goal_direction=goal_direction,
    )
    toward_goal_frontier = jnp.where(
        bilateral_task,
        jnp.where(
            goal_touched.astype(jnp.bool_),
            jnp.float32(1.0),
            jnp.clip(
                jnp.float32(1.0)
                - (
                    jnp.abs(goal_x.astype(jnp.float32) - final_goal_edge_x)
                    / goal_route
                ),
                0.0,
                1.0,
            ),
        ),
        0.0,
    )
    return any_progress, goal_progress, incorrect_progress, toward_goal_frontier, reversal_progress


def _episode_turn_motion_threshold(
    *,
    init_min_x: jnp.ndarray,
    init_front_x: jnp.ndarray,
) -> jnp.ndarray:
    """Return the minimum meaningful COM-x displacement for turn bookkeeping."""
    span_x = jnp.maximum(
        jnp.asarray(init_front_x, dtype=jnp.float32) - jnp.asarray(init_min_x, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    return jnp.maximum(
        jnp.float32(_TURN_DIRECTION_DELTA_THRESHOLD_MIN),
        span_x * jnp.float32(_TURN_DIRECTION_DELTA_THRESHOLD_SPAN_FRACTION),
    )


def _update_turn_tracking(
    *,
    last_significant_direction: jnp.ndarray,
    turn_count: jnp.ndarray,
    delta_com_x: jnp.ndarray,
    turn_motion_threshold: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update clipped COM-x turn count from one env-step displacement."""
    step_direction = _significant_step_direction(
        delta_com_x=delta_com_x,
        turn_motion_threshold=turn_motion_threshold,
    )
    significant = step_direction != 0
    flipped = significant & (last_significant_direction != 0) & (step_direction != last_significant_direction)
    next_direction = jnp.where(significant, step_direction, last_significant_direction)
    next_turn_count = jnp.where(
        flipped,
        jnp.minimum(turn_count + jnp.int32(1), jnp.int32(_MAX_TURN_COUNT)),
        turn_count,
    )
    return next_direction, next_turn_count


def _significant_step_direction(
    *,
    delta_com_x: jnp.ndarray,
    turn_motion_threshold: jnp.ndarray,
) -> jnp.ndarray:
    """Return {-1, 0, +1} using the same COM-x thresholding as turn tracking."""
    delta_com_x = jnp.asarray(delta_com_x, dtype=jnp.float32)
    turn_motion_threshold = jnp.asarray(turn_motion_threshold, dtype=jnp.float32)
    significant = jnp.abs(delta_com_x) >= turn_motion_threshold
    step_direction = jnp.where(delta_com_x >= 0.0, jnp.int32(1), jnp.int32(-1))
    return jnp.where(significant, step_direction, jnp.int32(0))


def _update_stable_turn_tracking(
    *,
    stable_turn_direction: jnp.ndarray,
    stable_turn_distance: jnp.ndarray,
    stable_turn_eligible: jnp.ndarray,
    stable_turn_count: jnp.ndarray,
    delta_com_x: jnp.ndarray,
    turn_motion_threshold: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update sustained turn count from cumulative COM-x direction segments."""
    delta_com_x = jnp.asarray(delta_com_x, dtype=jnp.float32)
    _ = turn_motion_threshold
    step_direction = jnp.where(
        delta_com_x >= 0.0,
        jnp.int32(1),
        jnp.int32(-1),
    )
    step_distance = jnp.abs(delta_com_x)
    has_motion = step_distance > jnp.float32(0.0)
    stable_turn_distance = stable_turn_distance.astype(jnp.float32)
    stable_turn_eligible = stable_turn_eligible.astype(jnp.bool_)

    def _no_change() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            stable_turn_direction,
            stable_turn_distance,
            stable_turn_eligible,
            stable_turn_count,
        )

    def _apply_motion() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def _start_segment() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            return step_direction, step_distance, jnp.bool_(False), stable_turn_count

        def _continue_segment() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            next_distance = stable_turn_distance + step_distance
            matured = stable_turn_eligible & (
                next_distance >= jnp.float32(_STABLE_TURN_DISTANCE_THRESHOLD)
            )
            next_count = jnp.where(
                matured,
                jnp.minimum(stable_turn_count + jnp.int32(1), jnp.int32(_MAX_TURN_COUNT)),
                stable_turn_count,
            )
            next_eligible = jnp.where(matured, jnp.bool_(False), stable_turn_eligible)
            return stable_turn_direction, next_distance, next_eligible, next_count

        def _flip_segment() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            matured = step_distance >= jnp.float32(_STABLE_TURN_DISTANCE_THRESHOLD)
            next_count = jnp.where(
                matured,
                jnp.minimum(stable_turn_count + jnp.int32(1), jnp.int32(_MAX_TURN_COUNT)),
                stable_turn_count,
            )
            return (
                step_direction,
                step_distance,
                ~matured,
                next_count,
            )

        return jax.lax.cond(
            stable_turn_direction == 0,
            _start_segment,
            lambda: jax.lax.cond(
                step_direction == stable_turn_direction,
                _continue_segment,
                _flip_segment,
            ),
        )

    return jax.lax.cond(has_motion, _apply_motion, _no_change)


def _terrain_render_object_ids(
    terrain: np.ndarray,
    *,
    goal_block_x: int,
    goal_block_y: int,
) -> np.ndarray:
    object_ids: list[int] = []
    terrain_np = np.asarray(terrain, dtype=np.int32)
    grid_h, grid_w = terrain_np.shape
    for vy in range(grid_h):
        for vx in range(grid_w):
            if int(terrain_np[vy, vx]) == EMPTY:
                continue
            if int(vx) == int(goal_block_x) and int(vy) == int(goal_block_y):
                object_ids.append(int(_GOAL_OBJECT_ID))
            else:
                object_ids.append(int(_TERRAIN_OBJECT_ID))
    return np.asarray(object_ids, dtype=np.int32)


def _robot_touches_goal_block(
    sim_state: SimState,
    robot_point_mask: jnp.ndarray,
    *,
    goal_block_x: jnp.ndarray | int,
    goal_block_y: jnp.ndarray | int,
) -> jnp.ndarray:
    goal_block_x = jnp.asarray(goal_block_x, dtype=jnp.int32)
    goal_block_y = jnp.asarray(goal_block_y, dtype=jnp.int32)
    has_goal = goal_block_x >= 0
    min_x = goal_block_x.astype(jnp.float32) * jnp.float32(CELL_SIZE) - jnp.float32(1e-5)
    max_x = (goal_block_x.astype(jnp.float32) + 1.0) * jnp.float32(CELL_SIZE) + jnp.float32(1e-5)
    min_y = goal_block_y.astype(jnp.float32) * jnp.float32(CELL_SIZE) - jnp.float32(1e-5)
    max_y = (goal_block_y.astype(jnp.float32) + 1.0) * jnp.float32(CELL_SIZE) + jnp.float32(1e-5)
    point_x = sim_state.positions[:, 0]
    point_y = sim_state.positions[:, 1]
    touched = robot_point_mask & (point_x >= min_x) & (point_x <= max_x) & (point_y >= min_y) & (point_y <= max_y)
    return has_goal & jnp.any(touched)


def eval_env_data(env_data: CliffEnvData | CliffEvalEnvData) -> CliffEvalEnvData:
    """Drop render-only assets from an environment payload."""
    if isinstance(env_data, CliffEvalEnvData):
        return env_data
    return CliffEvalEnvData(
        sim_state=env_data.sim_state,
        topology=env_data.topology,
        constants=env_data.constants,
        actuator_info=env_data.actuator_info,
        per_axis_info=env_data.per_axis_info,
        collision_data=env_data.collision_data,
        static_collider_data=env_data.static_collider_data,
        static_collision_worklist=env_data.static_collision_worklist,
        deformation_info=env_data.deformation_info,
        actuator_mask=env_data.actuator_mask,
        deform_mask=env_data.deform_mask,
        robot_point_mask=env_data.robot_point_mask,
        robot_point_count=env_data.robot_point_count,
        grid_h=env_data.grid_h,
        grid_w=env_data.grid_w,
        spawn_x=env_data.spawn_x,
        spawn_y=env_data.spawn_y,
        robot_h=env_data.robot_h,
        robot_w=env_data.robot_w,
        death_plane_y=env_data.death_plane_y,
        init_com_y=env_data.init_com_y,
        init_com_x=env_data.init_com_x,
        init_min_x=env_data.init_min_x,
        init_front_x=env_data.init_front_x,
        cliff_x=env_data.cliff_x,
        task_mode=env_data.task_mode,
        goal_direction=env_data.goal_direction,
        cliff_side=env_data.cliff_side,
        goal_distance=env_data.goal_distance,
        cliff_distance=env_data.cliff_distance,
        goal_block_x=env_data.goal_block_x,
        goal_block_y=env_data.goal_block_y,
        goal_x=env_data.goal_x,
        goal_object_id=env_data.goal_object_id,
        cliff_probe_ledge_width=env_data.cliff_probe_ledge_width,
        cliff_probe_ledge_cell_x=env_data.cliff_probe_ledge_cell_x,
        cliff_probe_ledge_cell_y=env_data.cliff_probe_ledge_cell_y,
    )


def render_assets(env_data: CliffEnvData) -> CliffRenderAssets:
    """Extract render-only assets from a full environment payload."""
    return CliffRenderAssets(render_info=env_data.render_info)


def _empty_static_collider_data() -> StaticColliderData:
    return StaticColliderData(
        point_positions=jnp.zeros((0, 2), dtype=jnp.float32),
        cell_vertices=jnp.zeros((0, 4), dtype=jnp.int32),
        cell_vertex_count=jnp.zeros((0,), dtype=jnp.int32),
        cell_edge_a=jnp.zeros((0, 4), dtype=jnp.int32),
        cell_edge_b=jnp.zeros((0, 4), dtype=jnp.int32),
        surface_edge_mask=jnp.zeros((0, 4), dtype=jnp.bool_),
        cell_types=jnp.zeros((0,), dtype=jnp.int32),
        cell_aabb_min=jnp.zeros((0, 2), dtype=jnp.float32),
        cell_aabb_max=jnp.zeros((0, 2), dtype=jnp.float32),
        column_cell_indices=jnp.full((0, 0), -1, dtype=jnp.int32),
        column_cell_count=jnp.zeros((0,), dtype=jnp.int32),
    )


def infer_static_collider_capacity(
    env_datas: list[CliffEnvData | CliffEvalEnvData],
) -> StaticColliderCapacity:
    """Infer fixed collider capacities across an environment collection."""
    if not env_datas:
        raise ValueError("Cannot infer static collider capacity from an empty environment list")
    eval_envs = [eval_env_data(env_data) for env_data in env_datas]
    return StaticColliderCapacity(
        max_static_points=max(
            int(env_data.static_collider_data.point_positions.shape[0]) for env_data in eval_envs
        ),
        max_static_cells=max(
            int(env_data.static_collider_data.cell_vertices.shape[0]) for env_data in eval_envs
        ),
        max_grid_columns=max(
            int(env_data.static_collider_data.column_cell_indices.shape[0]) for env_data in eval_envs
        ),
        max_cells_per_column=max(
            int(env_data.static_collider_data.column_cell_indices.shape[1]) for env_data in eval_envs
        ),
    )


def infer_static_template_capacity(
    templates: list[CliffStaticTemplate],
) -> StaticColliderCapacity:
    """Infer fixed capacities across terrain-only static templates."""
    if not templates:
        raise ValueError("Cannot infer static template capacity from an empty template list")
    return StaticColliderCapacity(
        max_static_points=max(
            int(template.static_collider_data.point_positions.shape[0]) for template in templates
        ),
        max_static_cells=max(
            int(template.static_collider_data.cell_vertices.shape[0]) for template in templates
        ),
        max_grid_columns=max(
            int(template.static_collider_data.column_cell_indices.shape[0]) for template in templates
        ),
        max_cells_per_column=max(
            int(template.static_collider_data.column_cell_indices.shape[1]) for template in templates
        ),
    )


def _pad_static_collider_data(
    collider_data: StaticColliderData,
    *,
    max_static_points: int,
    max_static_cells: int,
    max_grid_columns: int,
    max_cells_per_column: int,
) -> StaticColliderData:
    """Pad static collider arrays to fixed capacities for batched evaluation."""
    point_count = int(collider_data.point_positions.shape[0])
    cell_count = int(collider_data.cell_vertices.shape[0])
    grid_column_count = int(collider_data.column_cell_indices.shape[0])
    per_column_capacity = int(collider_data.column_cell_indices.shape[1])

    point_positions = jnp.pad(
        collider_data.point_positions,
        ((0, max_static_points - point_count), (0, 0)),
    )
    cell_vertices = jnp.pad(
        collider_data.cell_vertices,
        ((0, max_static_cells - cell_count), (0, 0)),
    )
    cell_vertex_count = jnp.pad(
        collider_data.cell_vertex_count,
        ((0, max_static_cells - cell_count),),
    )
    cell_edge_a = jnp.pad(
        collider_data.cell_edge_a,
        ((0, max_static_cells - cell_count), (0, 0)),
    )
    cell_edge_b = jnp.pad(
        collider_data.cell_edge_b,
        ((0, max_static_cells - cell_count), (0, 0)),
    )
    surface_edge_mask = jnp.pad(
        collider_data.surface_edge_mask,
        ((0, max_static_cells - cell_count), (0, 0)),
        constant_values=False,
    )
    cell_types = jnp.pad(
        collider_data.cell_types,
        ((0, max_static_cells - cell_count),),
    )
    cell_aabb_min = jnp.pad(
        collider_data.cell_aabb_min,
        ((0, max_static_cells - cell_count), (0, 0)),
    )
    cell_aabb_max = jnp.pad(
        collider_data.cell_aabb_max,
        ((0, max_static_cells - cell_count), (0, 0)),
    )
    column_cell_indices = jnp.pad(
        collider_data.column_cell_indices,
        (
            (0, max_grid_columns - grid_column_count),
            (0, max_cells_per_column - per_column_capacity),
        ),
        constant_values=-1,
    )
    column_cell_count = jnp.pad(
        collider_data.column_cell_count,
        ((0, max_grid_columns - grid_column_count),),
    )
    return StaticColliderData(
        point_positions=point_positions,
        cell_vertices=cell_vertices,
        cell_vertex_count=cell_vertex_count,
        cell_edge_a=cell_edge_a,
        cell_edge_b=cell_edge_b,
        surface_edge_mask=surface_edge_mask,
        cell_types=cell_types,
        cell_aabb_min=cell_aabb_min,
        cell_aabb_max=cell_aabb_max,
        column_cell_indices=column_cell_indices,
        column_cell_count=column_cell_count,
    )


def stack_eval_envs(
    env_datas: list[CliffEnvData | CliffEvalEnvData],
    *,
    static_capacity: StaticColliderCapacity | None = None,
) -> CliffEvalEnvData:
    """Stack array-only environment payloads along a terrain batch axis."""
    if not env_datas:
        raise ValueError("Cannot stack an empty environment batch")
    eval_envs = [eval_env_data(env_data) for env_data in env_datas]
    capacity = static_capacity or infer_static_collider_capacity(eval_envs)
    eval_envs = [
        env_data._replace(
            static_collider_data=padded_static,
            static_collision_worklist=build_static_collision_worklist(
                env_data.collision_data,
                padded_static,
                max_surface_boxels=int(env_data.robot_h) * int(env_data.robot_w),
            ),
        )
        for env_data in eval_envs
        for padded_static in (
            _pad_static_collider_data(
                env_data.static_collider_data,
                max_static_points=capacity.max_static_points,
                max_static_cells=capacity.max_static_cells,
                max_grid_columns=capacity.max_grid_columns,
                max_cells_per_column=capacity.max_cells_per_column,
            ),
        )
    ]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *eval_envs)


def stack_static_templates(
    templates: list[CliffStaticTemplate],
    *,
    static_capacity: StaticColliderCapacity | None = None,
) -> CliffStaticTemplateBatch:
    """Pad and stack terrain-only static templates along a terrain batch axis."""
    if not templates:
        raise ValueError("Cannot stack an empty template batch")
    capacity = static_capacity or infer_static_template_capacity(templates)
    padded = [
        template._replace(
            static_collider_data=_pad_static_collider_data(
                template.static_collider_data,
                max_static_points=capacity.max_static_points,
                max_static_cells=capacity.max_static_cells,
                max_grid_columns=capacity.max_grid_columns,
                max_cells_per_column=capacity.max_cells_per_column,
            )
        )
        for template in templates
    ]
    return CliffStaticTemplateBatch(
        static_collider_data=jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *(template.static_collider_data for template in padded),
        ),
        cliff_x=jnp.asarray([template.cliff_x for template in padded], dtype=jnp.int32),
        death_plane_y=jnp.asarray(
            [template.death_plane_y for template in padded],
            dtype=jnp.float32,
        ),
        task_mode=jnp.asarray([template.task_mode for template in padded], dtype=jnp.int32),
        goal_direction=jnp.asarray([template.goal_direction for template in padded], dtype=jnp.int32),
        cliff_side=jnp.asarray([template.cliff_side for template in padded], dtype=jnp.int32),
        goal_distance=jnp.asarray([template.goal_distance for template in padded], dtype=jnp.int32),
        cliff_distance=jnp.asarray([template.cliff_distance for template in padded], dtype=jnp.int32),
        goal_block_x=jnp.asarray([template.goal_block_x for template in padded], dtype=jnp.int32),
        goal_block_y=jnp.asarray([template.goal_block_y for template in padded], dtype=jnp.int32),
        goal_x=jnp.asarray([template.goal_x for template in padded], dtype=jnp.float32),
        goal_object_id=jnp.asarray([template.goal_object_id for template in padded], dtype=jnp.int32),
        cliff_probe_ledge_width=jnp.asarray(
            [template.cliff_probe_ledge_width for template in padded],
            dtype=jnp.int32,
        ),
        cliff_probe_ledge_cell_x=jnp.asarray(
            [template.cliff_probe_ledge_cell_x for template in padded],
            dtype=jnp.int32,
        ),
        cliff_probe_ledge_cell_y=jnp.asarray(
            [template.cliff_probe_ledge_cell_y for template in padded],
            dtype=jnp.int32,
        ),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


_ACTUATOR_TYPES = frozenset({C.H_ACT, C.V_ACT, C.CONTRACTILE})


def _terrain_without_robot_footprint(
    terrain: np.ndarray, robot: np.ndarray, spawn_x: int, spawn_y: int
) -> np.ndarray:
    """Erase robot footprint from terrain so they don't overlap in EvoWorld."""
    out = terrain.copy()
    rh, rw = robot.shape
    for ry in range(rh):
        for rx in range(rw):
            if robot[ry, rx] != EMPTY:
                wy, wx = spawn_y + ry, spawn_x + rx
                if 0 <= wy < out.shape[0] and 0 <= wx < out.shape[1]:
                    out[wy, wx] = EMPTY
    return out


def _instantiate_cliff_world(
    terrain: np.ndarray | None,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
    *,
    spring_stiffness_scale: float,
    collision_build_capacity: CollisionBuildCapacity | None,
):
    """Instantiate either a terrain+robot world or a robot-only world."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    world = EvoWorld()
    if terrain is not None:
        terrain_np = np.asarray(terrain, dtype=np.int32)
        terrain_clean = _terrain_without_robot_footprint(terrain_np, robot_np, spawn_x, spawn_y)
        if np.any(terrain_clean != EMPTY):
            world.add_from_array("terrain", np.flipud(terrain_clean), 0, 0)
    world.add_from_array("robot", np.flipud(robot_np), spawn_x, spawn_y)

    template_set = compile_world_templates(
        world,
        robot_name="robot",
        mirror_mode="none",
        spring_stiffness_scale=spring_stiffness_scale,
    )
    instantiate_kwargs: dict[str, int] = {}
    if collision_build_capacity is not None:
        instantiate_kwargs["max_triples"] = int(collision_build_capacity.max_triples)
        instantiate_kwargs["max_self_triples"] = int(collision_build_capacity.max_self_triples)
    return instantiate_world(template_set.primary, **instantiate_kwargs)


def _world_masks(
    robot_morph: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    spawn_x: int,
    spawn_y: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build world-aligned actuator/deformation masks for the robot footprint."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    actuator_mask_np = np.zeros((grid_h * grid_w,), dtype=bool)
    deform_mask_np = np.zeros((grid_h * grid_w,), dtype=bool)
    robot_h, robot_w = robot_np.shape
    for ry in range(robot_h):
        for rx in range(robot_w):
            if robot_np[ry, rx] == EMPTY:
                continue
            idx = (spawn_y + ry) * grid_w + (spawn_x + rx)
            deform_mask_np[idx] = True
            actuator_mask_np[idx] = int(robot_np[ry, rx]) in _ACTUATOR_TYPES
    return (
        jnp.asarray(actuator_mask_np, dtype=jnp.bool_),
        jnp.asarray(deform_mask_np, dtype=jnp.bool_),
    )


def _robot_world_slot_maps(
    robot_morph: np.ndarray,
    *,
    grid_w: int,
    spawn_x: int,
    spawn_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense and active world-slot mappings for a robot bounding box."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    robot_h, robot_w = robot_np.shape
    robot_row_offsets = (
        np.int32(spawn_y) + np.arange(robot_h, dtype=np.int32)
    )[:, None] * np.int32(grid_w)
    robot_col_offsets = (
        np.int32(spawn_x) + np.arange(robot_w, dtype=np.int32)
    )[None, :]
    robot_world_slots = (robot_row_offsets + robot_col_offsets).reshape(-1)
    active_local_mask = robot_np.reshape(-1) != EMPTY
    active_world_slots = robot_world_slots[active_local_mask]
    return robot_world_slots, active_world_slots, active_local_mask


def _world_aligned_per_axis_info(
    local_info: PerAxisActuatorInfo,
    robot_morph: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    spawn_x: int,
    spawn_y: int,
) -> PerAxisActuatorInfo:
    """Map compact actuator cell indices into world-grid slots."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    robot_h, robot_w = robot_np.shape
    local_cell_count = int(robot_h * robot_w)
    robot_world_slots, active_world_slots, _ = _robot_world_slot_maps(
        robot_np,
        grid_w=grid_w,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )

    world_cell_count = int(grid_h * grid_w)
    padded_h_pairs = np.zeros((world_cell_count, 2), dtype=np.int32)
    padded_v_pairs = np.zeros((world_cell_count, 2), dtype=np.int32)
    local_h_pairs = np.asarray(local_info.h_spring_pairs, dtype=np.int32)
    local_v_pairs = np.asarray(local_info.v_spring_pairs, dtype=np.int32)
    if local_h_pairs.shape[0] == local_cell_count:
        padded_h_pairs[robot_world_slots] = local_h_pairs
    elif local_h_pairs.shape[0] == active_world_slots.shape[0]:
        padded_h_pairs[active_world_slots] = local_h_pairs
    else:
        raise ValueError(
            "Per-axis actuator metadata does not match robot bounding box or active-cell count"
        )
    if local_v_pairs.shape[0] == local_cell_count:
        padded_v_pairs[robot_world_slots] = local_v_pairs
    elif local_v_pairs.shape[0] == active_world_slots.shape[0]:
        padded_v_pairs[active_world_slots] = local_v_pairs
    else:
        raise ValueError(
            "Per-axis actuator metadata does not match robot bounding box or active-cell count"
        )

    h_compact_world_indices = np.zeros((local_cell_count,), dtype=np.int32)
    v_compact_world_indices = np.zeros((local_cell_count,), dtype=np.int32)
    h_compact_count = int(np.asarray(local_info.h_compact_count))
    v_compact_count = int(np.asarray(local_info.v_compact_count))
    local_h_indices = np.asarray(local_info.h_compact_cell_indices, dtype=np.int32)
    local_v_indices = np.asarray(local_info.v_compact_cell_indices, dtype=np.int32)
    compact_slot_lookup = (
        active_world_slots
        if local_h_pairs.shape[0] == active_world_slots.shape[0]
        else robot_world_slots
    )
    if h_compact_count > 0:
        h_compact_world_indices[:h_compact_count] = compact_slot_lookup[
            local_h_indices[:h_compact_count]
        ]
    compact_slot_lookup = (
        active_world_slots
        if local_v_pairs.shape[0] == active_world_slots.shape[0]
        else robot_world_slots
    )
    if v_compact_count > 0:
        v_compact_world_indices[:v_compact_count] = compact_slot_lookup[
            local_v_indices[:v_compact_count]
        ]

    return PerAxisActuatorInfo(
        h_spring_pairs=jnp.asarray(padded_h_pairs, dtype=jnp.int32),
        v_spring_pairs=jnp.asarray(padded_v_pairs, dtype=jnp.int32),
        h_compact_cell_indices=jnp.asarray(h_compact_world_indices, dtype=jnp.int32),
        v_compact_cell_indices=jnp.asarray(v_compact_world_indices, dtype=jnp.int32),
        h_compact_spring_pairs=local_info.h_compact_spring_pairs,
        v_compact_spring_pairs=local_info.v_compact_spring_pairs,
        h_compact_count=local_info.h_compact_count,
        v_compact_count=local_info.v_compact_count,
        actuated_spring_mask=local_info.actuated_spring_mask,
        spring_act_count=local_info.spring_act_count,
    )


def _world_aligned_deformation_info(
    local_info: DeformationInfo,
    robot_morph: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    spawn_x: int,
    spawn_y: int,
) -> DeformationInfo:
    """Pad robot-local deformation metadata into world-grid slots."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    robot_h, robot_w = robot_np.shape
    robot_world_slots, active_world_slots, _ = _robot_world_slot_maps(
        robot_np,
        grid_w=grid_w,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )

    world_cell_count = int(grid_h * grid_w)
    padded_h = np.full((world_cell_count, 2), -1, dtype=np.int32)
    padded_v = np.full((world_cell_count, 2), -1, dtype=np.int32)
    local_h = np.asarray(local_info.voxel_horiz_springs, dtype=np.int32)
    local_v = np.asarray(local_info.voxel_vert_springs, dtype=np.int32)
    if local_h.shape[0] == int(robot_h * robot_w):
        padded_h[robot_world_slots] = local_h
    elif local_h.shape[0] == active_world_slots.shape[0]:
        padded_h[active_world_slots] = local_h
    else:
        raise ValueError(
            "Deformation metadata does not match robot bounding box or active-cell count"
        )
    if local_v.shape[0] == int(robot_h * robot_w):
        padded_v[robot_world_slots] = local_v
    elif local_v.shape[0] == active_world_slots.shape[0]:
        padded_v[active_world_slots] = local_v
    else:
        raise ValueError(
            "Deformation metadata does not match robot bounding box or active-cell count"
        )
    return DeformationInfo(
        voxel_horiz_springs=jnp.asarray(padded_h, dtype=jnp.int32),
        voxel_vert_springs=jnp.asarray(padded_v, dtype=jnp.int32),
        n_robot_voxels=int(world_cell_count),
    )


def _terrain_static_geometry(
    terrain: np.ndarray,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
):
    terrain_clean = _terrain_without_robot_footprint(
        np.asarray(terrain, dtype=np.int32),
        np.asarray(robot_morph, dtype=np.int32),
        spawn_x,
        spawn_y,
    )
    return build_sparse_cell_geometry(grid_active_cells(terrain_clean))


def _task_fields_from_metadata(task_metadata: dict[str, object] | None) -> dict[str, int | float]:
    metadata = dict(task_metadata or {})
    task_mode = _task_mode_code(metadata)
    goal_block_x = int(metadata.get("goal_block_x", -1))
    goal_block_y = int(metadata.get("goal_block_y", -1))
    raw_cliff_side = metadata.get("cliff_side", "right" if task_mode == _TASK_MODE_CODE_LEGACY else None)
    cliff_side = int(raw_cliff_side) if isinstance(raw_cliff_side, int) else _side_code(raw_cliff_side)
    return {
        "task_mode": int(task_mode),
        "goal_direction": int(
            metadata.get("goal_direction", _SIDE_CODE_NONE if task_mode == _TASK_MODE_CODE_LEGACY else _SIDE_CODE_RIGHT)
        ),
        "cliff_side": int(cliff_side),
        "goal_distance": int(metadata.get("goal_distance", 0)),
        "cliff_distance": int(metadata.get("cliff_distance", metadata.get("goal_distance", 0))),
        "goal_block_x": goal_block_x,
        "goal_block_y": goal_block_y,
        "goal_x": float(metadata.get("goal_x", _goal_world_x(goal_block_x))),
        "goal_object_id": int(metadata.get("goal_object_id", _GOAL_OBJECT_ID if goal_block_x >= 0 else 0)),
        "cliff_probe_ledge_width": int(metadata.get("cliff_probe_ledge_width", 0)),
        "cliff_probe_ledge_cell_x": int(metadata.get("cliff_probe_ledge_cell_x", -1)),
        "cliff_probe_ledge_cell_y": int(metadata.get("cliff_probe_ledge_cell_y", -1)),
    }


def _terrain_death_plane_y(terrain: np.ndarray) -> float:
    """Return a slightly permissive death plane below the lowest terrain row."""
    terrain_np = np.asarray(terrain, dtype=np.int32)
    occupied_rows = np.flatnonzero(np.any(terrain_np != EMPTY, axis=1))
    if occupied_rows.size == 0:
        return float(CELL_SIZE)
    terrain_floor = float(int(occupied_rows.min()) * CELL_SIZE)
    return max(0.0, terrain_floor - float(_DEATH_PLANE_MARGIN))


def _terrain_render_fragment(
    terrain_geometry,
    *,
    grid_h: int,
    grid_w: int,
    cell_object_id: np.ndarray | None = None,
) -> RenderInfo:
    n_cells = int(terrain_geometry.cell_vertices.shape[0])
    neg_pairs = np.full((n_cells, 2), -1, dtype=np.int32)
    object_ids = (
        np.asarray(cell_object_id, dtype=np.int32)
        if cell_object_id is not None
        else np.full((n_cells,), _TERRAIN_OBJECT_ID, dtype=np.int32)
    )
    return RenderInfo(
        cell_vertices=np.asarray(terrain_geometry.cell_vertices, dtype=np.int32),
        cell_vertex_count=np.asarray(terrain_geometry.cell_vertex_count, dtype=np.int32),
        cell_types=np.asarray(terrain_geometry.cell_types, dtype=np.int32),
        cell_mask=np.asarray(terrain_geometry.cell_mask, dtype=bool),
        surface_edge_mask=np.asarray(terrain_geometry.surface_edge_mask, dtype=bool),
        cell_is_robot=np.zeros((n_cells,), dtype=bool),
        cell_is_dynamic=np.zeros((n_cells,), dtype=bool),
        cell_object_id=object_ids,
        robot_point_indices=np.zeros((0,), dtype=np.int32),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        actuator_cell_indices=np.zeros((0,), dtype=np.int32),
        cell_h_spring_pairs=neg_pairs,
        cell_v_spring_pairs=neg_pairs.copy(),
        spring_init_rest_length=np.zeros((0,), dtype=np.float32),
        static_point_positions=np.asarray(terrain_geometry.point_positions, dtype=np.float32),
    )


def _surface_edge_mask_for_grid(grid: np.ndarray) -> np.ndarray:
    """Compute boundary edges for a local morphology grid."""
    grid_np = np.asarray(grid, dtype=np.int32)
    grid_h, grid_w = grid_np.shape
    surface = np.zeros((grid_h * grid_w, 4), dtype=bool)
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for vy in range(grid_h):
        for vx in range(grid_w):
            idx = vy * grid_w + vx
            if grid_np[vy, vx] == EMPTY:
                continue
            for edge_idx, (dy, dx) in enumerate(offsets):
                ny = vy + dy
                nx = vx + dx
                surface[idx, edge_idx] = (
                    ny < 0
                    or ny >= grid_h
                    or nx < 0
                    or nx >= grid_w
                    or grid_np[ny, nx] == EMPTY
                )
    return surface


def build_grown_dynamic_render_info(
    robot_morph: np.ndarray | jnp.ndarray,
    grid_data: GridData,
    *,
    spawn_x: int,
    spawn_y: int,
    spring_init_rest_length: np.ndarray | jnp.ndarray,
) -> RenderInfo:
    """Build full-world dynamic render metadata for a grown candidate."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    robot_h, robot_w = robot_np.shape
    local_cell_count = int(robot_h * robot_w)
    robot_flat = robot_np.reshape(-1)
    active_local_mask = robot_flat != EMPTY

    robot_row_offsets = (
        np.int32(spawn_y) + np.arange(robot_h, dtype=np.int32)
    )[:, None] * np.int32(grid_data.W)
    robot_col_offsets = (
        np.int32(spawn_x) + np.arange(robot_w, dtype=np.int32)
    )[None, :]
    robot_world_slots = (robot_row_offsets + robot_col_offsets).reshape(-1)
    active_world_slots = robot_world_slots[active_local_mask]

    cell_vertices_all = np.asarray(grid_data.cell_corner_indices, dtype=np.int32)
    cell_horiz_bottom = np.asarray(grid_data.cell_horiz_bottom, dtype=np.int32)
    cell_horiz_top = np.asarray(grid_data.cell_horiz_top, dtype=np.int32)
    cell_vert_left = np.asarray(grid_data.cell_vert_left, dtype=np.int32)
    cell_vert_right = np.asarray(grid_data.cell_vert_right, dtype=np.int32)
    surface_edge_mask = _surface_edge_mask_for_grid(robot_np).reshape(local_cell_count, 4)

    if active_world_slots.size == 0:
        cell_vertices = np.zeros((0, 4), dtype=np.int32)
        cell_types = np.zeros((0,), dtype=np.int32)
        cell_vertex_count = np.zeros((0,), dtype=np.int32)
        cell_mask = np.zeros((0,), dtype=bool)
        cell_surface = np.zeros((0, 4), dtype=bool)
        robot_point_indices = np.zeros((0,), dtype=np.int32)
        actuator_cell_indices = np.zeros((0,), dtype=np.int32)
        cell_h_pairs = np.zeros((0, 2), dtype=np.int32)
        cell_v_pairs = np.zeros((0, 2), dtype=np.int32)
    else:
        cell_vertices = cell_vertices_all[active_world_slots]
        cell_types = robot_flat[active_local_mask].astype(np.int32, copy=False)
        cell_vertex_count = np.full((active_world_slots.shape[0],), 4, dtype=np.int32)
        cell_mask = np.ones((active_world_slots.shape[0],), dtype=bool)
        cell_surface = surface_edge_mask[active_local_mask]
        robot_point_indices = np.unique(cell_vertices.reshape(-1)).astype(np.int32, copy=False)
        actuator_cell_indices = np.flatnonzero(
            np.isin(
                cell_types,
                np.array([C.H_ACT, C.V_ACT, C.CONTRACTILE], dtype=np.int32),
            )
        ).astype(np.int32, copy=False)
        cell_h_pairs = np.stack(
            [cell_horiz_bottom[active_world_slots], cell_horiz_top[active_world_slots]],
            axis=1,
        ).astype(np.int32, copy=False)
        cell_v_pairs = np.stack(
            [cell_vert_left[active_world_slots], cell_vert_right[active_world_slots]],
            axis=1,
        ).astype(np.int32, copy=False)

    return RenderInfo(
        cell_vertices=cell_vertices,
        cell_vertex_count=cell_vertex_count,
        cell_types=cell_types,
        cell_mask=cell_mask,
        surface_edge_mask=cell_surface,
        cell_is_robot=np.ones((cell_vertices.shape[0],), dtype=bool),
        cell_is_dynamic=np.ones((cell_vertices.shape[0],), dtype=bool),
        cell_object_id=np.zeros((cell_vertices.shape[0],), dtype=np.int32),
        robot_point_indices=robot_point_indices,
        grid_h=int(grid_data.H),
        grid_w=int(grid_data.W),
        actuator_cell_indices=actuator_cell_indices,
        cell_h_spring_pairs=cell_h_pairs,
        cell_v_spring_pairs=cell_v_pairs,
        spring_init_rest_length=np.asarray(spring_init_rest_length, dtype=np.float32),
        static_point_positions=np.zeros((0, 2), dtype=np.float32),
    )


def merge_render_info(
    dynamic_render: RenderInfo | None,
    terrain_render: RenderInfo | None,
    *,
    n_dynamic_points: int,
    grid_h: int,
    grid_w: int,
) -> RenderInfo | None:
    if dynamic_render is None:
        return terrain_render
    if terrain_render is None:
        return dynamic_render._replace(grid_h=int(grid_h), grid_w=int(grid_w))

    dynamic_static_points = np.asarray(dynamic_render.static_point_positions, dtype=np.float32)
    if dynamic_static_points.size != 0:
        raise ValueError("dynamic core render info must not carry static points")

    terrain_offset = int(n_dynamic_points)
    terrain_cell_vertices = np.asarray(terrain_render.cell_vertices, dtype=np.int32) + terrain_offset
    return RenderInfo(
        cell_vertices=np.concatenate(
            [
                np.asarray(dynamic_render.cell_vertices, dtype=np.int32),
                terrain_cell_vertices,
            ],
            axis=0,
        ),
        cell_vertex_count=np.concatenate(
            [
                np.asarray(dynamic_render.cell_vertex_count, dtype=np.int32),
                np.asarray(terrain_render.cell_vertex_count, dtype=np.int32),
            ],
            axis=0,
        ),
        cell_types=np.concatenate(
            [
                np.asarray(dynamic_render.cell_types, dtype=np.int32),
                np.asarray(terrain_render.cell_types, dtype=np.int32),
            ],
            axis=0,
        ),
        cell_mask=np.concatenate(
            [
                np.asarray(dynamic_render.cell_mask, dtype=bool),
                np.asarray(terrain_render.cell_mask, dtype=bool),
            ],
            axis=0,
        ),
        surface_edge_mask=np.concatenate(
            [
                np.asarray(dynamic_render.surface_edge_mask, dtype=bool),
                np.asarray(terrain_render.surface_edge_mask, dtype=bool),
            ],
            axis=0,
        ),
        cell_is_robot=np.concatenate(
            [
                np.asarray(dynamic_render.cell_is_robot, dtype=bool),
                np.asarray(terrain_render.cell_is_robot, dtype=bool),
            ],
            axis=0,
        ),
        cell_is_dynamic=np.concatenate(
            [
                np.asarray(dynamic_render.cell_is_dynamic, dtype=bool),
                np.asarray(terrain_render.cell_is_dynamic, dtype=bool),
            ],
            axis=0,
        ),
        cell_object_id=np.concatenate(
            [
                np.asarray(dynamic_render.cell_object_id, dtype=np.int32),
                np.asarray(terrain_render.cell_object_id, dtype=np.int32),
            ],
            axis=0,
        ),
        robot_point_indices=np.asarray(dynamic_render.robot_point_indices, dtype=np.int32),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        actuator_cell_indices=np.asarray(dynamic_render.actuator_cell_indices, dtype=np.int32),
        cell_h_spring_pairs=np.concatenate(
            [
                np.asarray(dynamic_render.cell_h_spring_pairs, dtype=np.int32),
                np.asarray(terrain_render.cell_h_spring_pairs, dtype=np.int32),
            ],
            axis=0,
        ),
        cell_v_spring_pairs=np.concatenate(
            [
                np.asarray(dynamic_render.cell_v_spring_pairs, dtype=np.int32),
                np.asarray(terrain_render.cell_v_spring_pairs, dtype=np.int32),
            ],
            axis=0,
        ),
        spring_init_rest_length=np.asarray(dynamic_render.spring_init_rest_length, dtype=np.float32),
        static_point_positions=np.asarray(terrain_render.static_point_positions, dtype=np.float32),
    )


def build_cliff_dynamic_core(
    robot_morph: np.ndarray,
    *,
    spawn_x: int,
    spawn_y: int,
    physics_substeps: int = REFERENCE_SUBSTEPS,
    actuator_convergence: float = 0.003,
    spring_stiffness_scale: float = 0.8,
    floor_static_friction_const: float = 0.0,
    floor_stiction_stiffness: float = 0.0,
    floor_stiction_switch_speed: float = 0.0,
    collision_build_capacity: CollisionBuildCapacity | None = None,
) -> CliffDynamicCore:
    """Build the robot-only dynamic simulation core once for a fixed morphology."""
    robot_np = np.asarray(robot_morph, dtype=np.int32)
    robot_h, robot_w = robot_np.shape
    built = _instantiate_cliff_world(
        None,
        robot_np,
        spawn_x,
        spawn_y,
        spring_stiffness_scale=spring_stiffness_scale,
        collision_build_capacity=collision_build_capacity,
    )
    dt, effective_convergence = derive_physics_params(
        physics_substeps,
        actuator_convergence,
    )
    constants = built.constants._replace(
        dt=dt,
        actuator_convergence=effective_convergence,
        floor_static_friction_const=float(floor_static_friction_const),
        floor_stiction_stiffness=float(floor_stiction_stiffness),
        floor_stiction_switch_speed=float(floor_stiction_switch_speed),
        diagnostics_enabled=False,
    )
    positions = np.asarray(built.sim_state.positions)
    mask = np.asarray(built.robot_point_mask)
    robot_x = positions[mask, 0]
    robot_y = positions[mask, 1]
    init_min_x = float(np.min(robot_x)) if len(robot_x) > 0 else 0.0
    init_front_x = float(np.max(robot_x)) if len(robot_x) > 0 else 0.0
    init_com_x = float(np.mean(robot_x)) if len(robot_x) > 0 else 0.0
    init_com_y = float(np.mean(robot_y)) if len(robot_y) > 0 else 0.0

    return CliffDynamicCore(
        sim_state=built.sim_state,
        topology=built.topology,
        constants=constants,
        actuator_info=built.actuator_info,
        per_axis_info=built.per_axis_info,
        collision_data=built.collision_data,
        deformation_info=built.deformation_info,
        robot_point_mask=built.robot_point_mask,
        robot_point_count=jnp.int32(int(np.sum(mask))),
        robot_morph=jnp.asarray(robot_np, dtype=jnp.int32),
        spawn_x=int(spawn_x),
        spawn_y=int(spawn_y),
        robot_h=robot_h,
        robot_w=robot_w,
        init_com_y=init_com_y,
        init_com_x=init_com_x,
        init_min_x=init_min_x,
        init_front_x=init_front_x,
        render_info=built.render_info,
    )


def build_cliff_terrain_assets(
    terrain: np.ndarray,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
    cliff_x: int,
    *,
    task_metadata: dict[str, object] | None = None,
    spring_stiffness_scale: float = 0.8,
    collision_build_capacity: CollisionBuildCapacity | None = None,
) -> CliffTerrainAssets:
    """Build terrain-specific static collider and render assets.

    `collision_build_capacity` is retained for API compatibility; static terrain
    assets no longer allocate a dynamic-triple buffer.
    """
    del spring_stiffness_scale
    del collision_build_capacity
    terrain_np = np.asarray(terrain, dtype=np.int32)
    geometry = _terrain_static_geometry(
        terrain_np,
        np.asarray(robot_morph, dtype=np.int32),
        spawn_x,
        spawn_y,
    )
    terrain_clean = _terrain_without_robot_footprint(
        terrain_np,
        np.asarray(robot_morph, dtype=np.int32),
        spawn_x,
        spawn_y,
    )
    task_fields = _task_fields_from_metadata(task_metadata)
    grid_h, grid_w = terrain_np.shape
    return CliffTerrainAssets(
        static_collider_template=make_static_collider_data(
            point_positions=np.asarray(geometry.point_positions, dtype=np.float32),
            cell_vertices=np.asarray(geometry.cell_vertices, dtype=np.int32),
            cell_vertex_count=np.asarray(geometry.cell_vertex_count, dtype=np.int32),
            cell_edge_a=np.asarray(geometry.cell_edge_a, dtype=np.int32),
            cell_edge_b=np.asarray(geometry.cell_edge_b, dtype=np.int32),
            surface_edge_mask=np.asarray(geometry.surface_edge_mask, dtype=bool),
            cell_types=np.asarray(geometry.cell_types, dtype=np.int32),
        ),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        cliff_x=int(cliff_x),
        death_plane_y=_terrain_death_plane_y(terrain_np),
        task_mode=int(task_fields["task_mode"]),
        goal_direction=int(task_fields["goal_direction"]),
        cliff_side=int(task_fields["cliff_side"]),
        goal_distance=int(task_fields["goal_distance"]),
        cliff_distance=int(task_fields["cliff_distance"]),
        goal_block_x=int(task_fields["goal_block_x"]),
        goal_block_y=int(task_fields["goal_block_y"]),
        goal_x=float(task_fields["goal_x"]),
        goal_object_id=int(task_fields["goal_object_id"]),
        cliff_probe_ledge_width=int(task_fields["cliff_probe_ledge_width"]),
        cliff_probe_ledge_cell_x=int(task_fields["cliff_probe_ledge_cell_x"]),
        cliff_probe_ledge_cell_y=int(task_fields["cliff_probe_ledge_cell_y"]),
        render_info=_terrain_render_fragment(
            geometry,
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            cell_object_id=_terrain_render_object_ids(
                terrain_clean,
                goal_block_x=int(task_fields["goal_block_x"]),
                goal_block_y=int(task_fields["goal_block_y"]),
            ),
        ),
    )


def assemble_cliff_env(
    dynamic_core: CliffDynamicCore,
    terrain_assets: CliffTerrainAssets,
) -> CliffEnvData:
    """Assemble a terrain-specific environment from shared dynamic core + static assets."""
    robot_morph = np.asarray(dynamic_core.robot_morph, dtype=np.int32)
    actuator_mask, deform_mask = _world_masks(
        robot_morph,
        grid_h=terrain_assets.grid_h,
        grid_w=terrain_assets.grid_w,
        spawn_x=dynamic_core.spawn_x,
        spawn_y=dynamic_core.spawn_y,
    )
    per_axis_info = _world_aligned_per_axis_info(
        dynamic_core.per_axis_info,
        robot_morph,
        grid_h=terrain_assets.grid_h,
        grid_w=terrain_assets.grid_w,
        spawn_x=dynamic_core.spawn_x,
        spawn_y=dynamic_core.spawn_y,
    )
    deformation_info = _world_aligned_deformation_info(
        dynamic_core.deformation_info,
        robot_morph,
        grid_h=terrain_assets.grid_h,
        grid_w=terrain_assets.grid_w,
        spawn_x=dynamic_core.spawn_x,
        spawn_y=dynamic_core.spawn_y,
    )
    static_collision_worklist = build_static_collision_worklist(
        dynamic_core.collision_data,
        terrain_assets.static_collider_template,
        max_surface_boxels=int(dynamic_core.robot_h) * int(dynamic_core.robot_w),
    )
    return CliffEnvData(
        sim_state=dynamic_core.sim_state,
        topology=dynamic_core.topology,
        constants=dynamic_core.constants,
        actuator_info=dynamic_core.actuator_info,
        per_axis_info=per_axis_info,
        collision_data=dynamic_core.collision_data,
        static_collider_data=terrain_assets.static_collider_template,
        static_collision_worklist=static_collision_worklist,
        deformation_info=deformation_info,
        actuator_mask=actuator_mask,
        deform_mask=deform_mask,
        robot_point_mask=dynamic_core.robot_point_mask,
        robot_point_count=dynamic_core.robot_point_count,
        grid_h=terrain_assets.grid_h,
        grid_w=terrain_assets.grid_w,
        spawn_x=dynamic_core.spawn_x,
        spawn_y=dynamic_core.spawn_y,
        robot_h=dynamic_core.robot_h,
        robot_w=dynamic_core.robot_w,
        death_plane_y=terrain_assets.death_plane_y,
        init_com_y=dynamic_core.init_com_y,
        init_com_x=dynamic_core.init_com_x,
        init_min_x=dynamic_core.init_min_x,
        init_front_x=dynamic_core.init_front_x,
        cliff_x=terrain_assets.cliff_x,
        task_mode=terrain_assets.task_mode,
        goal_direction=terrain_assets.goal_direction,
        cliff_side=terrain_assets.cliff_side,
        goal_distance=terrain_assets.goal_distance,
        cliff_distance=terrain_assets.cliff_distance,
        goal_block_x=terrain_assets.goal_block_x,
        goal_block_y=terrain_assets.goal_block_y,
        goal_x=terrain_assets.goal_x,
        goal_object_id=terrain_assets.goal_object_id,
        cliff_probe_ledge_width=terrain_assets.cliff_probe_ledge_width,
        cliff_probe_ledge_cell_x=terrain_assets.cliff_probe_ledge_cell_x,
        cliff_probe_ledge_cell_y=terrain_assets.cliff_probe_ledge_cell_y,
        render_info=merge_render_info(
            dynamic_core.render_info,
            terrain_assets.render_info,
            n_dynamic_points=int(dynamic_core.sim_state.positions.shape[0]),
            grid_h=terrain_assets.grid_h,
            grid_w=terrain_assets.grid_w,
        ),
    )


def build_cliff_env(
    terrain: np.ndarray,
    robot_morph: np.ndarray,
    spawn_x: int,
    spawn_y: int,
    cliff_x: int,
    *,
    task_metadata: dict[str, object] | None = None,
    physics_substeps: int = REFERENCE_SUBSTEPS,
    actuator_convergence: float = 0.003,
    spring_stiffness_scale: float = 0.8,
    floor_static_friction_const: float = 0.0,
    floor_stiction_stiffness: float = 0.0,
    floor_stiction_switch_speed: float = 0.0,
    collision_build_capacity: CollisionBuildCapacity | None = None,
) -> CliffEnvData:
    """Build a cliff-walk environment from terrain and robot arrays."""
    dynamic_core = build_cliff_dynamic_core(
        robot_morph,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        physics_substeps=physics_substeps,
        actuator_convergence=actuator_convergence,
        spring_stiffness_scale=spring_stiffness_scale,
        floor_static_friction_const=floor_static_friction_const,
        floor_stiction_stiffness=floor_stiction_stiffness,
        floor_stiction_switch_speed=floor_stiction_switch_speed,
        collision_build_capacity=collision_build_capacity,
    )
    terrain_assets = build_cliff_terrain_assets(
        terrain,
        robot_morph,
        spawn_x,
        spawn_y,
        cliff_x,
        task_metadata=task_metadata,
        spring_stiffness_scale=spring_stiffness_scale,
        collision_build_capacity=collision_build_capacity,
    )
    return assemble_cliff_env(dynamic_core, terrain_assets)


def precompute_shared_grid_data(grid_h: int, grid_w: int) -> GridData:
    """Precompute grid infrastructure once for grown-mode pure-JAX env builds."""
    return precompute_grid(int(grid_h), int(grid_w))


def build_static_collider_template(
    terrain: np.ndarray,
    *,
    cliff_x: int,
    task_metadata: dict[str, object] | None = None,
    canvas_h: int,
    canvas_w: int,
    spawn_x: int,
    spawn_y: int,
    spring_stiffness_scale: float = 0.8,
    collision_build_capacity: CollisionBuildCapacity | None = None,
) -> CliffStaticTemplate:
    """Build a terrain-only static collider template using an empty growth canvas."""
    empty_canvas = np.full((int(canvas_h), int(canvas_w)), EMPTY, dtype=np.int32)
    assets = build_cliff_terrain_assets(
        np.asarray(terrain, dtype=np.int32),
        empty_canvas,
        int(spawn_x),
        int(spawn_y),
        int(cliff_x),
        task_metadata=task_metadata,
        spring_stiffness_scale=spring_stiffness_scale,
        collision_build_capacity=collision_build_capacity,
    )
    return CliffStaticTemplate(
        static_collider_data=assets.static_collider_template,
        cliff_x=int(cliff_x),
        death_plane_y=float(assets.death_plane_y),
        task_mode=int(assets.task_mode),
        goal_direction=int(assets.goal_direction),
        cliff_side=int(assets.cliff_side),
        goal_distance=int(assets.goal_distance),
        cliff_distance=int(assets.cliff_distance),
        goal_block_x=int(assets.goal_block_x),
        goal_block_y=int(assets.goal_block_y),
        goal_x=float(assets.goal_x),
        goal_object_id=int(assets.goal_object_id),
        cliff_probe_ledge_width=int(assets.cliff_probe_ledge_width),
        cliff_probe_ledge_cell_x=int(assets.cliff_probe_ledge_cell_x),
        cliff_probe_ledge_cell_y=int(assets.cliff_probe_ledge_cell_y),
        grid_h=int(assets.grid_h),
        grid_w=int(assets.grid_w),
        render_info=assets.render_info,
    )


def _build_env_data_from_grids(
    *,
    robot_morph: jnp.ndarray,
    composited: jnp.ndarray,
    robot_mask: jnp.ndarray,
    collision_morphology: jnp.ndarray,
    grid_data: GridData,
    spawn_x: int,
    spawn_y: int,
    cliff_x: int,
    task_mode: int,
    goal_direction: int,
    cliff_side: int,
    goal_distance: int,
    cliff_distance: int,
    goal_block_x: int,
    goal_block_y: int,
    goal_x: float | jnp.ndarray,
    goal_object_id: int,
    cliff_probe_ledge_width: int,
    cliff_probe_ledge_cell_x: int,
    cliff_probe_ledge_cell_y: int,
    death_plane_y: float | jnp.ndarray,
    physics_constants: PhysicsConstants | None,
    max_triples: int,
    max_self_triples: int,
    static_collider_data: StaticColliderData,
    static_collision_worklist: StaticCollisionWorklist | None = None,
    max_surface_boxels: int | None = None,
    max_surface_columns: int | None = None,
    spring_stiffness_scale: float = 1.0,
) -> CliffEvalEnvData:
    """Pure-JAX env assembly for candidate-specific grown morphologies."""
    del max_surface_columns
    robot_h, robot_w = robot_morph.shape
    local_cell_count = int(robot_h * robot_w)
    sim_state, topology = jax_build_sim_state(
        composited,
        grid_data,
        spring_stiffness_scale=spring_stiffness_scale,
    )
    actuator_info, _ = jax_build_actuator_info(composited, grid_data)
    per_axis_info, actuator_mask = jax_build_per_axis_actuator_info(composited, grid_data)
    collision_data = jax_build_collision_data(
        collision_morphology,
        robot_mask,
        grid_data,
        max_triples=max_triples,
        max_self_triples=max_self_triples,
        max_active_boxels=max_surface_boxels,
    )
    deformation_info, deform_mask = jax_build_deformation_info(composited, grid_data, robot_mask)
    worklist = (
        build_static_collision_worklist(
            collision_data,
            static_collider_data,
            max_surface_boxels=max_surface_boxels,
        )
        if static_collision_worklist is None
        else static_collision_worklist
    )

    robot_row_offsets = (jnp.int32(spawn_y) + jnp.arange(robot_h, dtype=jnp.int32))[:, None] * jnp.int32(grid_data.W)
    robot_col_offsets = (jnp.int32(spawn_x) + jnp.arange(robot_w, dtype=jnp.int32))[None, :]
    robot_world_slots = (robot_row_offsets + robot_col_offsets).reshape(-1)
    robot_flat = robot_morph.reshape(-1)
    h_local_enabled = jnp.logical_or(
        robot_flat == jnp.int32(C.H_ACT),
        robot_flat == jnp.int32(C.CONTRACTILE),
    )
    v_local_enabled = jnp.logical_or(
        robot_flat == jnp.int32(C.V_ACT),
        robot_flat == jnp.int32(C.CONTRACTILE),
    )
    h_local_indices = jnp.nonzero(h_local_enabled, size=local_cell_count, fill_value=0)[0].astype(jnp.int32)
    v_local_indices = jnp.nonzero(v_local_enabled, size=local_cell_count, fill_value=0)[0].astype(jnp.int32)
    per_axis_info = per_axis_info._replace(
        h_compact_cell_indices=robot_world_slots[h_local_indices],
        v_compact_cell_indices=robot_world_slots[v_local_indices],
        h_compact_spring_pairs=per_axis_info.h_spring_pairs[robot_world_slots[h_local_indices]],
        v_compact_spring_pairs=per_axis_info.v_spring_pairs[robot_world_slots[v_local_indices]],
        h_compact_count=jnp.sum(h_local_enabled.astype(jnp.int32)),
        v_compact_count=jnp.sum(v_local_enabled.astype(jnp.int32)),
    )

    n_points = (grid_data.H + 1) * (grid_data.W + 1)
    robot_flat_mask = robot_mask.reshape(-1)
    morph_flat = composited.reshape(-1)
    is_robot_cell = (morph_flat != EMPTY) & robot_flat_mask
    robot_point_mask = jnp.zeros((n_points,), dtype=jnp.bool_)
    for corner_idx in range(4):
        robot_point_mask = robot_point_mask.at[grid_data.cell_corner_indices[:, corner_idx]].max(
            is_robot_cell
        )
    robot_point_count = jnp.sum(robot_point_mask.astype(jnp.int32))

    constants = default_physics_constants() if physics_constants is None else physics_constants
    init_front_x = jnp.max(jnp.where(robot_point_mask, sim_state.positions[:, 0], jnp.float32(-1e10)))
    init_min_x = jnp.min(jnp.where(robot_point_mask, sim_state.positions[:, 0], jnp.float32(1e10)))
    robot_point_count_f = jnp.maximum(robot_point_count.astype(jnp.float32), 1.0)
    init_com_x = jnp.sum(
        jnp.where(robot_point_mask, sim_state.positions[:, 0], jnp.float32(0.0))
    ) / robot_point_count_f
    init_com_y = jnp.sum(
        jnp.where(robot_point_mask, sim_state.positions[:, 1], jnp.float32(0.0))
    ) / robot_point_count_f
    return CliffEvalEnvData(
        sim_state=sim_state,
        topology=topology,
        constants=constants,
        actuator_info=actuator_info,
        per_axis_info=per_axis_info,
        collision_data=collision_data,
        static_collider_data=static_collider_data,
        static_collision_worklist=worklist,
        deformation_info=deformation_info,
        actuator_mask=actuator_mask,
        deform_mask=deform_mask,
        robot_point_mask=robot_point_mask,
        robot_point_count=robot_point_count,
        grid_h=jnp.int32(grid_data.H),
        grid_w=jnp.int32(grid_data.W),
        spawn_x=jnp.int32(spawn_x),
        spawn_y=jnp.int32(spawn_y),
        robot_h=jnp.int32(robot_h),
        robot_w=jnp.int32(robot_w),
        death_plane_y=jnp.asarray(death_plane_y, dtype=jnp.float32),
        init_com_y=jnp.asarray(init_com_y, dtype=jnp.float32),
        init_com_x=jnp.asarray(init_com_x, dtype=jnp.float32),
        init_min_x=jnp.asarray(init_min_x, dtype=jnp.float32),
        init_front_x=jnp.asarray(init_front_x, dtype=jnp.float32),
        cliff_x=jnp.int32(cliff_x),
        task_mode=jnp.int32(task_mode),
        goal_direction=jnp.int32(goal_direction),
        cliff_side=jnp.int32(cliff_side),
        goal_distance=jnp.int32(goal_distance),
        cliff_distance=jnp.int32(cliff_distance),
        goal_block_x=jnp.int32(goal_block_x),
        goal_block_y=jnp.int32(goal_block_y),
        goal_x=jnp.asarray(goal_x, dtype=jnp.float32),
        goal_object_id=jnp.int32(goal_object_id),
        cliff_probe_ledge_width=jnp.int32(cliff_probe_ledge_width),
        cliff_probe_ledge_cell_x=jnp.int32(cliff_probe_ledge_cell_x),
        cliff_probe_ledge_cell_y=jnp.int32(cliff_probe_ledge_cell_y),
    )


def build_cliff_eval_dynamic_core(
    robot_morph: jnp.ndarray,
    grid_data: GridData,
    spawn_x: int,
    spawn_y: int,
    *,
    cliff_x: int = 0,
    task_mode: int = _TASK_MODE_CODE_LEGACY,
    goal_direction: int = _SIDE_CODE_NONE,
    cliff_side: int = _SIDE_CODE_RIGHT,
    goal_distance: int = 0,
    cliff_distance: int = 0,
    goal_block_x: int = -1,
    goal_block_y: int = -1,
    goal_x: float = -1.0,
    goal_object_id: int = 0,
    cliff_probe_ledge_width: int = 0,
    cliff_probe_ledge_cell_x: int = -1,
    cliff_probe_ledge_cell_y: int = -1,
    physics_substeps: int = REFERENCE_SUBSTEPS,
    actuator_convergence: float = 0.003,
    spring_stiffness_scale: float = 0.8,
    floor_static_friction_const: float = 0.0,
    floor_stiction_stiffness: float = 0.0,
    floor_stiction_switch_speed: float = 0.0,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
) -> CliffEvalEnvData:
    """Build the terrain-independent dynamic env payload for one grown candidate."""
    empty_world = jnp.zeros((grid_data.H, grid_data.W), dtype=jnp.int32)
    robot_only_world, robot_mask = jax_compose_grid(empty_world, robot_morph, spawn_y, spawn_x)
    dt, effective_convergence = derive_physics_params(physics_substeps, actuator_convergence)
    physics_constants = default_physics_constants()._replace(
        dt=dt,
        actuator_convergence=effective_convergence,
        floor_static_friction_const=float(floor_static_friction_const),
        floor_stiction_stiffness=float(floor_stiction_stiffness),
        floor_stiction_switch_speed=float(floor_stiction_switch_speed),
        diagnostics_enabled=False,
    )
    robot_h, robot_w = robot_morph.shape
    return _build_env_data_from_grids(
        robot_morph=robot_morph,
        composited=robot_only_world,
        robot_mask=robot_mask,
        collision_morphology=robot_only_world,
        grid_data=grid_data,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        cliff_x=cliff_x,
        task_mode=task_mode,
        goal_direction=goal_direction,
        cliff_side=cliff_side,
        goal_distance=goal_distance,
        cliff_distance=cliff_distance,
        goal_block_x=goal_block_x,
        goal_block_y=goal_block_y,
        goal_x=goal_x,
        goal_object_id=goal_object_id,
        cliff_probe_ledge_width=cliff_probe_ledge_width,
        cliff_probe_ledge_cell_x=cliff_probe_ledge_cell_x,
        cliff_probe_ledge_cell_y=cliff_probe_ledge_cell_y,
        death_plane_y=jnp.float32(CELL_SIZE),
        physics_constants=physics_constants,
        max_triples=int(max_triples),
        max_self_triples=int(max_self_triples),
        static_collider_data=_empty_static_collider_data(),
        max_surface_boxels=int(robot_h * robot_w),
        spring_stiffness_scale=spring_stiffness_scale,
    )


def attach_static_template_to_eval_env(
    env_core: CliffEvalEnvData,
    static_collider_template: StaticColliderData,
    *,
    cliff_x: int,
    task_mode: int = _TASK_MODE_CODE_LEGACY,
    goal_direction: int = _SIDE_CODE_NONE,
    cliff_side: int = _SIDE_CODE_RIGHT,
    goal_distance: int = 0,
    cliff_distance: int = 0,
    goal_block_x: int = -1,
    goal_block_y: int = -1,
    goal_x: float | jnp.ndarray = -1.0,
    goal_object_id: int = 0,
    cliff_probe_ledge_width: int = 0,
    cliff_probe_ledge_cell_x: int = -1,
    cliff_probe_ledge_cell_y: int = -1,
    death_plane_y: float | jnp.ndarray,
    max_surface_boxels: int | None = None,
    max_surface_columns: int | None = None,
) -> CliffEvalEnvData:
    """Attach terrain-specific static colliders to a grown candidate's dynamic core."""
    del max_surface_columns
    if max_surface_boxels is None:
        max_surface_boxels = int(np.asarray(env_core.robot_h)) * int(np.asarray(env_core.robot_w))
    static_collision_worklist = build_static_collision_worklist(
        env_core.collision_data,
        static_collider_template,
        max_surface_boxels=max_surface_boxels,
    )
    return env_core._replace(
        static_collider_data=static_collider_template,
        static_collision_worklist=static_collision_worklist,
        cliff_x=jnp.int32(cliff_x),
        task_mode=jnp.int32(task_mode),
        goal_direction=jnp.int32(goal_direction),
        cliff_side=jnp.int32(cliff_side),
        goal_distance=jnp.int32(goal_distance),
        cliff_distance=jnp.int32(cliff_distance),
        goal_block_x=jnp.int32(goal_block_x),
        goal_block_y=jnp.int32(goal_block_y),
        goal_x=jnp.asarray(goal_x, dtype=jnp.float32),
        goal_object_id=jnp.int32(goal_object_id),
        cliff_probe_ledge_width=jnp.int32(cliff_probe_ledge_width),
        cliff_probe_ledge_cell_x=jnp.int32(cliff_probe_ledge_cell_x),
        cliff_probe_ledge_cell_y=jnp.int32(cliff_probe_ledge_cell_y),
        death_plane_y=jnp.asarray(death_plane_y, dtype=jnp.float32),
    )


def make_cliff_env_split_colliders(
    robot_morph: jnp.ndarray,
    grid_data: GridData,
    spawn_x: int,
    spawn_y: int,
    *,
    cliff_x: int,
    static_collider_template: StaticColliderData,
    task_mode: int = _TASK_MODE_CODE_LEGACY,
    goal_direction: int = _SIDE_CODE_NONE,
    cliff_side: int = _SIDE_CODE_RIGHT,
    goal_distance: int = 0,
    cliff_distance: int = 0,
    goal_block_x: int = -1,
    goal_block_y: int = -1,
    goal_x: float = -1.0,
    goal_object_id: int = 0,
    cliff_probe_ledge_width: int = 0,
    cliff_probe_ledge_cell_x: int = -1,
    cliff_probe_ledge_cell_y: int = -1,
    physics_substeps: int = REFERENCE_SUBSTEPS,
    actuator_convergence: float = 0.003,
    death_plane_y: float = CELL_SIZE,
    spring_stiffness_scale: float = 0.8,
    floor_static_friction_const: float = 0.0,
    floor_stiction_stiffness: float = 0.0,
    floor_stiction_switch_speed: float = 0.0,
    max_triples: int = C.MAX_TRIPLES,
    max_self_triples: int = C.MAX_SELF_TRIPLES,
) -> CliffEvalEnvData:
    """Pure-JAX grown-body env builder using split dynamic/static colliders."""
    env_core = build_cliff_eval_dynamic_core(
        robot_morph,
        grid_data,
        spawn_x,
        spawn_y,
        cliff_x=cliff_x,
        task_mode=task_mode,
        goal_direction=goal_direction,
        cliff_side=cliff_side,
        goal_distance=goal_distance,
        cliff_distance=cliff_distance,
        goal_block_x=goal_block_x,
        goal_block_y=goal_block_y,
        goal_x=goal_x,
        goal_object_id=goal_object_id,
        cliff_probe_ledge_width=cliff_probe_ledge_width,
        cliff_probe_ledge_cell_x=cliff_probe_ledge_cell_x,
        cliff_probe_ledge_cell_y=cliff_probe_ledge_cell_y,
        physics_substeps=physics_substeps,
        actuator_convergence=actuator_convergence,
        spring_stiffness_scale=spring_stiffness_scale,
        floor_static_friction_const=floor_static_friction_const,
        floor_stiction_stiffness=floor_stiction_stiffness,
        floor_stiction_switch_speed=floor_stiction_switch_speed,
        max_triples=max_triples,
        max_self_triples=max_self_triples,
    )
    return attach_static_template_to_eval_env(
        env_core,
        static_collider_template,
        cliff_x=cliff_x,
        task_mode=task_mode,
        goal_direction=goal_direction,
        cliff_side=cliff_side,
        goal_distance=goal_distance,
        cliff_distance=cliff_distance,
        goal_block_x=goal_block_x,
        goal_block_y=goal_block_y,
        goal_x=goal_x,
        goal_object_id=goal_object_id,
        cliff_probe_ledge_width=cliff_probe_ledge_width,
        cliff_probe_ledge_cell_x=cliff_probe_ledge_cell_x,
        cliff_probe_ledge_cell_y=cliff_probe_ledge_cell_y,
        death_plane_y=death_plane_y,
        max_surface_boxels=int(robot_morph.shape[0] * robot_morph.shape[1]),
    )


def derive_physics_params(
    substeps: int,
    reference_actuator_convergence: float,
) -> tuple[float, float]:
    """Derive dt and effective convergence from a 30-substep reference setting."""
    if substeps <= 0:
        raise ValueError(f"physics_substeps must be positive, got {substeps}")
    if not 0.0 <= reference_actuator_convergence < 1.0:
        raise ValueError(
            "actuator_convergence must be in [0.0, 1.0) as the 30-substep reference value"
        )

    dt = REFERENCE_DT * (REFERENCE_SUBSTEPS / float(substeps))
    per_env_step_convergence = 1.0 - (
        1.0 - float(reference_actuator_convergence)
    ) ** REFERENCE_SUBSTEPS
    effective_convergence = 1.0 - (1.0 - per_env_step_convergence) ** (1.0 / float(substeps))
    return dt, effective_convergence


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def reset_episode(
    env_data: CliffEnvData | CliffEvalEnvData,
    key: jax.Array,
    ctrnn_hidden: int,
    ctrnn_grid_hidden: int,
    robot_h: int | None = None,
    robot_w: int | None = None,
    gasnet_mode: str = "off",
    ou_enabled: bool = True,
    mace_mode: str = "off",
    mace_init_mass: float = 1.0,
    growth_enabled: bool = False,
    fixed_hidden_init_scale: float = 0.0,
    frozen_identity: jnp.ndarray | None = None,
    initial_nca_state: jnp.ndarray | None = None,
    initial_ctrnn_carry: jnp.ndarray | None = None,
) -> CliffEpisodeState:
    """Initialize episode state."""
    static_robot_h = int(robot_h if robot_h is not None else env_data.robot_h)
    static_robot_w = int(robot_w if robot_w is not None else env_data.robot_w)
    gc = grid_channel_count(
        ctrnn_grid_hidden,
        gasnet_mode,
        ou_enabled,
        growth_enabled=growth_enabled,
    )
    if initial_nca_state is None:
        nca_state = jnp.zeros((static_robot_h, static_robot_w, gc), dtype=jnp.float32)
        if (
            (not bool(growth_enabled))
            and int(ctrnn_grid_hidden) > 0
            and float(fixed_hidden_init_scale) > 0.0
        ):
            gh_start = base_channel_count(ou_enabled=ou_enabled)
            key, key_hidden = jax.random.split(key)
            hidden_noise = float(fixed_hidden_init_scale) * jax.random.uniform(
                key_hidden,
                (static_robot_h, static_robot_w, ctrnn_grid_hidden),
                minval=-1.0,
                maxval=1.0,
            )
            nca_state = nca_state.at[:, :, gh_start:gh_start + ctrnn_grid_hidden].set(
                hidden_noise
            )
    else:
        nca_state = initial_nca_state
    if initial_nca_state is None and mace_mode != "off" and ctrnn_grid_hidden > 0:
        gh_start = base_channel_count(ou_enabled=ou_enabled)
        nca_state = nca_state.at[:, :, gh_start:gh_start + ctrnn_grid_hidden].set(
            jnp.float32(mace_init_mass)
        )
    if initial_ctrnn_carry is None:
        ctrnn_carry = jnp.zeros(
            (static_robot_h, static_robot_w, ctrnn_hidden), dtype=jnp.float32
        )
    else:
        ctrnn_carry = initial_ctrnn_carry
    if bool(growth_enabled) and frozen_identity is not None:
        nca_state = clamp_growth_channels(
            nca_state,
            frozen_identity,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
        )
    return CliffEpisodeState(
        sim_state=env_data.sim_state,
        nca_state=nca_state,
        ctrnn_carry=ctrnn_carry,
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
        peak_front_x=jnp.float32(env_data.init_front_x),
        safe_front_x=jnp.float32(env_data.init_front_x),
        touched_death_plane=jnp.bool_(False),
        went_airborne=jnp.bool_(False),
        com_x_went_negative=jnp.bool_(False),
        cumulative_activation=jnp.float32(0.0),
        cumulative_h_activation=jnp.float32(0.0),
        cumulative_v_activation=jnp.float32(0.0),
        cumulative_speed=jnp.float32(0.0),
        cumulative_lower_ledge_exposure_steps=jnp.int32(0),
        cumulative_cliff_vicinity_steps=jnp.int32(0),
        cumulative_lofted_steps=jnp.int32(0),
        cumulative_soft_loft_excess_cells=jnp.float32(0.0),
        first_cliff_vicinity_step=jnp.int32(-1),
        peak_any_direction_progress=jnp.float32(0.0),
        peak_goal_progress=jnp.float32(0.0),
        peak_incorrect_progress=jnp.float32(0.0),
        peak_toward_goal_frontier=jnp.float32(0.0),
        goal_touched=jnp.bool_(False),
        cliff_fell=jnp.bool_(False),
        first_goal_touch_step=jnp.int32(-1),
        last_significant_direction=jnp.int32(0),
        turn_count=jnp.int32(0),
        stable_turn_direction=jnp.int32(0),
        stable_turn_distance=jnp.float32(0.0),
        stable_turn_eligible=jnp.bool_(False),
        stable_turn_count=jnp.int32(0),
        steps_alive=jnp.int32(0),
        deformation_channel_max=zero_deformation_channel_max(),
        controller_hidden_output_max_abs=jnp.float32(0.0),
        action_output_max_abs=jnp.float32(0.0),
        peak_com_y_delta_cells=jnp.float32(0.0),
        first_significant_direction=jnp.int32(0),
    )


# ---------------------------------------------------------------------------
# Sensing helpers
# ---------------------------------------------------------------------------


def _compute_dual_deformation(
    sim_state: SimState,
    env_data: CliffEnvData | CliffEvalEnvData,
) -> jnp.ndarray:
    """Compute dual deformation: (n_robot_voxels, 4) [abs_h, abs_v, comp_h, comp_v]."""
    return compute_dual_deformation(
        sim_state.positions,
        env_data.topology,
        sim_state.spring_init_rest_length,
        env_data.deformation_info,
        sim_state.spring_rest_length,
    )


def _deformation_variant_forces_zero_channels(variant: str) -> bool:
    """Whether a post-run deformation variant should hard-zero controller inputs."""
    normalized_variant = str(variant).strip().lower()
    return normalized_variant in {"zero", "deformation_off"}


def _sample_deformation_variant_noise(
    *,
    signed_delta_shape: tuple[int, ...],
    noise_sigma_signed_delta: float,
    variant_phase_key: jax.Array,
) -> jnp.ndarray:
    sigma = float(noise_sigma_signed_delta)
    if sigma <= 0.0:
        return jnp.zeros(signed_delta_shape, dtype=jnp.float32)
    return (
        jax.random.normal(
            variant_phase_key,
            shape=signed_delta_shape,
            dtype=jnp.float32,
        )
        * jnp.float32(sigma)
    )


def _validate_forced_zero_deformation_channels(
    deformation: jnp.ndarray,
    *,
    variant: str,
    atol: float = 1e-6,
) -> None:
    """Fail fast if a forced-zero deformation variant leaves any signed signal."""
    if not _deformation_variant_forces_zero_channels(variant):
        return
    signed_delta = jnp.asarray(deformation, dtype=jnp.float32) - jnp.float32(1.0)
    channel_max_abs = jnp.max(jnp.abs(signed_delta), axis=0)

    def _raise_if_nonzero(max_abs_by_channel: np.ndarray) -> None:
        max_abs_by_channel = np.asarray(max_abs_by_channel, dtype=np.float32)
        if np.any(max_abs_by_channel > float(atol)):
            raise ValueError(
                "Forced-zero deformation variant "
                f"{variant!r} produced non-zero signed deformation channels "
                f"(abs_h={max_abs_by_channel[0]:.6g}, "
                f"abs_v={max_abs_by_channel[1]:.6g}, "
                f"comp_h={max_abs_by_channel[2]:.6g}, "
                f"comp_v={max_abs_by_channel[3]:.6g})"
            )

    jax.debug.callback(_raise_if_nonzero, channel_max_abs)


def _apply_deformation_variant(
    deformation: jnp.ndarray,
    *,
    variant: str,
    noise_sigma_signed_delta: float,
    shuffle_mode: str,
    variant_phase_key: jax.Array,
    variant_episode_key: jax.Array,
) -> jnp.ndarray:
    """Apply a deterministic per-episode deformation perturbation variant."""
    deformation = jnp.asarray(deformation, dtype=jnp.float32)
    signed_delta = deformation - jnp.float32(1.0)
    normalized_variant = str(variant).strip().lower()
    if normalized_variant in {"clean", "intact"}:
        return deformation
    if _deformation_variant_forces_zero_channels(normalized_variant):
        return jnp.ones_like(deformation, dtype=jnp.float32)
    if normalized_variant == "absolute_only":
        kept = signed_delta.at[:, 2:].set(0.0)
        return kept + jnp.float32(1.0)
    if normalized_variant == "compliance_only":
        kept = signed_delta.at[:, :2].set(0.0)
        return kept + jnp.float32(1.0)
    if normalized_variant == "deformation_reversed":
        return jnp.float32(1.0) - signed_delta
    if normalized_variant == "noisy":
        noise = _sample_deformation_variant_noise(
            signed_delta_shape=signed_delta.shape,
            noise_sigma_signed_delta=noise_sigma_signed_delta,
            variant_phase_key=variant_phase_key,
        )
        if float(noise_sigma_signed_delta) <= 0.0:
            return deformation
        return signed_delta + noise + jnp.float32(1.0)
    if normalized_variant == "noise_only":
        noise = _sample_deformation_variant_noise(
            signed_delta_shape=signed_delta.shape,
            noise_sigma_signed_delta=noise_sigma_signed_delta,
            variant_phase_key=variant_phase_key,
        )
        return noise + jnp.float32(1.0)
    if normalized_variant == "shuffled":
        if shuffle_mode != "episode_fixed":
            raise ValueError(
                "Only deformation_shuffle_mode='episode_fixed' is supported."
            )
        permutation = jax.random.permutation(
            variant_episode_key,
            signed_delta.shape[0],
        )
        return signed_delta[permutation] + jnp.float32(1.0)
    raise ValueError(f"Unsupported deformation_variant={variant!r}")


class RobotKinematics(NamedTuple):
    """Robot aggregate kinematics derived from point positions."""

    min_x: jnp.ndarray
    front_x: jnp.ndarray
    min_y: jnp.ndarray
    max_y: jnp.ndarray
    com_x: jnp.ndarray
    com_y: jnp.ndarray


def _robot_kinematics(
    sim_state: SimState,
    robot_point_mask: jnp.ndarray,
) -> RobotKinematics:
    """Compute the robot aggregates needed for termination and fitness bookkeeping."""
    x = sim_state.positions[:, 0]
    y = sim_state.positions[:, 1]
    # Avoid bool->numeric casts in per-step runtime math.
    count = jnp.maximum(jnp.sum(jnp.where(robot_point_mask, 1.0, 0.0)), 1.0)
    return RobotKinematics(
        min_x=jnp.min(jnp.where(robot_point_mask, x, 1e10)),
        front_x=jnp.max(jnp.where(robot_point_mask, x, -1e10)),
        min_y=jnp.min(jnp.where(robot_point_mask, y, 1e10)),
        max_y=jnp.max(jnp.where(robot_point_mask, y, -1e10)),
        com_x=jnp.sum(jnp.where(robot_point_mask, x, 0.0)) / count,
        com_y=jnp.sum(jnp.where(robot_point_mask, y, 0.0)) / count,
    )


def _lower_ledge_exposure_now(
    robot_stats: RobotKinematics,
    *,
    cliff_probe_ledge_width: int | jnp.ndarray,
    cliff_probe_ledge_cell_x: int | jnp.ndarray,
    cliff_probe_ledge_cell_y: int | jnp.ndarray,
) -> jnp.ndarray:
    """Approximate lowered-ledge contact using robot envelope overlap and depth."""
    ledge_width = jnp.asarray(cliff_probe_ledge_width, dtype=jnp.int32)
    ledge_cell_x = jnp.asarray(cliff_probe_ledge_cell_x, dtype=jnp.int32)
    ledge_cell_y = jnp.asarray(cliff_probe_ledge_cell_y, dtype=jnp.int32)
    has_ledge = (ledge_width > 0) & (ledge_cell_x >= 0) & (ledge_cell_y >= 0)
    ledge_min_x = ledge_cell_x.astype(jnp.float32) * jnp.float32(CELL_SIZE)
    ledge_max_x = (
        (ledge_cell_x + ledge_width).astype(jnp.float32) * jnp.float32(CELL_SIZE)
    )
    ledge_top_y = (
        (ledge_cell_y.astype(jnp.float32) + jnp.float32(1.0)) * jnp.float32(CELL_SIZE)
    )
    overlaps_x = (
        (robot_stats.front_x >= (ledge_min_x - jnp.float32(1e-5)))
        & (robot_stats.min_x <= (ledge_max_x + jnp.float32(1e-5)))
    )
    below_ledge_top = robot_stats.min_y < ledge_top_y
    return has_ledge & overlaps_x & below_ledge_top


def _robot_com_x(sim_state: SimState, robot_point_mask: jnp.ndarray) -> jnp.ndarray:
    """Robot center of mass x-coordinate."""
    x = sim_state.positions[:, 0]
    masked = jnp.where(robot_point_mask, x, 0.0)
    n = jnp.maximum(jnp.sum(jnp.where(robot_point_mask, 1.0, 0.0)), 1.0)
    return jnp.sum(masked) / n


def _phase_activation_from_compact_actions(
    h_action_compact: jnp.ndarray,
    v_action_compact: jnp.ndarray,
    h_valid: jnp.ndarray,
    v_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert applied compact actuator goals back into contraction-equivalent activity."""

    h_valid_f = jnp.where(h_valid, 1.0, 0.0)
    v_valid_f = jnp.where(v_valid, 1.0, 0.0)
    total_axis_count = jnp.maximum(jnp.sum(h_valid_f) + jnp.sum(v_valid_f), 1.0)
    h_contraction_mag = 2.0 * jnp.abs(h_action_compact - 1.0)
    v_contraction_mag = 2.0 * jnp.abs(v_action_compact - 1.0)
    h_activation = jnp.sum(h_contraction_mag * h_valid_f) / total_axis_count
    v_activation = jnp.sum(v_contraction_mag * v_valid_f) / total_axis_count
    activation = h_activation + v_activation
    return activation, h_activation, v_activation


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _episode_result_from_state(
    final_state: CliffEpisodeState,
    env_data: CliffEnvData | CliffEvalEnvData,
    *,
    max_steps: int,
) -> EpisodeResult:
    total_phases = jnp.maximum(final_state.steps_alive.astype(jnp.float32), 1.0)
    total_env_steps = jnp.maximum(final_state.step_count.astype(jnp.float32), 1.0)
    max_steps_f = jnp.float32(max(max_steps, 1))
    mean_activation = final_state.cumulative_activation / total_phases
    mean_h_activation = final_state.cumulative_h_activation / total_phases
    mean_v_activation = final_state.cumulative_v_activation / total_phases
    mean_speed = final_state.cumulative_speed / total_env_steps
    turn_motion_threshold = _episode_turn_motion_threshold(
        init_min_x=jnp.asarray(env_data.init_min_x, dtype=jnp.float32),
        init_front_x=jnp.asarray(env_data.init_front_x, dtype=jnp.float32),
    )
    cliff_vicinity_steps = final_state.cumulative_cliff_vicinity_steps
    cliff_vicinity_fraction = cliff_vicinity_steps.astype(jnp.float32) / max_steps_f
    final_robot_stats = _robot_kinematics(final_state.sim_state, env_data.robot_point_mask)
    lower_ledge_exposure_steps = final_state.cumulative_lower_ledge_exposure_steps
    lower_ledge_exposure_fraction = (
        lower_ledge_exposure_steps.astype(jnp.float32) / max_steps_f
    )
    lofted_fraction = (
        final_state.cumulative_lofted_steps.astype(jnp.float32) / total_phases
    )
    mean_lofted_excess_cells = (
        final_state.cumulative_soft_loft_excess_cells / total_phases
    )
    time_to_edge = jnp.where(
        final_state.first_cliff_vicinity_step >= 0,
        final_state.first_cliff_vicinity_step.astype(jnp.float32) / max_steps_f,
        jnp.float32(1.0),
    )
    bilateral_task = jnp.asarray(env_data.task_mode) == jnp.int32(_TASK_MODE_CODE_BILATERAL_GOAL_CLIFF)
    any_progress, goal_progress, incorrect_progress, toward_goal_frontier, reversal_progress = _bilateral_progress_metrics(
        bilateral_task=bilateral_task,
        init_com_x=jnp.asarray(env_data.init_com_x, dtype=jnp.float32),
        init_min_x=jnp.asarray(env_data.init_min_x, dtype=jnp.float32),
        init_front_x=jnp.asarray(env_data.init_front_x, dtype=jnp.float32),
        goal_x=jnp.asarray(env_data.goal_x, dtype=jnp.float32),
        goal_direction=jnp.asarray(env_data.goal_direction, dtype=jnp.int32),
        cliff_x=jnp.asarray(env_data.cliff_x, dtype=jnp.int32),
        cliff_side=jnp.asarray(env_data.cliff_side, dtype=jnp.int32),
        goal_touched=final_state.goal_touched,
        final_min_x=final_robot_stats.min_x,
        final_front_x=final_robot_stats.front_x,
        peak_any_direction_progress=final_state.peak_any_direction_progress,
        peak_goal_progress=final_state.peak_goal_progress,
        peak_incorrect_progress=final_state.peak_incorrect_progress,
    )
    time_to_goal = jnp.where(
        final_state.first_goal_touch_step >= 0,
        final_state.first_goal_touch_step.astype(jnp.float32) / max_steps_f,
        jnp.float32(1.0),
    )
    elapsed_time_fraction = jnp.maximum(
        final_state.steps_alive.astype(jnp.float32) / max_steps_f,
        jnp.float32(1.0) / max_steps_f,
    )
    return EpisodeResult(
        peak_front_x=final_state.peak_front_x,
        safe_front_x=final_state.safe_front_x,
        init_front_x=jnp.float32(env_data.init_front_x),
        init_com_x=jnp.float32(env_data.init_com_x),
        touched_death_plane=final_state.touched_death_plane,
        went_airborne=final_state.went_airborne,
        com_x_went_negative=final_state.com_x_went_negative,
        mean_activation=mean_activation,
        mean_h_activation=mean_h_activation,
        mean_v_activation=mean_v_activation,
        mean_speed=mean_speed,
        cliff_vicinity_steps=cliff_vicinity_steps,
        cliff_vicinity_fraction=cliff_vicinity_fraction,
        time_to_edge=time_to_edge,
        elapsed_time_fraction=elapsed_time_fraction,
        steps_alive=final_state.steps_alive,
        cliff_x=jnp.int32(env_data.cliff_x),
        task_mode=jnp.int32(env_data.task_mode),
        goal_direction=jnp.int32(env_data.goal_direction),
        cliff_side=jnp.int32(env_data.cliff_side),
        goal_distance=jnp.int32(env_data.goal_distance),
        cliff_distance=jnp.int32(env_data.cliff_distance),
        goal_block_x=jnp.int32(env_data.goal_block_x),
        goal_block_y=jnp.int32(env_data.goal_block_y),
        goal_x=jnp.asarray(env_data.goal_x, dtype=jnp.float32),
        any_progress=any_progress,
        goal_progress=goal_progress,
        incorrect_progress=incorrect_progress,
        toward_goal_frontier=toward_goal_frontier,
        goal_touched=final_state.goal_touched,
        cliff_fell=final_state.cliff_fell,
        time_to_goal=time_to_goal,
        turn_count=final_state.turn_count,
        deformation_channel_max=final_state.deformation_channel_max,
        controller_hidden_output_max_abs=final_state.controller_hidden_output_max_abs,
        action_output_max_abs=final_state.action_output_max_abs,
        lower_ledge_exposure_steps=lower_ledge_exposure_steps,
        lower_ledge_exposure_fraction=lower_ledge_exposure_fraction,
        reversal_progress=reversal_progress,
        stable_turn_count=final_state.stable_turn_count,
        turn_motion_threshold=turn_motion_threshold,
        first_significant_direction=final_state.first_significant_direction,
        final_com_x=final_robot_stats.com_x,
        final_front_x=final_robot_stats.front_x,
        final_back_x=final_robot_stats.min_x,
        lofted_fraction=lofted_fraction,
        mean_lofted_excess_cells=mean_lofted_excess_cells,
        peak_com_y_delta_cells=final_state.peak_com_y_delta_cells,
    )


def _run_episode_impl(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    *,
    capture_mode: str = "none",
    max_steps: int = 500,
    physics_substeps: int = 30,
    sense_act_phases: int = 3,
    grid_h: int | None = None,
    grid_w: int | None = None,
    spawn_x: int | None = None,
    spawn_y: int | None = None,
    robot_h: int | None = None,
    robot_w: int | None = None,
    nca_steps_per_phase: int = 1,
    ctrnn_hidden: int = 8,
    ctrnn_grid_hidden: int = 12,
    ctrnn_dt: float = 1.0,
    ctrnn_tau_min: float = 0.1,
    ctrnn_tau_max: float = 10.0,
    p_update: float = 0.5,
    active_comm_filters: tuple[str, ...] = COMM_FILTERS,
    ou_theta: float = 0.1,
    ou_sigma: float = 0.3,
    ou_enabled: bool = True,
    ablate_deformation: bool = False,
    deformation_variant: str = "clean",
    deformation_noise_sigma_signed_delta: float = 0.05,
    deformation_shuffle_mode: str = "episode_fixed",
    signaling_variant: str = "intact",
    sensing_freeze_mode: str = "none",
    gasnet_mode: str = "off",
    gasnet_type: str = "gain",
    gas_decay_rate: float = 0.95,
    gas_diffusion_radius: float = 3.0,
    gas_gain_sensitivity: float = 2.0,
    gas_max_concentration: float = 1.0,
    gas_gain_min: float = -4.0,
    gas_gain_max: float = 4.0,
    update_mode: str = "hybrid",
    state_clip_value: float = 3.0,
    mace_mode: str = "cross_channel",
    mace_beta: float = 1.0,
    mace_init_mass: float = 1.0,
    deformation_gain_absolute: float = 0.0,
    deformation_gain_compliance: float = 0.0,
    airborne_kill_height_margin_cells: float = 2.0,
    soft_loft_threshold_cells: float = 0.0,
    collision_strategy: str = "dynamic_broadphase",
    dynamic_max_candidates: int = 4096,
    dynamic_bin_size: float = 0.2,
    dynamic_impl: str = "sparse_candidate_list",
    dynamic_max_narrow: int | None = None,
    growth_enabled: bool = False,
    fixed_hidden_init_scale: float = 0.0,
    frozen_identity: jnp.ndarray | None = None,
    initial_nca_state: jnp.ndarray | None = None,
    initial_ctrnn_carry: jnp.ndarray | None = None,
    substep_probe_step_index: int | None = None,
    substep_probe_phase_index: int | None = None,
) -> tuple[
    EpisodeResult,
    RenderStepOutput
    | EpisodePhaseProbeTrace
    | EpisodeStateTrace
    | EpisodeComTrace
    | EpisodeControllerTrace
    | EpisodeStateSpaceTrace
    | EpisodeSubstepProbeTrace
    | None,
]:
    """Shared episode core used by traced and untraced rollouts."""
    normalized_sensing_freeze_mode = str(sensing_freeze_mode).strip().lower()
    if normalized_sensing_freeze_mode not in {
        "none",
        "deformation_frozen_start",
        "perception_frozen_start",
    }:
        raise ValueError(
            "sensing_freeze_mode must be one of: none, "
            "deformation_frozen_start, perception_frozen_start"
        )
    if capture_mode not in {
        "none",
        "render",
        "phase_probe",
        "state_trace",
        "com_trace",
        "controller_trace",
        "state_space_trace",
        "substep_probe",
    }:
        raise ValueError(
            "capture_mode must be one of 'none', 'render', 'phase_probe', "
            "'state_trace', 'com_trace', 'controller_trace', "
            "'state_space_trace', or 'substep_probe'"
        )
    if capture_mode == "substep_probe" and (
        substep_probe_step_index is None or substep_probe_phase_index is None
    ):
        raise ValueError(
            "substep_probe_step_index and substep_probe_phase_index are required "
            "when capture_mode='substep_probe'"
        )
    if collision_strategy != "dynamic_broadphase":
        raise ValueError(
            "sensing_edges rollout requires collision_strategy='dynamic_broadphase'"
        )
    substeps_per_phase = physics_substeps // sense_act_phases
    nca_updates_per_env_step = sense_act_phases * nca_steps_per_phase
    ctrnn_dt_per_update = ctrnn_dt / nca_updates_per_env_step
    task_mode = jnp.asarray(env_data.task_mode, dtype=jnp.int32)
    bilateral_task = task_mode == jnp.int32(_TASK_MODE_CODE_BILATERAL_GOAL_CLIFF)
    cliff_vicinity_threshold = (
        (jnp.int32(env_data.cliff_x) + 1).astype(jnp.float32) * CELL_SIZE
        - (3.0 * CELL_SIZE)
    )
    turn_motion_threshold = _episode_turn_motion_threshold(
        init_min_x=jnp.asarray(env_data.init_min_x, dtype=jnp.float32),
        init_front_x=jnp.asarray(env_data.init_front_x, dtype=jnp.float32),
    )
    static_robot_h = int(robot_h if robot_h is not None else alive_mask.shape[0])
    static_robot_w = int(robot_w if robot_w is not None else alive_mask.shape[1])
    static_grid_h = int(grid_h if grid_h is not None else env_data.grid_h)
    static_grid_w = int(grid_w if grid_w is not None else env_data.grid_w)
    static_spawn_x = int(spawn_x if spawn_x is not None else env_data.spawn_x)
    static_spawn_y = int(spawn_y if spawn_y is not None else env_data.spawn_y)
    variant_episode_key = jax.random.fold_in(key, 17)

    key, key_reset = jax.random.split(key)
    step_keys = jax.random.split(key, max_steps)
    state = reset_episode(
        env_data,
        key_reset,
        ctrnn_hidden,
        ctrnn_grid_hidden,
        robot_h=static_robot_h,
        robot_w=static_robot_w,
        gasnet_mode=gasnet_mode,
        ou_enabled=ou_enabled,
        mace_mode=mace_mode,
        mace_init_mass=mace_init_mass,
        growth_enabled=growth_enabled,
        fixed_hidden_init_scale=fixed_hidden_init_scale,
        frozen_identity=frozen_identity,
        initial_nca_state=initial_nca_state,
        initial_ctrnn_carry=initial_ctrnn_carry,
    )
    frozen_deformation_start = _apply_deformation_variant(
        _compute_dual_deformation(state.sim_state, env_data),
        variant=deformation_variant,
        noise_sigma_signed_delta=deformation_noise_sigma_signed_delta,
        shuffle_mode=deformation_shuffle_mode,
        variant_phase_key=variant_episode_key,
        variant_episode_key=variant_episode_key,
    )
    _validate_forced_zero_deformation_channels(
        frozen_deformation_start,
        variant=deformation_variant,
    )
    frozen_perception_input_start = None
    if normalized_sensing_freeze_mode == "perception_frozen_start":
        frozen_perception_state = inject_deformation_channels(
            state.nca_state,
            frozen_deformation_start,
            env_data.deform_mask,
            alive_mask,
            robot_offset_y=static_spawn_y,
            robot_offset_x=static_spawn_x,
            h=static_grid_h,
            w=static_grid_w,
            ablate_deformation=(
                ablate_deformation
                or _deformation_variant_forces_zero_channels(deformation_variant)
            ),
            deformation_gain_absolute=deformation_gain_absolute,
            deformation_gain_compliance=deformation_gain_compliance,
        )
        frozen_perception_input_start = controller_perception_input(
            frozen_perception_state,
            alive_mask,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
            signaling_variant=signaling_variant,
            growth_enabled=growth_enabled,
            active_comm_filters=active_comm_filters,
        )
    h_compact_indices = env_data.per_axis_info.h_compact_cell_indices
    v_compact_indices = env_data.per_axis_info.v_compact_cell_indices
    h_compact_valid = (
        jnp.arange(h_compact_indices.shape[0], dtype=jnp.int32)
        < env_data.per_axis_info.h_compact_count
    )
    v_compact_valid = (
        jnp.arange(v_compact_indices.shape[0], dtype=jnp.int32)
        < env_data.per_axis_info.v_compact_count
    )
    zero_h_compact = jnp.zeros_like(h_compact_indices, dtype=jnp.float32)
    zero_v_compact = jnp.zeros_like(v_compact_indices, dtype=jnp.float32)
    nca_steps_per_phase_f32 = jnp.float32(max(int(nca_steps_per_phase), 1))
    capture_substep_probe = capture_mode == "substep_probe"
    target_substep_probe_step = jnp.int32(
        -1 if substep_probe_step_index is None else substep_probe_step_index
    )
    target_substep_probe_phase = jnp.int32(
        -1 if substep_probe_phase_index is None else substep_probe_phase_index
    )

    def _phase_probe_step_from_state(
        s: CliffEpisodeState,
        *,
        h_compact: jnp.ndarray,
        v_compact: jnp.ndarray,
        post_sim_state: SimState,
        done_after: jnp.ndarray,
    ) -> EpisodePhaseProbeStep:
        return EpisodePhaseProbeStep(
            pre_positions=s.sim_state.positions,
            pre_velocities=s.sim_state.velocities,
            pre_spring_rest_length=s.sim_state.spring_rest_length,
            h_compact=h_compact,
            v_compact=v_compact,
            controller_hidden_output_max_abs=s.controller_hidden_output_max_abs,
            action_output_max_abs=s.action_output_max_abs,
            post_positions=post_sim_state.positions,
            post_velocities=post_sim_state.velocities,
            post_velocities_true=post_sim_state.velocities_true,
            post_spring_rest_length=post_sim_state.spring_rest_length,
            last_substep_post_integrator_velocities=post_sim_state.velocities,
            last_substep_final_velocities=post_sim_state.velocities,
            last_substep_final_velocities_true=post_sim_state.velocities_true,
            done_before=s.done,
            done_after=done_after,
        )

    def _zero_substep_probe(sim_state: SimState) -> PhysicsSubstepProbe:
        n_points = sim_state.positions.shape[0]
        return PhysicsSubstepProbe(
            dynamic_selected_pair_mask=jnp.zeros_like(
                env_data.collision_data.candidate_pair_active, dtype=jnp.bool_
            ),
            dynamic_overflow=jnp.bool_(False),
            dynamic_selected_triple_active=jnp.zeros_like(
                env_data.collision_data.triple_active, dtype=jnp.bool_
            ),
            dynamic_triple_active=jnp.zeros_like(
                env_data.collision_data.triple_active, dtype=jnp.bool_
            ),
            dynamic_forces=jnp.zeros_like(sim_state.positions),
            static_forces=jnp.zeros_like(sim_state.positions),
            static_contact_mask=jnp.zeros((n_points,), dtype=jnp.bool_),
            static_memory_contact_mask=jnp.zeros((n_points,), dtype=jnp.bool_),
            tangential_deformation=jnp.zeros_like(sim_state.tangential_deformation),
            static_manifold_active=jnp.zeros((n_points,), dtype=jnp.bool_),
            post_update_actuator_spring_rest_length=sim_state.spring_rest_length,
            post_integrator_positions=sim_state.positions,
            post_integrator_velocities=sim_state.velocities,
            post_constraint_positions=sim_state.positions,
            post_anchor_positions=sim_state.positions,
            final_positions=sim_state.positions,
            velocity_true_positions_last=sim_state.positions_last,
            velocity_true_displacement=sim_state.positions - sim_state.positions_last,
            final_velocities=sim_state.velocities,
            final_velocities_true=sim_state.velocities_true,
        )

    def _substep_trace_from_phase(
        s: CliffEpisodeState,
        *,
        step_index: jnp.ndarray,
        phase_index: jnp.ndarray,
        h_compact: jnp.ndarray,
        v_compact: jnp.ndarray,
        controller_hidden_output_max_abs: jnp.ndarray,
        action_output_max_abs: jnp.ndarray,
        substep_outputs: PhysicsSubstepProbe,
    ) -> EpisodeSubstepProbeTrace:
        return EpisodeSubstepProbeTrace(
            step_index=step_index,
            phase_index=phase_index,
            phase_pre_positions=s.sim_state.positions,
            phase_pre_velocities=s.sim_state.velocities,
            phase_pre_spring_rest_length=s.sim_state.spring_rest_length,
            phase_h_compact=h_compact,
            phase_v_compact=v_compact,
            phase_controller_hidden_output_max_abs=controller_hidden_output_max_abs,
            phase_action_output_max_abs=action_output_max_abs,
            substep_dynamic_selected_pair_mask=substep_outputs.dynamic_selected_pair_mask,
            substep_dynamic_overflow=substep_outputs.dynamic_overflow,
            substep_dynamic_selected_triple_active=substep_outputs.dynamic_selected_triple_active,
            substep_dynamic_triple_active=substep_outputs.dynamic_triple_active,
            substep_dynamic_forces=substep_outputs.dynamic_forces,
            substep_static_forces=substep_outputs.static_forces,
            substep_static_contact_mask=substep_outputs.static_contact_mask,
            substep_static_memory_contact_mask=substep_outputs.static_memory_contact_mask,
            substep_tangential_deformation=substep_outputs.tangential_deformation,
            substep_static_manifold_active=substep_outputs.static_manifold_active,
            substep_post_update_actuator_spring_rest_length=substep_outputs.post_update_actuator_spring_rest_length,
            substep_post_integrator_positions=substep_outputs.post_integrator_positions,
            substep_post_integrator_velocities=substep_outputs.post_integrator_velocities,
            substep_post_constraint_positions=substep_outputs.post_constraint_positions,
            substep_post_anchor_positions=substep_outputs.post_anchor_positions,
            substep_final_positions=substep_outputs.final_positions,
            substep_velocity_true_positions_last=substep_outputs.velocity_true_positions_last,
            substep_velocity_true_displacement=substep_outputs.velocity_true_displacement,
            substep_final_velocities=substep_outputs.final_velocities,
            substep_final_velocities_true=substep_outputs.final_velocities_true,
        )

    def _state_trace_step(
        s: CliffEpisodeState,
        *,
        env_step: jnp.ndarray,
    ) -> EpisodeStateTrace:
        stats = _robot_kinematics(s.sim_state, env_data.robot_point_mask)
        return EpisodeStateTrace(
            env_step=env_step,
            nca_state=s.nca_state,
            ctrnn_carry=s.ctrnn_carry,
            action_outputs=s.nca_state[:, :, ACTION_H_IDX:ACTION_V_IDX + 1],
            com_x=stats.com_x,
            min_x=stats.min_x,
            front_x=stats.front_x,
            goal_touched=s.goal_touched,
            cliff_fell=s.cliff_fell,
            done=s.done,
        )

    def _com_trace_step(s: CliffEpisodeState) -> EpisodeComTrace:
        stats = _robot_kinematics(s.sim_state, env_data.robot_point_mask)
        return EpisodeComTrace(com_x=stats.com_x)

    def _zero_controller_trace_step(
        s: CliffEpisodeState,
    ) -> EpisodeControllerTrace:
        zero_scalar = jnp.zeros((nca_steps_per_phase,), dtype=jnp.float32)
        return EpisodeControllerTrace(
            nca_state=jnp.zeros(
                (nca_steps_per_phase, *s.nca_state.shape),
                dtype=s.nca_state.dtype,
            ),
            ctrnn_carry=jnp.zeros(
                (nca_steps_per_phase, *s.ctrnn_carry.shape),
                dtype=s.ctrnn_carry.dtype,
            ),
            action_outputs=jnp.zeros(
                (
                    nca_steps_per_phase,
                    *s.nca_state[:, :, ACTION_H_IDX:ACTION_V_IDX + 1].shape,
                ),
                dtype=s.nca_state.dtype,
            ),
            com_x=zero_scalar,
            back_x=zero_scalar,
            front_x=zero_scalar,
            active=jnp.zeros((nca_steps_per_phase,), dtype=jnp.bool_),
        )

    def _zero_state_space_phase_step(s: CliffEpisodeState) -> EpisodeStateSpacePhaseStep:
        return EpisodeStateSpacePhaseStep(
            active=jnp.bool_(False),
            deformation_inputs=jnp.zeros(
                (env_data.deform_mask.shape[0], 4),
                dtype=s.nca_state.dtype,
            ),
            h_compact=zero_h_compact,
            v_compact=zero_v_compact,
            controller_hidden_output_max_abs=s.controller_hidden_output_max_abs,
            action_output_max_abs=s.action_output_max_abs,
            nca_state=s.nca_state,
            ctrnn_carry=s.ctrnn_carry,
            action_outputs=s.nca_state[:, :, ACTION_H_IDX:ACTION_V_IDX + 1],
            post_positions=s.sim_state.positions,
            post_velocities=s.sim_state.velocities,
            post_velocities_true=s.sim_state.velocities_true,
            post_spring_rest_length=s.sim_state.spring_rest_length,
        )

    def _state_space_trace_step(
        *,
        env_step: jnp.ndarray,
        pre_state: CliffEpisodeState,
        post_state: CliffEpisodeState,
        phase_trace: EpisodeStateSpacePhaseStep,
    ) -> EpisodeStateSpaceTrace:
        stats = _robot_kinematics(post_state.sim_state, env_data.robot_point_mask)
        return EpisodeStateSpaceTrace(
            env_step=env_step,
            active=~pre_state.done,
            done_before=pre_state.done,
            done_after=post_state.done,
            pre_positions=pre_state.sim_state.positions,
            pre_velocities=pre_state.sim_state.velocities,
            pre_spring_rest_length=pre_state.sim_state.spring_rest_length,
            post_positions=post_state.sim_state.positions,
            post_velocities=post_state.sim_state.velocities,
            post_velocities_true=post_state.sim_state.velocities_true,
            post_spring_rest_length=post_state.sim_state.spring_rest_length,
            nca_state=post_state.nca_state,
            ctrnn_carry=post_state.ctrnn_carry,
            action_outputs=post_state.nca_state[:, :, ACTION_H_IDX:ACTION_V_IDX + 1],
            phase=phase_trace,
            com_x=stats.com_x,
            com_y=stats.com_y,
            min_x=stats.min_x,
            max_x=stats.front_x,
            min_y=stats.min_y,
            max_y=stats.max_y,
            front_x=stats.front_x,
            back_x=stats.min_x,
            goal_touched=post_state.goal_touched,
            cliff_fell=post_state.cliff_fell,
        )

    zero_substep_probe_trace = _substep_trace_from_phase(
        state,
        step_index=jnp.int32(-1),
        phase_index=jnp.int32(-1),
        h_compact=zero_h_compact,
        v_compact=zero_v_compact,
        controller_hidden_output_max_abs=state.controller_hidden_output_max_abs,
        action_output_max_abs=state.action_output_max_abs,
        substep_outputs=jax.tree_util.tree_map(
            lambda array: jnp.repeat(array[None, ...], substeps_per_phase, axis=0),
            _zero_substep_probe(state.sim_state),
        ),
    )

    def _phase_step(phase_carry, phase_inputs):
        if capture_substep_probe:
            phase_key, variant_phase_key, phase_index = phase_inputs
        else:
            phase_key, variant_phase_key = phase_inputs
            phase_index = jnp.int32(-1)

        def _skip_phase(carry_in):
            s = carry_in.episode_state if capture_substep_probe else carry_in
            if capture_mode == "phase_probe":
                return s, _phase_probe_step_from_state(
                    s,
                    h_compact=zero_h_compact,
                    v_compact=zero_v_compact,
                    post_sim_state=s.sim_state,
                    done_after=s.done,
                )
            if capture_mode == "controller_trace":
                return s, _zero_controller_trace_step(s)
            if capture_mode == "state_space_trace":
                return s, _zero_state_space_phase_step(s)
            if capture_substep_probe:
                return carry_in, None
            return s, None

        def _active_phase(carry_in):
            s = carry_in.episode_state if capture_substep_probe else carry_in
            current_substep_trace = (
                carry_in.substep_trace if capture_substep_probe else None
            )
            force_zero_deformation_channels = _deformation_variant_forces_zero_channels(
                deformation_variant
            )
            pre_state = s
            phase_robot_stats = _robot_kinematics(s.sim_state, env_data.robot_point_mask)
            deformation = _compute_dual_deformation(s.sim_state, env_data)
            deformation = _apply_deformation_variant(
                deformation,
                variant=deformation_variant,
                noise_sigma_signed_delta=deformation_noise_sigma_signed_delta,
                shuffle_mode=deformation_shuffle_mode,
                variant_phase_key=variant_phase_key,
                variant_episode_key=variant_episode_key,
            )
            if normalized_sensing_freeze_mode == "deformation_frozen_start":
                deformation = frozen_deformation_start
            _validate_forced_zero_deformation_channels(
                deformation,
                variant=deformation_variant,
            )
            deformation_channel_max = update_deformation_channel_max(
                s.deformation_channel_max,
                deformation,
            )
            nca_s = s.nca_state
            ctrnn_c = s.ctrnn_carry

            def _nca_micro(micro_carry, micro_key):
                ns, cc, h_sum, v_sum, hidden_max_abs, action_max_abs = micro_carry
                ns_new, cc_new, h_a, v_a, step_diag = nca_step_with_diagnostics(
                    ns,
                    cc,
                    deformation,
                    env_data.deform_mask,
                    env_data.actuator_mask,
                    alive_mask,
                    params,
                    micro_key,
                    robot_offset_y=static_spawn_y,
                    robot_offset_x=static_spawn_x,
                    h=static_grid_h,
                    w=static_grid_w,
                    ctrnn_grid_hidden=ctrnn_grid_hidden,
                    p_update=p_update,
                    active_comm_filters=active_comm_filters,
                    ou_theta=ou_theta,
                    ou_sigma=ou_sigma,
                    ou_enabled=ou_enabled,
                    ctrnn_tau_min=ctrnn_tau_min,
                    ctrnn_tau_max=ctrnn_tau_max,
                    ctrnn_dt=ctrnn_dt_per_update,
                    gasnet_mode=gasnet_mode,
                    gasnet_type=gasnet_type,
                    gas_decay_rate=gas_decay_rate,
                    gas_diffusion_radius=gas_diffusion_radius,
                    gas_gain_sensitivity=gas_gain_sensitivity,
                    gas_max_concentration=gas_max_concentration,
                    gas_gain_min=gas_gain_min,
                    gas_gain_max=gas_gain_max,
                    ablate_deformation=ablate_deformation or force_zero_deformation_channels,
                    update_mode=update_mode,
                    state_clip_value=state_clip_value,
                    mace_mode=mace_mode,
                    mace_beta=mace_beta,
                    deformation_gain_absolute=deformation_gain_absolute,
                    deformation_gain_compliance=deformation_gain_compliance,
                    signaling_variant=signaling_variant,
                    frozen_perception_input=frozen_perception_input_start,
                    growth_enabled=growth_enabled,
                )
                if bool(growth_enabled) and frozen_identity is not None:
                    ns_new = clamp_growth_channels(
                        ns_new,
                        frozen_identity,
                        ctrnn_grid_hidden=ctrnn_grid_hidden,
                        gasnet_mode=gasnet_mode,
                        ou_enabled=ou_enabled,
                    )
                h_compact = jnp.where(h_compact_valid, h_a[h_compact_indices], 0.0)
                v_compact = jnp.where(v_compact_valid, v_a[v_compact_indices], 0.0)
                micro_trace = None
                if capture_mode == "controller_trace":
                    micro_trace = EpisodeControllerTrace(
                        nca_state=ns_new,
                        ctrnn_carry=cc_new,
                        action_outputs=ns_new[:, :, ACTION_H_IDX:ACTION_V_IDX + 1],
                        com_x=phase_robot_stats.com_x,
                        back_x=phase_robot_stats.min_x,
                        front_x=phase_robot_stats.front_x,
                        active=jnp.bool_(True),
                    )
                return (
                    ns_new,
                    cc_new,
                    h_sum + h_compact,
                    v_sum + v_compact,
                    jnp.maximum(hidden_max_abs, step_diag.controller_hidden_output_max_abs),
                    jnp.maximum(action_max_abs, step_diag.action_output_max_abs),
                ), micro_trace

            micro_keys = jax.random.split(phase_key, nca_steps_per_phase)
            (
                nca_s,
                ctrnn_c,
                h_compact_sum,
                v_compact_sum,
                controller_hidden_output_max_abs,
                action_output_max_abs,
            ), micro_trace_outputs = jax.lax.scan(
                _nca_micro,
                (
                    nca_s,
                    ctrnn_c,
                    zero_h_compact,
                    zero_v_compact,
                    s.controller_hidden_output_max_abs,
                    s.action_output_max_abs,
                ),
                micro_keys,
            )
            h_compact = h_compact_sum / nca_steps_per_phase_f32
            v_compact = v_compact_sum / nca_steps_per_phase_f32
            sim = set_per_axis_goals_compact(
                s.sim_state, env_data.per_axis_info, h_compact, v_compact
            )
            capture_this_phase = capture_substep_probe & (
                (s.step_count == target_substep_probe_step)
                & (phase_index == target_substep_probe_phase)
            )

            def _physics_substep_with_probe(sim_carry, _):
                return physics_substep_dynamic_broadphase_with_probe(
                    sim_carry,
                    env_data.topology,
                    env_data.constants,
                    collision_data=env_data.collision_data,
                    static_collider_data=env_data.static_collider_data,
                    static_collision_worklist=env_data.static_collision_worklist,
                    dynamic_max_candidates=jnp.int32(dynamic_max_candidates),
                    dynamic_bin_size=jnp.float32(dynamic_bin_size),
                    dynamic_impl=dynamic_impl,
                    dynamic_max_narrow=dynamic_max_narrow,
                )

            if capture_mode in {"phase_probe", "substep_probe"}:
                sim, substep_outputs = jax.lax.scan(
                    _physics_substep_with_probe,
                    sim,
                    None,
                    length=substeps_per_phase,
                )
            else:

                def _physics_substep(sim_carry, _):
                    next_sim = physics_substep_dynamic_broadphase(
                        sim_carry,
                        env_data.topology,
                        env_data.constants,
                        collision_data=env_data.collision_data,
                        static_collider_data=env_data.static_collider_data,
                        static_collision_worklist=env_data.static_collision_worklist,
                        dynamic_max_candidates=jnp.int32(dynamic_max_candidates),
                        dynamic_bin_size=jnp.float32(dynamic_bin_size),
                        dynamic_impl=dynamic_impl,
                        dynamic_max_narrow=dynamic_max_narrow,
                    )
                    return next_sim, None

                sim, _ = jax.lax.scan(
                    _physics_substep,
                    sim,
                    None,
                    length=substeps_per_phase,
                )

            robot_stats = _robot_kinematics(sim, env_data.robot_point_mask)

            hit_death_plane = robot_stats.min_y < env_data.death_plane_y
            airborne = robot_stats.com_y > (
                env_data.init_com_y
                + (jnp.float32(airborne_kill_height_margin_cells) * CELL_SIZE)
            )
            com_y_delta_cells = jnp.maximum(
                (robot_stats.com_y - env_data.init_com_y) / jnp.float32(CELL_SIZE),
                jnp.float32(0.0),
            )
            soft_loft_excess_cells = jnp.maximum(
                com_y_delta_cells - jnp.float32(soft_loft_threshold_cells),
                jnp.float32(0.0),
            )
            soft_loft_excess_cells = jnp.where(~s.done, soft_loft_excess_cells, 0.0)
            lofted_step = (~s.done) & (soft_loft_excess_cells > 0.0)
            left_edge = robot_stats.com_x < 0.0
            goal_touched_now = jax.lax.cond(
                bilateral_task,
                lambda _: _robot_touches_goal_block(
                    sim,
                    env_data.robot_point_mask,
                    goal_block_x=env_data.goal_block_x,
                    goal_block_y=env_data.goal_block_y,
                ),
                lambda _: jnp.bool_(False),
                operand=None,
            )
            cliff_fell_now = hit_death_plane | airborne
            left_edge_failure = (~bilateral_task) & left_edge

            new_done = s.done | cliff_fell_now | left_edge_failure | goal_touched_now
            new_touched = s.touched_death_plane | hit_death_plane
            new_airborne = s.went_airborne | airborne
            new_com_neg = s.com_x_went_negative | left_edge
            new_goal_touched = s.goal_touched | goal_touched_now
            new_cliff_fell = s.cliff_fell | cliff_fell_now
            new_peak = jnp.where(
                ~s.done,
                jnp.maximum(s.peak_front_x, robot_stats.front_x),
                s.peak_front_x,
            )
            new_safe = jnp.where(~new_done, robot_stats.front_x, s.safe_front_x)

            activation, h_activation, v_activation = _phase_activation_from_compact_actions(
                h_compact,
                v_compact,
                h_compact_valid,
                v_compact_valid,
            )
            activation = jnp.where(~s.done, activation, 0.0)
            h_activation = jnp.where(~s.done, h_activation, 0.0)
            v_activation = jnp.where(~s.done, v_activation, 0.0)
            new_state = CliffEpisodeState(
                sim_state=sim,
                nca_state=nca_s,
                ctrnn_carry=ctrnn_c,
                step_count=s.step_count,
                done=new_done,
                peak_front_x=new_peak,
                safe_front_x=new_safe,
                touched_death_plane=new_touched,
                went_airborne=new_airborne,
                com_x_went_negative=new_com_neg,
                cumulative_activation=s.cumulative_activation + activation,
                cumulative_h_activation=s.cumulative_h_activation + h_activation,
                cumulative_v_activation=s.cumulative_v_activation + v_activation,
                cumulative_speed=s.cumulative_speed,
                cumulative_lower_ledge_exposure_steps=s.cumulative_lower_ledge_exposure_steps,
                cumulative_cliff_vicinity_steps=s.cumulative_cliff_vicinity_steps,
                cumulative_lofted_steps=(
                    s.cumulative_lofted_steps
                    + jnp.where(lofted_step, jnp.int32(1), jnp.int32(0))
                ),
                cumulative_soft_loft_excess_cells=(
                    s.cumulative_soft_loft_excess_cells + soft_loft_excess_cells
                ),
                first_cliff_vicinity_step=s.first_cliff_vicinity_step,
                peak_any_direction_progress=s.peak_any_direction_progress,
                peak_goal_progress=s.peak_goal_progress,
                peak_incorrect_progress=s.peak_incorrect_progress,
                peak_toward_goal_frontier=s.peak_toward_goal_frontier,
                goal_touched=new_goal_touched,
                cliff_fell=new_cliff_fell,
                first_goal_touch_step=jnp.where(
                    (s.first_goal_touch_step < 0) & goal_touched_now,
                    s.step_count + jnp.int32(1),
                    s.first_goal_touch_step,
                ),
                last_significant_direction=s.last_significant_direction,
                turn_count=s.turn_count,
                stable_turn_direction=s.stable_turn_direction,
                stable_turn_distance=s.stable_turn_distance,
                stable_turn_eligible=s.stable_turn_eligible,
                stable_turn_count=s.stable_turn_count,
                steps_alive=s.steps_alive
                + jnp.where(~s.done, jnp.int32(1), jnp.int32(0)),
                deformation_channel_max=deformation_channel_max,
                controller_hidden_output_max_abs=controller_hidden_output_max_abs,
                action_output_max_abs=action_output_max_abs,
                peak_com_y_delta_cells=jnp.where(
                    ~s.done,
                    jnp.maximum(s.peak_com_y_delta_cells, com_y_delta_cells),
                    s.peak_com_y_delta_cells,
                ),
                first_significant_direction=s.first_significant_direction,
            )
            if capture_mode == "phase_probe":
                return new_state, EpisodePhaseProbeStep(
                    pre_positions=pre_state.sim_state.positions,
                    pre_velocities=pre_state.sim_state.velocities,
                    pre_spring_rest_length=pre_state.sim_state.spring_rest_length,
                    h_compact=h_compact,
                    v_compact=v_compact,
                    controller_hidden_output_max_abs=controller_hidden_output_max_abs,
                    action_output_max_abs=action_output_max_abs,
                    post_positions=sim.positions,
                    post_velocities=sim.velocities,
                    post_velocities_true=sim.velocities_true,
                    post_spring_rest_length=sim.spring_rest_length,
                    last_substep_post_integrator_velocities=substep_outputs.post_integrator_velocities[
                        -1
                    ],
                    last_substep_final_velocities=substep_outputs.final_velocities[-1],
                    last_substep_final_velocities_true=substep_outputs.final_velocities_true[
                        -1
                    ],
                    done_before=pre_state.done,
                    done_after=new_done,
                )
            if capture_mode == "controller_trace":
                assert micro_trace_outputs is not None
                return new_state, micro_trace_outputs
            if capture_mode == "state_space_trace":
                return new_state, EpisodeStateSpacePhaseStep(
                    active=jnp.bool_(True),
                    deformation_inputs=deformation,
                    h_compact=h_compact,
                    v_compact=v_compact,
                    controller_hidden_output_max_abs=controller_hidden_output_max_abs,
                    action_output_max_abs=action_output_max_abs,
                    nca_state=nca_s,
                    ctrnn_carry=ctrnn_c,
                    action_outputs=nca_s[:, :, ACTION_H_IDX:ACTION_V_IDX + 1],
                    post_positions=sim.positions,
                    post_velocities=sim.velocities,
                    post_velocities_true=sim.velocities_true,
                    post_spring_rest_length=sim.spring_rest_length,
                )
            if capture_substep_probe:
                next_trace = jax.lax.cond(
                    capture_this_phase,
                    lambda _: _substep_trace_from_phase(
                        pre_state,
                        step_index=s.step_count,
                        phase_index=phase_index,
                        h_compact=h_compact,
                        v_compact=v_compact,
                        controller_hidden_output_max_abs=controller_hidden_output_max_abs,
                        action_output_max_abs=action_output_max_abs,
                        substep_outputs=substep_outputs,
                    ),
                    lambda _: current_substep_trace,
                    operand=None,
                )
                return EpisodeSubstepProbeCarry(new_state, next_trace), None
            return new_state, None

        done_pred = (
            phase_carry.episode_state.done if capture_substep_probe else phase_carry.done
        )
        return jax.lax.cond(done_pred, _skip_phase, _active_phase, phase_carry)

    def _env_step(carry, step_inputs):
        step_index, step_key = step_inputs

        def _step_output(s: CliffEpisodeState):
            if capture_mode == "render":
                return RenderStepOutput(
                    positions=s.sim_state.positions,
                    spring_rest_length=s.sim_state.spring_rest_length,
                    reward=jnp.float32(0.0),
                    done=s.done,
                )
            if capture_mode == "state_trace":
                return _state_trace_step(s, env_step=step_index + jnp.int32(1))
            if capture_mode == "com_trace":
                return _com_trace_step(s)
            if capture_mode == "controller_trace":
                return jax.tree_util.tree_map(
                    lambda array: jnp.repeat(array[None, ...], sense_act_phases, axis=0),
                    _zero_controller_trace_step(s),
                )
            if capture_mode == "state_space_trace":
                zero_phase_step = jax.tree_util.tree_map(
                    lambda array: jnp.repeat(array[None, ...], sense_act_phases, axis=0),
                    _zero_state_space_phase_step(s),
                )
                return _state_space_trace_step(
                    env_step=step_index + jnp.int32(1),
                    pre_state=s,
                    post_state=s,
                    phase_trace=zero_phase_step,
                )
            return None

        def _skip_env_step(carry_in):
            s = carry_in.episode_state if capture_substep_probe else carry_in
            if capture_mode == "phase_probe":
                zero_phase_step = _phase_probe_step_from_state(
                    s,
                    h_compact=zero_h_compact,
                    v_compact=zero_v_compact,
                    post_sim_state=s.sim_state,
                    done_after=s.done,
                )
                return s, jax.tree_util.tree_map(
                    lambda array: jnp.repeat(array[None, ...], sense_act_phases, axis=0),
                    zero_phase_step,
                )
            if capture_mode == "state_space_trace":
                zero_phase_step = jax.tree_util.tree_map(
                    lambda array: jnp.repeat(array[None, ...], sense_act_phases, axis=0),
                    _zero_state_space_phase_step(s),
                )
                return s, _state_space_trace_step(
                    env_step=step_index + jnp.int32(1),
                    pre_state=s,
                    post_state=s,
                    phase_trace=zero_phase_step,
                )
            if capture_substep_probe:
                return carry_in, None
            return s, _step_output(s)

        def _active_env_step(carry_in):
            s = carry_in.episode_state if capture_substep_probe else carry_in
            pre_env_state = s
            current_substep_trace = (
                carry_in.substep_trace if capture_substep_probe else None
            )
            step_start_com_x = _robot_com_x(s.sim_state, env_data.robot_point_mask)
            phase_keys = jax.random.split(step_key, sense_act_phases)
            variant_phase_keys = jax.random.split(
                jax.random.fold_in(step_key, 1),
                sense_act_phases,
            )
            if capture_substep_probe:
                phase_indices = jnp.arange(sense_act_phases, dtype=jnp.int32)
                phase_carry, _ = jax.lax.scan(
                    _phase_step,
                    EpisodeSubstepProbeCarry(s, current_substep_trace),
                    (phase_keys, variant_phase_keys, phase_indices),
                )
                s = phase_carry.episode_state
                next_substep_trace = phase_carry.substep_trace
                phase_outputs = None
            else:
                s, phase_outputs = jax.lax.scan(
                    _phase_step, s, (phase_keys, variant_phase_keys)
                )
                next_substep_trace = None
            step_stats = _robot_kinematics(s.sim_state, env_data.robot_point_mask)
            step_com_x = step_stats.com_x
            step_delta_com_x = step_com_x - step_start_com_x
            step_speed = jnp.abs(step_delta_com_x)
            lower_ledge_exposure = _lower_ledge_exposure_now(
                step_stats,
                cliff_probe_ledge_width=env_data.cliff_probe_ledge_width,
                cliff_probe_ledge_cell_x=env_data.cliff_probe_ledge_cell_x,
                cliff_probe_ledge_cell_y=env_data.cliff_probe_ledge_cell_y,
            )
            in_cliff_vicinity = jnp.where(
                bilateral_task,
                jnp.bool_(False),
                step_com_x >= cliff_vicinity_threshold,
            )
            any_direction_progress = jnp.abs(step_com_x - jnp.asarray(env_data.init_com_x, dtype=jnp.float32))
            init_goal_edge_x = _goal_facing_edge_x(
                min_x=jnp.asarray(env_data.init_min_x, dtype=jnp.float32),
                front_x=jnp.asarray(env_data.init_front_x, dtype=jnp.float32),
                goal_direction=jnp.asarray(env_data.goal_direction, dtype=jnp.int32),
            )
            step_goal_edge_x = _goal_facing_edge_x(
                min_x=step_stats.min_x,
                front_x=step_stats.front_x,
                goal_direction=jnp.asarray(env_data.goal_direction, dtype=jnp.int32),
            )
            goal_progress = jnp.maximum(
                jnp.asarray(env_data.goal_direction, dtype=jnp.float32)
                * (step_goal_edge_x - init_goal_edge_x),
                0.0,
            )
            init_wrong_edge_x = _wrong_facing_edge_x(
                min_x=jnp.asarray(env_data.init_min_x, dtype=jnp.float32),
                front_x=jnp.asarray(env_data.init_front_x, dtype=jnp.float32),
                goal_direction=jnp.asarray(env_data.goal_direction, dtype=jnp.int32),
            )
            step_wrong_edge_x = _wrong_facing_edge_x(
                min_x=step_stats.min_x,
                front_x=step_stats.front_x,
                goal_direction=jnp.asarray(env_data.goal_direction, dtype=jnp.int32),
            )
            incorrect_progress = jnp.maximum(
                -jnp.asarray(env_data.goal_direction, dtype=jnp.float32)
                * (step_wrong_edge_x - init_wrong_edge_x),
                0.0,
            )
            next_direction, next_turn_count = _update_turn_tracking(
                last_significant_direction=s.last_significant_direction,
                turn_count=s.turn_count,
                delta_com_x=step_delta_com_x,
                turn_motion_threshold=turn_motion_threshold,
            )
            step_direction = _significant_step_direction(
                delta_com_x=step_delta_com_x,
                turn_motion_threshold=turn_motion_threshold,
            )
            (
                next_stable_turn_direction,
                next_stable_turn_distance,
                next_stable_turn_eligible,
                next_stable_turn_count,
            ) = _update_stable_turn_tracking(
                stable_turn_direction=s.stable_turn_direction,
                stable_turn_distance=s.stable_turn_distance,
                stable_turn_eligible=s.stable_turn_eligible,
                stable_turn_count=s.stable_turn_count,
                delta_com_x=step_delta_com_x,
                turn_motion_threshold=turn_motion_threshold,
            )
            s = s._replace(
                cumulative_speed=s.cumulative_speed + step_speed,
                cumulative_lower_ledge_exposure_steps=(
                    s.cumulative_lower_ledge_exposure_steps
                    + lower_ledge_exposure.astype(jnp.int32)
                ),
                cumulative_cliff_vicinity_steps=(
                    s.cumulative_cliff_vicinity_steps + in_cliff_vicinity.astype(jnp.int32)
                ),
                first_cliff_vicinity_step=jnp.where(
                    (s.first_cliff_vicinity_step < 0) & in_cliff_vicinity,
                    s.step_count + jnp.int32(1),
                    s.first_cliff_vicinity_step,
                ),
                peak_any_direction_progress=jnp.where(
                    bilateral_task & (~s.done),
                    jnp.maximum(s.peak_any_direction_progress, any_direction_progress),
                    s.peak_any_direction_progress,
                ),
                peak_goal_progress=jnp.where(
                    bilateral_task & (~s.done),
                    jnp.maximum(s.peak_goal_progress, goal_progress),
                    s.peak_goal_progress,
                ),
                peak_incorrect_progress=jnp.where(
                    bilateral_task & (~s.done),
                    jnp.maximum(s.peak_incorrect_progress, incorrect_progress),
                    s.peak_incorrect_progress,
                ),
                peak_toward_goal_frontier=jnp.where(
                    bilateral_task & (~s.done),
                    s.peak_toward_goal_frontier,
                    s.peak_toward_goal_frontier,
                ),
                last_significant_direction=jnp.where(
                    bilateral_task & (~s.done),
                    next_direction,
                    s.last_significant_direction,
                ),
                first_significant_direction=jnp.where(
                    bilateral_task
                    & (~s.done)
                    & (s.first_significant_direction == 0)
                    & (step_direction != 0),
                    step_direction,
                    s.first_significant_direction,
                ),
                turn_count=jnp.where(
                    bilateral_task & (~s.done),
                    next_turn_count,
                    s.turn_count,
                ),
                stable_turn_direction=jnp.where(
                    bilateral_task & (~s.done),
                    next_stable_turn_direction,
                    s.stable_turn_direction,
                ),
                stable_turn_distance=jnp.where(
                    bilateral_task & (~s.done),
                    next_stable_turn_distance,
                    s.stable_turn_distance,
                ),
                stable_turn_eligible=jnp.where(
                    bilateral_task & (~s.done),
                    next_stable_turn_eligible,
                    s.stable_turn_eligible,
                ),
                stable_turn_count=jnp.where(
                    bilateral_task & (~s.done),
                    next_stable_turn_count,
                    s.stable_turn_count,
                ),
            )
            new_step = s.step_count + jnp.int32(1)
            new_done = s.done | (new_step >= max_steps)
            s = s._replace(step_count=new_step, done=new_done)
            if capture_mode == "phase_probe":
                phase_outputs = phase_outputs._replace(
                    done_after=phase_outputs.done_after.at[-1].set(s.done)
                )
                return s, phase_outputs
            if capture_mode == "controller_trace":
                assert phase_outputs is not None
                return s, phase_outputs
            if capture_mode == "state_space_trace":
                assert phase_outputs is not None
                return s, _state_space_trace_step(
                    env_step=step_index + jnp.int32(1),
                    pre_state=pre_env_state,
                    post_state=s,
                    phase_trace=phase_outputs,
                )
            if capture_substep_probe:
                return EpisodeSubstepProbeCarry(s, next_substep_trace), None
            return s, _step_output(s)

        done_pred = carry.episode_state.done if capture_substep_probe else carry.done
        return jax.lax.cond(done_pred, _skip_env_step, _active_env_step, carry)

    initial_carry = (
        EpisodeSubstepProbeCarry(state, zero_substep_probe_trace)
        if capture_substep_probe
        else state
    )
    step_indices = jnp.arange(max_steps, dtype=jnp.int32)
    final_carry, trace_outputs = jax.lax.scan(
        _env_step,
        initial_carry,
        (step_indices, step_keys),
    )
    final_state = final_carry.episode_state if capture_substep_probe else final_carry
    episode_result = _episode_result_from_state(final_state, env_data, max_steps=max_steps)
    if capture_mode == "phase_probe":
        assert trace_outputs is not None
        phase_trace = EpisodePhaseProbeTrace(
            initial_positions=state.sim_state.positions,
            initial_velocities=state.sim_state.velocities,
            initial_spring_rest_length=state.sim_state.spring_rest_length,
            initial_nca_state=state.nca_state,
            initial_ctrnn_carry=state.ctrnn_carry,
            phase_pre_positions=trace_outputs.pre_positions,
            phase_pre_velocities=trace_outputs.pre_velocities,
            phase_pre_spring_rest_length=trace_outputs.pre_spring_rest_length,
            phase_h_compact=trace_outputs.h_compact,
            phase_v_compact=trace_outputs.v_compact,
            phase_controller_hidden_output_max_abs=trace_outputs.controller_hidden_output_max_abs,
            phase_action_output_max_abs=trace_outputs.action_output_max_abs,
            phase_post_positions=trace_outputs.post_positions,
            phase_post_velocities=trace_outputs.post_velocities,
            phase_post_velocities_true=trace_outputs.post_velocities_true,
            phase_post_spring_rest_length=trace_outputs.post_spring_rest_length,
            phase_last_substep_post_integrator_velocities=trace_outputs.last_substep_post_integrator_velocities,
            phase_last_substep_final_velocities=trace_outputs.last_substep_final_velocities,
            phase_last_substep_final_velocities_true=trace_outputs.last_substep_final_velocities_true,
            phase_done_before=trace_outputs.done_before,
            phase_done_after=trace_outputs.done_after,
        )
        return episode_result, phase_trace
    if capture_mode == "controller_trace":
        assert trace_outputs is not None
        return episode_result, trace_outputs
    if capture_substep_probe:
        return episode_result, final_carry.substep_trace
    return episode_result, trace_outputs


def run_episode(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> EpisodeResult:
    """Run a full episode with phased sense-act-physics interleaving."""
    result, _ = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="none",
        **kwargs,
    )
    return result


def run_episode_with_trace(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> tuple[EpisodeResult, RenderStepOutput]:
    """Like run_episode but also captures per-step positions for rendering."""
    result, trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="render",
        **kwargs,
    )
    assert trace is not None
    return result, trace


def run_episode_with_state_trace(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> tuple[EpisodeResult, EpisodeStateTrace]:
    """Like run_episode but captures full robot-local state at every env step."""
    result, state_trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="state_trace",
        **kwargs,
    )
    assert state_trace is not None
    return result, state_trace


def run_episode_with_com_trace(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> tuple[EpisodeResult, EpisodeComTrace]:
    """Like run_episode but captures only per-step COM-x for gait analysis."""
    result, com_trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="com_trace",
        **kwargs,
    )
    assert com_trace is not None
    return result, com_trace


def run_episode_with_controller_trace(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> tuple[EpisodeResult, EpisodeControllerTrace]:
    """Like run_episode but captures full controller-step state for PCA."""
    result, controller_trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="controller_trace",
        **kwargs,
    )
    assert controller_trace is not None
    return result, controller_trace


def run_episode_with_state_space_trace(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> tuple[EpisodeResult, EpisodeStateSpaceTrace]:
    """Like run_episode but captures env-step plant/controller state for offline analysis."""
    result, state_space_trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="state_space_trace",
        **kwargs,
    )
    assert state_space_trace is not None
    return result, state_space_trace


def run_episode_with_phase_probe(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    **kwargs,
) -> tuple[EpisodeResult, EpisodePhaseProbeTrace]:
    """Like run_episode but also captures per-phase controller and post-physics state."""
    result, phase_trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="phase_probe",
        **kwargs,
    )
    assert phase_trace is not None
    return result, phase_trace


def run_episode_with_substep_probe(
    env_data: CliffEnvData | CliffEvalEnvData,
    params: CTRNNParams,
    alive_mask: jnp.ndarray,
    key: jax.Array,
    *,
    substep_probe_step_index: int,
    substep_probe_phase_index: int,
    **kwargs,
) -> tuple[EpisodeResult, EpisodeSubstepProbeTrace]:
    """Like run_episode but captures one selected phase's per-substep physics diagnostics."""
    result, substep_trace = _run_episode_impl(
        env_data,
        params,
        alive_mask,
        key,
        capture_mode="substep_probe",
        substep_probe_step_index=substep_probe_step_index,
        substep_probe_phase_index=substep_probe_phase_index,
        **kwargs,
    )
    assert substep_trace is not None
    return result, substep_trace
