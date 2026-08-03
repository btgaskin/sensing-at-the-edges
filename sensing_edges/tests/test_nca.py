"""Tests for NCA-CTRNN controller."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sensing_edges.nca import (
    ACTION_H_IDX,
    CTRNNParams,
    base_channel_count,
    clamp_growth_channels,
    controller_flat_dim,
    ctrnn_flat_to_params,
    ctrnn_params_to_flat,
    grid_channel_count,
    init_ctrnn_params,
    nca_step,
    nca_step_with_diagnostics,
    normalize_active_comm_filters,
    perceive,
    perception_dim,
    perception_layout,
    run_growth_phase,
)


class TestDimensions:
    """Verify param counts match the plan."""

    def test_default_ou_on(self):
        layout = perception_layout(12, "off", ou_enabled=True)
        assert grid_channel_count(12, "off", ou_enabled=True) == 19
        assert layout.n_private == 7
        assert layout.n_comm == 12
        assert perception_dim(layout.n_private, layout.n_comm) == 55
        assert controller_flat_dim(8, 12, "off", ou_enabled=True) == 632

    def test_default_ou_off(self):
        layout = perception_layout(12, "off", ou_enabled=False)
        assert grid_channel_count(12, "off", ou_enabled=False) == 18
        assert layout.n_private == 6
        assert layout.n_comm == 12
        assert perception_dim(layout.n_private, layout.n_comm) == 54
        assert controller_flat_dim(8, 12, "off", ou_enabled=False) == 624

    def test_gasnet_single_counts(self):
        layout = perception_layout(12, "single", ou_enabled=True)
        assert grid_channel_count(12, "single", ou_enabled=True) == 21
        assert layout.n_private == 8
        assert layout.n_comm == 13
        assert perception_dim(layout.n_private, layout.n_comm) == 60
        assert controller_flat_dim(8, 12, "single", ou_enabled=True) == 680

    def test_gasnet_dual_counts(self):
        layout = perception_layout(12, "dual", ou_enabled=True)
        assert grid_channel_count(12, "dual", ou_enabled=True) == 23
        assert layout.n_private == 9
        assert layout.n_comm == 14
        assert perception_dim(layout.n_private, layout.n_comm) == 65
        assert controller_flat_dim(8, 12, "dual", ou_enabled=True) == 728

    def test_growth_counts(self):
        layout = perception_layout(12, "off", ou_enabled=True, growth_enabled=True)
        assert grid_channel_count(12, "off", ou_enabled=True, growth_enabled=True) == 21
        assert layout.n_private == 8
        assert layout.n_comm == 13
        assert perception_dim(layout.n_private, layout.n_comm) == 60
        assert controller_flat_dim(8, 12, "off", ou_enabled=True, growth_enabled=True) == 680

    def test_laplacian_identity_counts(self):
        layout = perception_layout(12, "off", ou_enabled=True)
        active_comm_filters = ("laplacian", "identity")
        assert perception_dim(
            layout.n_private,
            layout.n_comm,
            active_comm_filters=active_comm_filters,
        ) == 31
        assert controller_flat_dim(
            8,
            12,
            "off",
            ou_enabled=True,
            active_comm_filters=active_comm_filters,
        ) == 440

    def test_active_comm_filters_normalize_to_canonical_order(self):
        assert normalize_active_comm_filters(("identity", "laplacian")) == (
            "laplacian",
            "identity",
        )


class TestPerceptionSplit:
    def test_private_channels_use_identity_only(self):
        layout = perception_layout(2, "off", ou_enabled=True)
        state = jnp.zeros((3, 3, grid_channel_count(2, "off", True)), dtype=jnp.float32)
        state = state.at[1, 1, ACTION_H_IDX].set(1.0)
        alive_mask = jnp.ones((3, 3), dtype=jnp.bool_)

        perceived = perceive(
            state,
            alive_mask,
            private_channel_indices=layout.private_channel_indices,
            comm_channel_indices=layout.comm_channel_indices,
        )

        assert perceived.shape == (3, 3, perception_dim(layout.n_private, layout.n_comm))
        assert perceived[1, 1, ACTION_H_IDX] == pytest.approx(1.0)
        assert jnp.allclose(perceived[:, :, layout.n_private :], 0.0)

    def test_comm_channels_receive_filter_bank(self):
        layout = perception_layout(2, "off", ou_enabled=True)
        gh_start = base_channel_count(ou_enabled=True)
        state = jnp.zeros((3, 3, grid_channel_count(2, "off", True)), dtype=jnp.float32)
        state = state.at[1, 1, gh_start].set(1.0)
        alive_mask = jnp.ones((3, 3), dtype=jnp.bool_)

        perceived = perceive(
            state,
            alive_mask,
            private_channel_indices=layout.private_channel_indices,
            comm_channel_indices=layout.comm_channel_indices,
        )

        assert jnp.allclose(perceived[:, :, : layout.n_private], 0.0)
        assert float(jnp.sum(jnp.abs(perceived[:, :, layout.n_private :]))) > 0.0

    def test_comm_channels_respect_active_filter_subset(self):
        layout = perception_layout(2, "off", ou_enabled=True)
        gh_start = base_channel_count(ou_enabled=True)
        state = jnp.zeros((3, 3, grid_channel_count(2, "off", True)), dtype=jnp.float32)
        state = state.at[1, 1, gh_start].set(1.0)
        alive_mask = jnp.ones((3, 3), dtype=jnp.bool_)

        perceived = perceive(
            state,
            alive_mask,
            private_channel_indices=layout.private_channel_indices,
            comm_channel_indices=layout.comm_channel_indices,
            active_comm_filters=("laplacian", "identity"),
        )

        assert perceived.shape == (3, 3, layout.n_private + 2 * layout.n_comm)


class TestParamRoundtrip:
    """Flatten → unflatten preserves values."""

    def test_roundtrip(self):
        key = jax.random.PRNGKey(42)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12)
        flat = ctrnn_params_to_flat(params)
        assert flat.shape == (632,)
        restored = ctrnn_flat_to_params(flat, 8, 12)
        np.testing.assert_allclose(np.array(restored.w_input), np.array(params.w_input), atol=1e-6)
        np.testing.assert_allclose(np.array(restored.w_rec), np.array(params.w_rec), atol=1e-6)
        np.testing.assert_allclose(np.array(restored.w_output), np.array(params.w_output), atol=1e-6)
        np.testing.assert_allclose(np.array(restored.bias), np.array(params.bias), atol=1e-6)
        np.testing.assert_allclose(np.array(restored.log_tau), np.array(params.log_tau), atol=1e-6)

    def test_roundtrip_gasnet_single(self):
        key = jax.random.PRNGKey(99)
        params = init_ctrnn_params(key, ctrnn_hidden=8, ctrnn_grid_hidden=12, gasnet_mode="single")
        flat = ctrnn_params_to_flat(params)
        assert flat.shape == (680,)
        restored = ctrnn_flat_to_params(flat, 8, 12, gasnet_mode="single")
        np.testing.assert_allclose(np.array(restored.w_input), np.array(params.w_input), atol=1e-6)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Flat length mismatch"):
            ctrnn_flat_to_params(jnp.zeros(100), 8, 12)


class TestNCAStep:
    """Integration test for a single NCA step."""

    def _make_test_inputs(self, ctrnn_hidden=8, ctrnn_grid_hidden=12, ou_enabled=True, gasnet_mode="off"):
        key = jax.random.PRNGKey(0)
        canvas_h, canvas_w = 2, 7
        gc = grid_channel_count(ctrnn_grid_hidden, gasnet_mode, ou_enabled)
        nca_state = jnp.zeros((canvas_h, canvas_w, gc))
        ctrnn_carry = jnp.zeros((canvas_h, canvas_w, ctrnn_hidden))
        alive_mask = jnp.ones((canvas_h, canvas_w), dtype=jnp.bool_)
        grid_h, grid_w = 10, 35
        deformation = jnp.ones((grid_h * grid_w, 4))
        deform_mask = jnp.zeros(grid_h * grid_w, dtype=jnp.bool_)
        actuator_mask = jnp.zeros(grid_h * grid_w, dtype=jnp.bool_)
        params = init_ctrnn_params(
            key,
            ctrnn_hidden,
            ctrnn_grid_hidden,
            gasnet_mode=gasnet_mode,
            ou_enabled=ou_enabled,
        )
        return dict(
            nca_state=nca_state,
            ctrnn_carry=ctrnn_carry,
            deformation=deformation,
            deform_mask=deform_mask,
            actuator_mask=actuator_mask,
            alive_mask=alive_mask,
            params=params,
            key=key,
            robot_offset_y=3,
            robot_offset_x=1,
            h=grid_h,
            w=grid_w,
            ctrnn_grid_hidden=ctrnn_grid_hidden,
            ou_enabled=ou_enabled,
            gasnet_mode=gasnet_mode,
        )

    def test_step_runs(self):
        inputs = self._make_test_inputs()
        new_state, new_carry, h_act, v_act = nca_step(**inputs)
        assert new_state.shape == inputs["nca_state"].shape
        assert new_carry.shape == inputs["ctrnn_carry"].shape
        assert h_act.shape == (10 * 35,)
        assert not jnp.any(jnp.isnan(new_state))

    def test_ablation_zeros_deformation(self):
        inputs = self._make_test_inputs()
        inputs["nca_state"] = inputs["nca_state"].at[:, :, :4].set(1.0)
        inputs["ablate_deformation"] = True
        new_state, _, _, _ = nca_step(**inputs)
        assert jnp.allclose(new_state[:, :, :4], 0.0)

    def test_ou_disabled(self):
        inputs = self._make_test_inputs(ou_enabled=False)
        new_state, _, _, _ = nca_step(**inputs)
        assert new_state.shape[-1] == 18

    def test_gasnet_single_step_runs(self):
        inputs = self._make_test_inputs(gasnet_mode="single")
        new_state, new_carry, _, _ = nca_step(**inputs)
        assert new_state.shape[-1] == 21
        assert new_carry.shape == inputs["ctrnn_carry"].shape

    def test_deform_on_vs_off_differ(self):
        inputs_on = self._make_test_inputs()
        inputs_on["deform_mask"] = jnp.ones(10 * 35, dtype=jnp.bool_)
        inputs_on["deformation"] = jnp.ones((10 * 35, 4)) * 1.5
        inputs_on["ablate_deformation"] = False

        inputs_off = {**inputs_on, "ablate_deformation": True}

        _, _, h_on, _ = nca_step(**inputs_on)
        _, _, h_off, _ = nca_step(**inputs_off)
        assert h_on.shape == h_off.shape

    def test_equal_params_deform_on_vs_off(self):
        dim_on = controller_flat_dim(8, 12, "off", ou_enabled=True)
        dim_off = controller_flat_dim(8, 12, "off", ou_enabled=True)
        assert dim_on == dim_off == 632

    def test_signed_deformation_channels_use_symmetric_log_scaling(self):
        inputs = self._make_test_inputs(ou_enabled=False)
        inputs["deform_mask"] = jnp.ones(10 * 35, dtype=jnp.bool_)
        inputs["deformation"] = jnp.tile(
            jnp.asarray([[1.2, 0.8, 1.1, 0.95]], dtype=jnp.float32),
            (10 * 35, 1),
        )
        inputs["deformation_gain_absolute"] = 10.0
        inputs["deformation_gain_compliance"] = 100.0
        new_state, _, _, _ = nca_step(**inputs)
        observed = np.asarray(new_state[0, 0, :4], dtype=np.float32)
        expected = np.asarray(
            [
                np.log1p(10.0 * 0.2),
                -np.log1p(10.0 * 0.2),
                np.log1p(100.0 * 0.1),
                -np.log1p(100.0 * 0.05),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observed, expected, atol=1e-6)

    def test_mace_cross_channel_uses_seeded_mass(self):
        inputs = self._make_test_inputs(ctrnn_hidden=8, ctrnn_grid_hidden=2)
        layout = perception_layout(2, "off", ou_enabled=True)
        gh_start = base_channel_count(ou_enabled=True)
        inputs["nca_state"] = inputs["nca_state"].at[:, :, gh_start : gh_start + 2].set(1.0)
        inputs["params"] = CTRNNParams(
            w_input=jnp.zeros(
                (perception_dim(layout.n_private, layout.n_comm), 8),
                dtype=jnp.float32,
            ),
            w_rec=jnp.zeros((8, 8), dtype=jnp.float32),
            w_output=jnp.ones((8, 4), dtype=jnp.float32),
            bias=jnp.ones((8,), dtype=jnp.float32) * 4.0,
            log_tau=jnp.zeros((8,), dtype=jnp.float32),
        )
        inputs["p_update"] = 1.0
        new_state, new_carry, _, _ = nca_step(**inputs, mace_mode="cross_channel")
        assert float(new_carry.mean()) > 0.0
        assert float(new_state[:, :, gh_start : gh_start + 2].mean()) > 0.0

    def test_diagnostics_use_pre_mace_hidden_outputs(self):
        inputs = self._make_test_inputs(ctrnn_hidden=1, ctrnn_grid_hidden=1)
        layout = perception_layout(1, "off", ou_enabled=True)
        gh_start = base_channel_count(ou_enabled=True)
        inputs["nca_state"] = inputs["nca_state"].at[:, :, gh_start].set(1.0)
        inputs["params"] = CTRNNParams(
            w_input=jnp.zeros(
                (perception_dim(layout.n_private, layout.n_comm), 1),
                dtype=jnp.float32,
            ),
            w_rec=jnp.zeros((1, 1), dtype=jnp.float32),
            w_output=jnp.asarray([[0.0, 0.0, 0.5]], dtype=jnp.float32),
            bias=jnp.asarray([4.0], dtype=jnp.float32),
            log_tau=jnp.zeros((1,), dtype=jnp.float32),
        )
        inputs["p_update"] = 1.0
        new_state, _, _, _, diagnostics = nca_step_with_diagnostics(
            **inputs,
            mace_mode="cross_channel",
        )
        expected_hidden_output = np.tanh(4.0 * (1.0 - np.exp(-1.0 / 5.05))) * 0.5
        assert float(diagnostics.controller_hidden_output_max_abs) == pytest.approx(
            expected_hidden_output,
            rel=1e-4,
        )
        assert float(new_state[0, 0, gh_start]) > float(diagnostics.controller_hidden_output_max_abs)

    def test_neutral_action_maps_to_rest_length(self):
        inputs = self._make_test_inputs()
        layout = perception_layout(12, "off", ou_enabled=True)
        inputs["actuator_mask"] = jnp.ones(10 * 35, dtype=jnp.bool_)
        inputs["params"] = CTRNNParams(
            w_input=jnp.zeros((perception_dim(layout.n_private, layout.n_comm), 8), dtype=jnp.float32),
            w_rec=jnp.zeros((8, 8), dtype=jnp.float32),
            w_output=jnp.zeros((8, 14), dtype=jnp.float32),
            bias=jnp.zeros((8,), dtype=jnp.float32),
            log_tau=jnp.zeros((8,), dtype=jnp.float32),
        )
        inputs["p_update"] = 1.0
        _, _, h_act, v_act = nca_step(**inputs)
        assert jnp.allclose(h_act, 1.0)
        assert jnp.allclose(v_act, 1.0)


class TestGrowthPhase:
    def test_growth_phase_preserves_seed(self):
        params = init_ctrnn_params(
            jax.random.PRNGKey(0),
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            growth_enabled=True,
        )
        alive_mask, frozen_identity, final_state, final_carry = run_growth_phase(
            params,
            jax.random.PRNGKey(1),
            growth_steps=3,
            canvas_h=4,
            canvas_w=10,
            ctrnn_hidden=8,
            ctrnn_grid_hidden=12,
            seed_y=2,
            seed_x=5,
        )
        assert alive_mask.shape == (4, 10)
        assert frozen_identity.shape == (4, 10)
        assert final_state.shape == (4, 10, 21)
        assert final_carry.shape == (4, 10, 8)
        assert bool(alive_mask[2, 5])
        assert float(frozen_identity[2, 5]) == pytest.approx(1.0)

    def test_clamp_growth_channels_freezes_identity_and_zeroes_signal(self):
        state = jnp.ones((4, 10, 21), dtype=jnp.float32)
        frozen_identity = jnp.zeros((4, 10), dtype=jnp.float32).at[2, 5].set(1.0)
        clamped = clamp_growth_channels(
            state,
            frozen_identity,
            ctrnn_grid_hidden=12,
            gasnet_mode="off",
            ou_enabled=True,
        )
        assert jnp.allclose(clamped[:, :, 19], frozen_identity)
        assert jnp.allclose(clamped[:, :, 20], 0.0)
