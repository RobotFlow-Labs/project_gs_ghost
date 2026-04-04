"""Tests for grasp detection and HO alignment — PRD-0205."""

import torch
import pytest


class TestGraspDetection:
    def test_centroid_motion_shape(self):
        from anima_gs_ghost.alignment.grasp_detection import centroid_motion
        masks = torch.zeros(10, 64, 64, dtype=torch.bool)
        masks[:, 20:40, 20:40] = True
        motion = centroid_motion(masks)
        assert motion.shape == (9, 2)

    def test_grasp_score_identical_motion(self):
        from anima_gs_ghost.alignment.grasp_detection import grasp_score
        motion = torch.randn(9, 2)
        assert grasp_score(motion, motion) == pytest.approx(1.0, abs=1e-5)

    def test_grasp_score_opposite_motion(self):
        from anima_gs_ghost.alignment.grasp_detection import grasp_score
        motion = torch.randn(9, 2)
        assert grasp_score(motion, -motion) == pytest.approx(-1.0, abs=1e-5)

    def test_grasp_score_zero_motion(self):
        from anima_gs_ghost.alignment.grasp_detection import grasp_score
        zero = torch.zeros(9, 2)
        nonzero = torch.randn(9, 2)
        assert grasp_score(zero, nonzero) == 0.0

    def test_detect_grasping_hands(self):
        from anima_gs_ghost.alignment.grasp_detection import detect_grasping_hands
        T, H, W = 10, 32, 32
        obj = torch.zeros(T, H, W, dtype=torch.bool)
        hands = torch.zeros(T, 2, H, W, dtype=torch.bool)
        # Object and hand 0 move together
        for t in range(T):
            obj[t, 10 + t : 20 + t, 10:20] = True
            hands[t, 0, 10 + t : 20 + t, 10:20] = True
            hands[t, 1, 10:20, 10:20] = True  # Hand 1 is stationary
        result = detect_grasping_hands(obj, hands, tau_sim=0.3)
        assert 0 in result


class TestHOAlignment:
    def test_contact_loss_zero_when_touching(self):
        from anima_gs_ghost.alignment.ho_alignment import contact_loss
        # Joints exactly on object points
        joints = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        obj = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        idx = torch.tensor([0, 1])
        loss = contact_loss(joints, obj, idx)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_temporal_smoothness_constant(self):
        from anima_gs_ghost.alignment.ho_alignment import temporal_smoothness_loss
        # Constant translations → zero acceleration → zero loss
        transl = torch.ones(10, 1, 3) * 0.5
        loss = temporal_smoothness_loss(transl)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_total_alignment_loss_weights(self):
        from anima_gs_ghost.alignment.ho_alignment import total_alignment_loss
        lc = torch.tensor(1.0)
        lp = torch.tensor(1.0)
        lt = torch.tensor(1.0)
        total = total_alignment_loss(lc, lp, lt)
        # 1e3 * 1 + 1e-1 * 1 + 10 * 1 = 1010.1
        assert float(total) == pytest.approx(1010.1, rel=1e-3)
