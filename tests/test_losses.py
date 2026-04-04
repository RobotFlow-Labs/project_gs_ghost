"""Tests for reconstruction losses — PRD-0206."""

import torch
import pytest


class TestRGBLoss:
    def test_identical_images_zero_loss(self):
        from anima_gs_ghost.reconstruction.losses import rgb_loss
        img = torch.rand(1, 3, 64, 64)
        loss = rgb_loss(img, img)
        assert float(loss) < 1e-5

    def test_different_images_positive_loss(self):
        from anima_gs_ghost.reconstruction.losses import rgb_loss
        a = torch.zeros(1, 3, 64, 64)
        b = torch.ones(1, 3, 64, 64)
        loss = rgb_loss(a, b)
        assert float(loss) > 0.5


class TestBackgroundHandLoss:
    def test_no_overlap_zero_loss(self):
        from anima_gs_ghost.reconstruction.losses import background_hand_loss
        alpha = torch.ones(1, 1, 32, 32)
        hand = torch.zeros(1, 1, 32, 32)
        loss = background_hand_loss(alpha, hand)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_full_overlap_positive_loss(self):
        from anima_gs_ghost.reconstruction.losses import background_hand_loss
        alpha = torch.ones(1, 1, 32, 32)
        hand = torch.ones(1, 1, 32, 32)
        loss = background_hand_loss(alpha, hand)
        assert float(loss) > 0


class TestGeometricConsistencyLoss:
    def test_coincident_points_near_zero(self):
        from anima_gs_ghost.reconstruction.losses import geometric_consistency_loss
        pts = torch.randn(100, 3)
        loss = geometric_consistency_loss(pts, pts, tau_out=0.05, tau_fill=0.005)
        assert float(loss) < 0.1

    def test_distant_points_high_loss(self):
        from anima_gs_ghost.reconstruction.losses import geometric_consistency_loss
        g = torch.randn(50, 3)
        p = torch.randn(50, 3) + 10.0  # Far away
        loss = geometric_consistency_loss(g, p)
        assert float(loss) > 1.0


class TestCombinedLoss:
    def test_combined_loss_returns_all_keys(self):
        from anima_gs_ghost.reconstruction.losses import combined_object_loss
        pred = torch.rand(1, 3, 32, 32)
        gt = torch.rand(1, 3, 32, 32)
        alpha = torch.rand(1, 1, 32, 32)
        hmask = torch.zeros(1, 1, 32, 32)
        gc = torch.randn(20, 3)
        pp = torch.randn(20, 3)
        losses = combined_object_loss(pred, gt, alpha, hmask, gc, pp)
        assert set(losses.keys()) == {"total", "rgb", "bkg_hand", "geo"}
        assert float(losses["total"]) > 0
