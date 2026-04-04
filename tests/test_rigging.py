"""Tests for hand Gaussian rigging — PRD-0206."""

import torch


class TestHandModel:
    def test_simple_hand_model_forward(self):
        from anima_gs_ghost.hand_model import SimpleHandModel
        model = SimpleHandModel(device="cpu")
        pose = torch.zeros(2, 20)
        shape = torch.zeros(2, 10)
        out = model(pose, shape)
        assert out.vertices.shape[0] == 2
        assert out.joints.shape == (2, 21, 3)
        assert out.faces.dim() == 2

    def test_simple_hand_translation(self):
        from anima_gs_ghost.hand_model import SimpleHandModel
        model = SimpleHandModel(device="cpu")
        pose = torch.zeros(1, 20)
        shape = torch.zeros(1, 10)
        transl = torch.tensor([[1.0, 2.0, 3.0]])
        out = model(pose, shape, transl=transl)
        # All joints should be shifted
        out_no_t = model(pose, shape)
        diff = out.joints - out_no_t.joints
        assert torch.allclose(diff, transl.unsqueeze(1).expand_as(diff), atol=1e-5)


class TestHandGaussianRig:
    def test_barycentric_sampling(self):
        from anima_gs_ghost.reconstruction.hand_gs import sample_face_barycentrics
        bary = sample_face_barycentrics(n_per_edge=5)
        # Sum of barycentrics should be ~1
        assert torch.allclose(bary.sum(dim=-1), torch.ones(bary.shape[0]), atol=1e-5)
        assert bary.shape[1] == 3

    def test_rig_output_shapes(self):
        from anima_gs_ghost.hand_model import SimpleHandModel
        from anima_gs_ghost.reconstruction.hand_gs import HandGaussianRig
        hand = SimpleHandModel(device="cpu")
        pose = torch.zeros(1, 20)
        shape = torch.zeros(1, 10)
        hand_out = hand(pose, shape)
        n_faces = hand_out.faces.shape[0]
        rig = HandGaussianRig(n_faces=n_faces, gaussians_per_edge=3, device="cpu")
        result = rig(hand_out, frame_idx=0)
        assert result["xyz"].shape[0] == rig.n_gaussians
        assert result["xyz"].shape[1] == 3
        assert result["opacity"].shape == (rig.n_gaussians, 1)

    def test_deformation_identity(self):
        from anima_gs_ghost.reconstruction.hand_gs import HandGaussianRig
        n_faces = 10
        rig = HandGaussianRig(n_faces=n_faces, gaussians_per_edge=2, device="cpu")
        canonical = torch.randn(rig.n_gaussians, 3)
        # Identity transform
        identity = torch.zeros(n_faces, 3, 4)
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        identity[:, 2, 2] = 1
        deformed = rig.deform(canonical, identity)
        assert torch.allclose(canonical, deformed, atol=1e-5)
