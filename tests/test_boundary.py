import unittest
import numpy as np
import torch
from ournet.data import boundary_distance_target
from ournet.loss import BDLoss, OurNetLoss


class BoundaryTests(unittest.TestCase):
    def test_gaussian_distance_and_occlusion_interface(self):
        label = np.zeros((17, 17), dtype=np.uint8)
        label[4:13, 4:13] = 1
        label[4:13, 9:13] = 2
        target = boundary_distance_target(label, sigma=3)
        self.assertEqual(target[4, 8], 1.)
        self.assertAlmostEqual(float(target[3, 8]), np.exp(-1/18), places=6)
        self.assertLess(target[2, 8], target[3, 8])
        self.assertLess(target[8, 8], target[4, 8])
        np.testing.assert_array_equal(target, boundary_distance_target(label > 0, 3))
        self.assertTrue(np.isfinite(target).all())

    def test_empty_and_full_masks(self):
        for value in (0, 1):
            self.assertFalse(boundary_distance_target(np.full((8, 8), value)).any())
        for sigma in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                boundary_distance_target(np.zeros((8, 8)), sigma)

    def test_boundary_loss(self):
        truth = torch.from_numpy(boundary_distance_target(np.eye(8))).reshape(1, 1, 8, 8)
        self.assertEqual(BDLoss()(truth, truth).item(), 0.)
        prediction = torch.zeros_like(truth, requires_grad=True)
        loss = BDLoss()(prediction, truth)
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertGreater(prediction.grad.abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            BDLoss()(truth, truth.squeeze())

    def test_multitask_loss_for_both_datasets(self):
        for classes in (3, 5):
            pred = {'roadocc_masks': torch.randn(2, classes, 8, 8, requires_grad=True),
                    'roadwo_masks': torch.full((2, 1, 8, 8), .5, requires_grad=True),
                    'roadwo_edge_masks': torch.full((2, 1, 8, 8), .5, requires_grad=True)}
            loss = OurNetLoss(num_classes=classes)(pred, {'label': torch.zeros(2, 1, 8, 8),
                                                          'dist_maps': torch.zeros(2, 1, 8, 8)})
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in pred.values()))


if __name__ == '__main__':
    unittest.main()
