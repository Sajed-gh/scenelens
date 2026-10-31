"""Fast deterministic checks; no model downloads or GUI required."""
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from scenelens.evaluation.depth import depth_metrics, summarize
from scenelens.analysis.distance import masked_distance
from scenelens.analysis.objects import SceneObject
from scenelens.models.segmenter import Segmenter
from scenelens.rendering.renderer import extract_object_crop, compose_sidebar, sidebar_objects
from scenelens import config


class ComponentTests(unittest.TestCase):
    def test_batched_mask_prompts(self):
        segmenter = Segmenter.__new__(Segmenter)
        segmenter.batch_prompts = True
        segmenter.predictor = Mock()
        segmenter.predictor.device = 'cpu'
        segmenter.predictor.transform.apply_boxes_torch.side_effect = lambda boxes, shape: boxes
        masks = torch.zeros((2,1,80,120),dtype=torch.bool)
        masks[0,0,10:30,10:30] = True
        masks[1,0,40:60,40:60] = True
        segmenter.predictor.predict_torch.return_value = (masks,None,None)
        objects = [SceneObject(1,'person',.9,(10,10,30,30)),
                   SceneObject(2,'chair',.8,(40,40,60,60))]
        segmenter.segment(np.zeros((80,120,3),np.uint8),objects,3,.5)
        segmenter.predictor.set_image.assert_called_once()
        segmenter.predictor.predict_torch.assert_called_once()
        segmenter.predictor.predict.assert_not_called()
        self.assertTrue(objects[0].mask[10:30,10:30].all())
        self.assertTrue(objects[1].mask[40:60,40:60].all())



    def test_fusion_ignores_background_and_invalid_depth(self):
        depth = np.full((10,10),99.,np.float32)
        mask = np.zeros((10,10),bool)
        mask[2:8,2:8] = True
        depth[mask] = 3.
        depth[2,2:5] = [np.nan,0,1000]
        self.assertEqual(masked_distance(depth,mask,20),3.)
        self.assertIsNone(masked_distance(depth,mask,35))


    def test_crop_and_sidebar(self):
        frame = np.zeros((360,640,3),np.uint8)
        mask = np.zeros(frame.shape[:2],bool)
        mask[20:120,20:70] = True
        frame[mask] = (10,100,200)
        crop = extract_object_crop(frame,mask,(160,160))
        self.assertEqual(crop.shape,(160,160,4))
        ys,xs=np.nonzero(crop[:,:,3])
        self.assertEqual((xs.max()-xs.min()+1,ys.max()-ys.min()+1),(80,160))
        self.assertTrue(np.all(crop[crop[:,:,3]==0,:3] == 0))
        objects=[SceneObject(1,'person',.9,(0,0,20,20),distance=4.),SceneObject(2,'chair',.8,(0,0,20,20),distance=2.,crop=crop),SceneObject(3,'bag',.7,(0,0,20,20))]
        self.assertEqual([o.object_id for o in sidebar_objects(objects,2)],[2,1])
        display=compose_sidebar(frame,objects)
        self.assertEqual(display.shape,(360,640+config.SIDEBAR_WIDTH,3))
        np.testing.assert_array_equal(display[:,:640],frame)

    def test_metric_evaluation(self):
        metrics=summarize(depth_metrics(np.array([[2.,4.],[3.,np.nan]]),np.array([[1.,2.],[0.,1.]]),20))
        self.assertEqual(metrics['pixels'],2)
        self.assertEqual(metrics['mae_m'],1.5)
        self.assertAlmostEqual(metrics['rmse_m'],np.sqrt(2.5))
        self.assertEqual(metrics['abs_rel'],1.)
        self.assertEqual(metrics['prediction_coverage'],2/3)



if __name__=='__main__':
    unittest.main()
