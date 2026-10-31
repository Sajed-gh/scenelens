"""Fast deterministic checks; no model downloads or GUI required."""
import unittest
from unittest.mock import patch

import cv2

import numpy as np

from scenelens.evaluation.depth import depth_metrics, summarize
from scenelens.analysis.distance import masked_distance
from scenelens.analysis.objects import SceneObject
from scenelens.rendering.renderer import extract_object_crop, compose_sidebar, sidebar_objects
from scenelens import config


class ComponentTests(unittest.TestCase):
    def test_gallery_includes_all_objects_beyond_old_capacity(self):
        image=np.zeros((120,160,3),np.uint8)
        crop=np.zeros((160,160,4),np.uint8)
        crop[20:140,20:140]=(20,200,40,255)
        objects=[SceneObject(i,'chair',.9,(0,0,20,20),crop=crop,
                             distance=float(13-i) if i%2 else None) for i in range(1,13)]
        with patch.object(cv2,'putText',wraps=cv2.putText) as text:
            display=compose_sidebar(image,objects)
        titles=[call.args[1] for call in text.call_args_list]
        for i in range(1,13):
            self.assertIn(f'Chair #{i}  90%',titles)
        self.assertEqual([o.object_id for o in sidebar_objects(objects)],list(range(1,13)))
        self.assertGreater(display.shape[0],image.shape[0])
        self.assertGreater(display.shape[1],image.shape[1])
        np.testing.assert_array_equal(display[:120,:160],image)

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
        self.assertEqual([o.object_id for o in sidebar_objects(objects)],[1,2,3])
        display=compose_sidebar(frame,objects)
        self.assertGreaterEqual(display.shape[0],360)
        self.assertGreaterEqual(display.shape[1],640+config.SIDEBAR_WIDTH)
        np.testing.assert_array_equal(display[:360,:640],frame)

    def test_metric_evaluation(self):
        metrics=summarize(depth_metrics(np.array([[2.,4.],[3.,np.nan]]),np.array([[1.,2.],[0.,1.]]),20))
        self.assertEqual(metrics['pixels'],2)
        self.assertEqual(metrics['mae_m'],1.5)
        self.assertAlmostEqual(metrics['rmse_m'],np.sqrt(2.5))
        self.assertEqual(metrics['abs_rel'],1.)
        self.assertEqual(metrics['prediction_coverage'],2/3)



if __name__=='__main__':
    unittest.main()
