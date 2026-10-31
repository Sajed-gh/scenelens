"""Exercise the single-image workflow without downloading models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scenelens import config
from scenelens import pipeline as main
from scenelens.models.detector import Detection


class ImagePipelineTests(unittest.TestCase):
    def run_image(self, root, detections):
        source = root / 'photo.png'
        output = root / 'result.png'
        image = np.full((100,160,3),80,np.uint8)
        cv2.imwrite(str(source),image)
        detector = SimpleNamespace(device='cpu',detect=lambda frame: detections)
        def segment(frame,objects,*args):
            for obj in objects:
                obj.mask=np.zeros(frame.shape[:2],bool)
                x1,y1,x2,y2=map(int,obj.bbox)
                obj.mask[y1:y2,x1:x2]=True
        depth=SimpleNamespace(estimate=lambda *args:np.full((100,160),3.,np.float32))
        with patch.object(config,'IMAGE_PATH',source), patch.object(config,'OUTPUT_IMAGE',output), \
             patch.object(config,'HEADLESS',True), patch.object(config,'SAM_ENABLED',True), \
             patch.object(config,'DEPTH_ENABLED',True), patch.object(main,'Detector',return_value=detector), \
             patch.object(main,'Segmenter',return_value=SimpleNamespace(segment=segment)), \
             patch.object(main,'DepthEstimator',return_value=depth):
            report=main.run()
        np.testing.assert_array_equal(cv2.imread(str(source)),image)
        self.assertTrue(output.exists())
        self.assertTrue(output.with_suffix('.json').exists())
        self.assertNotIn('motion_state',json.dumps(report))
        self.assertNotIn('fps',json.dumps(report).lower())
        self.assertEqual(report['analyzed_size'],[160,100])
        return report

    def test_image_objects_and_exports(self):
        with TemporaryDirectory() as directory:
            report=self.run_image(Path(directory),[Detection((10,10,40,70),0,'person',.9)])
            self.assertEqual(report['objects'][0]['object_id'],1)
            self.assertEqual(report['objects'][0]['distance_m'],3.)
            crop=cv2.imread(report['objects'][0]['crop'],cv2.IMREAD_UNCHANGED)
            self.assertEqual(crop.shape[2],4)
            self.assertTrue((crop[:,:,3]==0).any())

    def test_empty_image_results(self):
        with TemporaryDirectory() as directory:
            report=self.run_image(Path(directory),[])
            self.assertEqual(report['objects'],[])

    def test_missing_image_fails_before_models_load(self):
        with patch.object(config,'IMAGE_PATH',Path('/tmp/scenelens-missing-input.png')), \
             patch.object(main,'Detector') as detector:
            with self.assertRaisesRegex(RuntimeError,'Cannot read image'):
                main.run()
            detector.assert_not_called()


if __name__=='__main__':
    unittest.main()
