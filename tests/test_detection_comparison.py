import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scenelens.evaluation import detection
from scenelens.models.detector import Detection


class DetectionComparisonTests(unittest.TestCase):
    def test_unavailable_model_preserves_completed_results(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'input.png'
            cv2.imwrite(str(image),np.zeros((80,120,3),np.uint8))
            def load(weights,*args,**kwargs):
                if str(weights).endswith('yolo26m-seg.pt'):
                    raise RuntimeError('Checkpoint unavailable')
                return SimpleNamespace(device='cpu',detect=lambda image:[Detection((10,10,40,60),0,'person',.9)])
            with patch('sys.argv',['compare','--images',str(image),'--repeats','1','--output',str(root/'results')]),patch.object(detection,'Detector',side_effect=load):
                self.assertEqual(detection.main(),1)
            report=json.loads((root/'results/results.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual([v['status'] for v in report['variants']],['complete','complete','failed'])
            self.assertEqual(report['variants'][0]['images'][0]['counts'],{'person':1})
            self.assertIsNone(report['variants'][0]['images'][0]['peak_gpu_allocated_mb'])
            self.assertTrue((root/'results/01-input-comparison.png').exists())


if __name__=='__main__':
    unittest.main()
