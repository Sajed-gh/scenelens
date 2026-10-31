from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from scenelens.models.detector import Detector


class YoloSegmentationTests(unittest.TestCase):
    def detector(self, image, boxes, masks):
        model=Detector.__new__(Detector)
        model.model=Mock()
        model.model.predict.return_value=[Results(image,'sample.png',{0:'person',56:'chair'},boxes=boxes,masks=masks)]
        model.device='cpu'
        model.confidence=.5
        model.image_size=960
        model.half=False
        model.require_masks=True
        return model

    def test_original_image_masks_and_detection_order(self):
        image=np.zeros((80,120,3),np.uint8)
        boxes=torch.tensor([[10,10,30,60,.9,0],[60,20,100,70,.8,56]])
        masks=torch.zeros((2,80,120))
        masks[0,10:60,10:30]=1
        masks[1,20:70,60:100]=1
        model=self.detector(image,boxes,masks)
        results=model.detect(image)
        self.assertEqual([d.label for d in results],['person','chair'])
        self.assertEqual(results[0].mask.dtype,np.bool_)
        self.assertEqual(results[0].mask.shape,(80,120))
        self.assertTrue(results[0].mask[10:60,10:30].all())
        self.assertTrue(results[1].mask[20:70,60:100].all())
        self.assertTrue(model.model.predict.call_args.kwargs['retina_masks'])

    def test_missing_and_unaligned_masks_rejected(self):
        image=np.zeros((80,120,3),np.uint8)
        boxes=torch.tensor([[10,10,30,60,.9,0]])
        model=self.detector(image,boxes,None)
        with self.assertRaisesRegex(RuntimeError,'without instance masks'):
            model.detect(image)
        model=self.detector(image,boxes,torch.ones((1,40,60)))
        with self.assertRaisesRegex(RuntimeError,'input image dimensions'):
            model.detect(image)

    def test_empty_results(self):
        image=np.zeros((80,120,3),np.uint8)
        self.assertEqual(self.detector(image,torch.empty((0,6)),None).detect(image),[])

    def test_detection_only_weights_rejected_for_pipeline(self):
        with patch('scenelens.models.detector.YOLO',return_value=SimpleNamespace(task='detect')):
            with self.assertRaisesRegex(ValueError,'segmentation weights'):
                Detector('unused.pt',device='cpu',require_masks=True)


if __name__=='__main__':
    unittest.main()
