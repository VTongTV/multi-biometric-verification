"""
Wrapper for Worldcoin's Pretrained Iris Segmentation Model (ONNX).

Source: https://huggingface.co/worldcoin/iris-semantic-segmentation
Model: UNet++ with MobileNetV2 encoder (scSE attention)
Input: 640x480 (or similar aspect), normalized.
Output: 4 channel mask (Eyeball, Iris, Pupil, Eyelashes).
"""

import cv2
import numpy as np
from loguru import logger
from pathlib import Path

# Try importing onnxruntime, handle failure gracefully
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("onnxruntime not found. Worldcoin segmentation will be disabled.")


class WorldcoinIrisSegmenter:
    """
    Wrapper for Worldcoin's ONNX iris segmentation model.
    """
    
    def __init__(self, model_path=None):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required for WorldcoinIrisSegmenter")
            
        if model_path is None:
            # Default to looking in weights directory
            project_root = Path(__file__).parent.parent.parent.parent.parent
            model_path = project_root / 'weights' / 'iris_semseg_upp_scse_mobilenetv2.onnx'
            
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            logger.warning(f"ONNX model not found at {self.model_path}")
            self.session = None
        else:
            try:
                # Prefer CUDA if available
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(str(self.model_path), providers=providers)
                logger.info(f"Loaded Worldcoin segmentation model from {self.model_path}")
                logger.info(f"Active provider: {self.session.get_providers()[0]}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                self.session = None

    def segment(self, image):
        """
        Segment the image.
        
        Args:
            image: numpy array (H, W, 3) BGR or RGB
            
        Returns:
            masks: (4, H, W) numpy array (probabilities)
                   Channels: 0=Eyeball, 1=Iris, 2=Pupil, 3=Eyelashes
            quality: float (0.0-1.0)
        """
        if self.session is None:
            return None, 0.0
            
        # Preprocessing expected by Worldcoin model
        target_h, target_w = 480, 640
        h, w = image.shape[:2]
        
        input_img = cv2.resize(image, (target_w, target_h))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 float
        input_tensor = input_img.astype(np.float32) / 255.0
        
        # CHW format
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, 480, 640)
        
        # Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Output is logits, apply sigmoid
        prob_map = outputs[0][0]
        prob_map = 1 / (1 + np.exp(-prob_map))
        
        # Resize back to original size
        prob_map_t = prob_map.transpose(1, 2, 0)
        prob_map_orig = cv2.resize(prob_map_t, (w, h), interpolation=cv2.INTER_LINEAR)
        masks = prob_map_orig.transpose(2, 0, 1)
        
        # Quality score based on iris visibility
        iris_mask = masks[1] > 0.5
        if iris_mask.sum() == 0:
            quality = 0.0
        else:
            quality = 1.0
            
        return masks, quality

    def extract_iris_roi_mask(self, image, target_size=(224, 224)):
        """
        Extract ROI using segmentation mask center.
        """
        masks, quality = self.segment(image)
        if masks is None:
            return None, 0.0
             
        iris_map = masks[1]
        
        # Find center of mass of iris
        M = cv2.moments((iris_map > 0.5).astype(np.uint8))
        if M["m00"] == 0:
            return None, 0.0
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Determine radius (approximate) - Sqrt(Area / Pi)
        area = M["m00"]
        r = int(np.sqrt(area / np.pi))
        
        # Crop 2.5x radius
        crop_size = int(r * 2.5)
        h, w = image.shape[:2]
        
        x1 = max(0, cx - crop_size//2)
        y1 = max(0, cy - crop_size//2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None, 0.0
            
        roi = cv2.resize(crop, target_size)
        return roi, quality
