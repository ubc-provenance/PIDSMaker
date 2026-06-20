import sys
import unittest.mock
sys.modules['numpy'] = unittest.mock.MagicMock()
sys.modules['torch'] = unittest.mock.MagicMock()
sys.modules['vispy'] = unittest.mock.MagicMock()
sys.modules['vispy.scene'] = unittest.mock.MagicMock()
sys.modules['PyQt5'] = unittest.mock.MagicMock()
from pidsmaker.vizgen.native.loader import load_data
try:
    path = '/home/artifacts/evaluation/evaluation/6d0e933173345f9108ccc4cddec8eb563406d65b10f8ed860359bfd421539c43/CADETS_E3/viz/embedding_viz_CADETS_E3_encoder_epoch_1_points.json'
    data = load_data(path)
    print("Success loading epoch 1")
except Exception as e:
    import traceback
    traceback.print_exc()
