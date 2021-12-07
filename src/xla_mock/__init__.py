import os
import sys


if not os.environ.get('XRT_TPU_CONFIG'):
    xla_mock = __import__(__name__)
    sys.modules['torch_xla'] = xla_mock
