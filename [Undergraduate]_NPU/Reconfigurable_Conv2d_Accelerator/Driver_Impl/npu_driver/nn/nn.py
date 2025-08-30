import numpy as np
import os
from ..driver import NPUDriver  # driver.py에서 NPUDriver 임포트

class Module:
    def __init__(self):
        self._driver = None
        # 기본 비트파일 경로 설정
        default_bitfile_path = "no_quant_2.bit"
        # 비트파일 존재 여부 확인
        if not os.path.exists(default_bitfile_path):
            raise FileNotFoundError(f"Default bitfile 'no_quant_2.bit' not found at {default_bitfile_path}")
        self._driver = NPUDriver(default_bitfile_path)
        self.register_driver(self._driver)

    def register_driver(self, driver):
        """드라이버 등록"""
        self._driver = driver
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module) and attr is not self:
                attr.register_driver(driver)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (3, 3)  # NPUDriver 제약: kernel_size=3 고정
        self.padding = 0  # NPUDriver 제약: padding=0 고정
        self.weight = None

    def set_weights(self, weight):
        """가중치 설정: (out_channels, in_channels, kernel_h, kernel_w)"""
        if weight.shape != (self.out_channels, self.in_channels, 3, 3):
            raise ValueError(f"Weight shape must be ({self.out_channels}, {self.in_channels}, 3, 3), got {weight.shape}")
        self.weight = weight.astype(np.float32)

    def forward(self, x):
        """입력 x: (in_channels, height, width)"""
        if self._driver is None:
            raise ValueError("Driver not registered for Conv2d layer")
        if self.weight is None:
            raise ValueError("Weights not set for Conv2d layer")
        
        if not isinstance(self._driver, NPUDriver):
            raise ValueError("Invalid driver type for Conv2d layer")
        
        return self._driver.run_conv_2d(x, self.weight)

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self._driver is None:
            raise ValueError("Driver not registered for ReLU layer")
        return self._driver.relu_loop(x, alpha=0.0)

class LeakyReLU(Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        if self._driver is None:
            raise ValueError("Driver not registered for LeakyReLU layer")
        return self._driver.leaky_relu_loop(x, alpha=self.alpha)

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if self._driver is None:
            raise ValueError("Driver not registered for MaxPool2d layer")
        return self._driver.maxpool_loop(x, pool_size=self.kernel_size, stride=self.stride)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None

    def set_weights(self, weight):
        """가중치 설정: (out_features, in_features)"""
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape must be ({self.out_features}, {self.in_features}), got {weight.shape}")
        self.weight = weight.astype(np.float32)

    def forward(self, x):
        """입력 x: (in_features,)"""
        if self._driver is None:
            raise ValueError("Driver not registered for Linear layer")
        if self.weight is None:
            raise ValueError("Weights not set for Linear layer")
        return self._driver.run_fc_2d(x, self.weight).flatten()