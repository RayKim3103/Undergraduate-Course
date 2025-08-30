import numpy as np
import struct
from pynq import Overlay

class LayerConfig:
    """
    NPU 레이어 설정 클래스
    - input_w, input_h: 입력 너비/높이
    - kernel_w, kernel_h: 커널 너비/높이
    - input_ch: 입력 채널 수
    - output_ch: 출력 채널 수
    """
    def __init__(self, input_w, input_h, kernel_w, kernel_h, input_ch, output_ch):
        self.input_w = input_w
        self.input_h = input_h
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.stride = 1

class NPUDriver:
    def __init__(self, bitfile_path):
        self.hw = Overlay(bitfile_path)
        self.csr = self.hw.csr_0.mmio.array
        self.imem = self.hw.INPUT_MEM.mmio.array
        self.omem = self.hw.OUT_MEM.mmio.array
        self.wmem = self.hw.WEIGHT_MEM.mmio.array
        print(f"IMEM Shape: {self.imem.shape}")
        print(f"WMEM Shape: {self.wmem.shape}")
        print(f"OMEM Shape: {self.omem.shape}")

    def write_csr(self, address, value):
        """CSR write"""
        self.csr[address//4] = value

    def read_csr(self, address):
        return self.csr[address//4]

    def config_layer(self, layer_config):
        """레이어 config 설정"""
        self.write_csr(0x08, layer_config.kernel_w)    # kernel_w
        self.write_csr(0x0C, layer_config.kernel_h)    # kernel_h
        self.write_csr(0x10, layer_config.input_ch)    # input_ch
        self.write_csr(0x14, layer_config.input_w)     # input_w
        self.write_csr(0x18, layer_config.input_h)     # input_h
        self.write_csr(0x1C, layer_config.output_ch)   # output_ch

    def load_data(self, input_data, weight_data):
        """데이터 로드: WMEM (weight), IMEM (input) -> flatten 시켜서"""
        TOTAL_MEM_SIZE = 16384
    
        input_data_size = input_data.size
        weight_data_size = weight_data.size
        
        in_data_float32 = input_data.astype(np.float32).ravel()
        weight_data_float32 = weight_data.astype(np.float32).ravel()
        
        # float32를 16진수 정수로 변환
        in_hex_int = [struct.unpack('<I', np.float32(x).tobytes())[0] for x in in_data_float32]
        weight_hex_int = [struct.unpack('<I', np.float32(x).tobytes())[0] for x in weight_data_float32]

        # input 데이터 패딩
        padded_in_hex_int = in_hex_int[:input_data_size] + [0] * (TOTAL_MEM_SIZE - input_data_size)
        self.imem[0:TOTAL_MEM_SIZE] = padded_in_hex_int

        # weight 데이터 패딩
        padded_weight_hex_int = weight_hex_int[:weight_data_size] + [0] * (TOTAL_MEM_SIZE - weight_data_size)
        self.wmem[0:TOTAL_MEM_SIZE] = padded_weight_hex_int

    def get_data(self, num_elements):
        """데이터 가져오기: OMEM (output)"""
        return self.omem[0:num_elements].astype(np.int8)

    def start_npu(self, input_data, weight_data, layer_config):
        """데이터 로드 및 NPU 실행"""
        self.load_data(input_data, weight_data)
        self.config_layer(layer_config)
        self.write_csr(0x04, 1) 
        
        while True:
            if self.read_csr(0x00) == 1:
                break
            
        output_size = ((layer_config.input_w - layer_config.kernel_w) // layer_config.stride + 1) * \
                      ((layer_config.input_h - layer_config.kernel_h) // layer_config.stride + 1) * \
                      layer_config.output_ch
        return self.get_data(output_size)

    def run_conv_2d(self, input_data, weight_data, tile_h=8, tile_w=8, tile_oc=8):
        """
        conv2d (타일링 기반) 실행.
        - input_data: (input_ch, input_h, input_w)
        - weight_data: (output_ch, input_ch, kernel_h, kernel_w)
        """
        stride = 1
        
        input_ch, input_h, input_w = input_data.shape
        output_ch, _, kernel_h, kernel_w = weight_data.shape

        output_h = (input_h - kernel_h) // stride + 1
        output_w = (input_w - kernel_w) // stride + 1
        o_act = np.zeros((output_ch, output_h, output_w), dtype=np.float32)

        for oh in range(0, output_h, tile_h):
            for ow in range(0, output_w, tile_w):
                for oc in range(0, output_ch, tile_oc):
                    h_range = min(tile_h, output_h - oh)
                    w_range = min(tile_w, output_w - ow)
                    oc_range = min(tile_oc, output_ch - oc)

                    h_start = oh * stride
                    w_start = ow * stride
                    h_end = h_start + h_range * stride + kernel_h - 1
                    w_end = w_start + w_range * stride + kernel_w - 1
                    
                    i_act_tile = input_data[:, h_start:h_end, w_start:w_end]
                    weight_tile = weight_data[oc:oc + oc_range, :, :, :]
                    
                    tile_o_act_flat = np.zeros(oc_range * h_range * w_range).astype(np.float32)
                    tile_o_act = np.zeros((oc_range, h_range, w_range)).astype(np.float32)
                    
                    input_ch, tile_h_input, tile_w_input = i_act_tile.shape
                    layer_config = LayerConfig(tile_w_input, tile_h_input, kernel_w, kernel_h, input_ch, oc_range)
                    
                    tile_o_act_flat = self.start_npu(i_act_tile, weight_tile, layer_config)
                    tile_o_act = tile_o_act_flat.reshape(oc_range, h_range, w_range)
                    
                    o_act[oc:oc + oc_range, oh:oh + h_range, ow:ow + w_range] = tile_o_act

        return o_act

    def reshape_input_and_weights(x, weights, fc_key='fc', kernel_size=3, oc_size=10):
        # Flatten input
        input_flat = x.flatten()
        
        # Calculate padding length
        total_length = np.ceil(len(input_flat) / (kernel_size * kernel_size * 8)) * (kernel_size * kernel_size * 8)
        pad_length = int(total_length - len(input_flat))
        
        # Pad input
        input_padded = np.pad(input_flat, (0, pad_length), mode='constant', constant_values=0)
        
        # Reshape input
        ic = int(total_length // (kernel_size * kernel_size))
        input_reshaped = input_padded.reshape(ic, kernel_size, kernel_size)
        
        # Load weight data
        weight_data = weights[fc_key]
        
        # Initialize reshaped weights array
        weight_reshaped = np.zeros((oc_size, ic, kernel_size, kernel_size), dtype=weight_data.dtype)
        
        # Reshape weights for each output channel
        for oc in range(oc_size):
            weight_flat = weight_data[oc]
            weight_padded = np.pad(weight_flat, (0, pad_length), mode='constant', constant_values=0)
            weight_oc_reshaped = weight_padded.reshape(ic, kernel_size, kernel_size)
            weight_reshaped[oc] = weight_oc_reshaped
        
        return input_reshaped, weight_reshaped

    def run_fc_2d(self, i_act, weight, tile_h=3, tile_w=3, tile_oc=1, padding=0):
        """
        FC를 conv2d로 구현 (Tiling).
        - i_act: (input_ch, input_h, input_w)
        - weight: (output_ch, input_ch, kernel_h, kernel_w)
        """
        stride = 1
        
        input_reshaped, weight_reshaped = self.reshape_input_and_weights(i_act, weight)

        input_ch, input_h, input_w = input_reshaped.shape
        output_ch, _, kernel_h, kernel_w = weight_reshaped.shape

        output_h = (input_h - kernel_h) // stride + 1
        output_w = (input_w - kernel_w) // stride + 1
        o_act = np.zeros((output_ch, output_h, output_w), dtype=np.float32)
        
        for oh in range(0, output_h, tile_h):
            for ow in range(0, output_w, tile_w):
                for oc in range(0, output_ch, tile_oc):
                    h_range = min(tile_h, output_h - oh)
                    w_range = min(tile_w, output_w - ow)
                    oc_range = min(tile_oc, output_ch - oc)

                    h_start = oh * stride
                    w_start = ow * stride
                    h_end = h_start + h_range * stride + kernel_h - 1
                    w_end = w_start + w_range * stride + kernel_w - 1

                    i_act_tile = input_reshaped[:, h_start:h_end, w_start:w_end]
                    weight_tile = weight_reshaped[oc:oc + oc_range, :, :, :]

                    input_ch, tile_h_input, tile_w_input = i_act_tile.shape
                    layer_config = LayerConfig(tile_w_input, tile_h_input, kernel_w, kernel_h, input_ch, oc_range)
                    
                    tile_o_act_flat = self.start_npu(i_act_tile, weight_tile, layer_config)
                    tile_o_act = tile_o_act_flat.reshape(oc_range, h_range, w_range)
                    
                    o_act[oc:oc + oc_range, oh:oh + h_range, ow:ow + w_range] = tile_o_act

        return o_act