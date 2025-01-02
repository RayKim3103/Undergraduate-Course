clear;      % Clear all variables from the workspace
clc;        % Clear the Command Window
close all;  % Close all figure windows

% Sampling frequency
fs = 44100;

% Load the .mat file
Y = load('received_signal4.mat');

% Extract the received signal
received_signal = Y.received_signal;

% Estimated channel
h = 0.1938 + 0.7159i;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        1. Equalization                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% equalization (1-tap channel equalization)
X_hat = received_signal / h;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        2. Demodulation                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the 16-QAM constellation points
non_qamMap = [-3-3i, -3-1i, -3+1i, -3+3i, ...
          -1-3i, -1-1i, -1+1i, -1+3i, ...
           1-3i,  1-1i,  1+1i,  1+3i, ...
           3-3i,  3-1i,  3+1i,  3+3i];

% Calculate the average power of the constellation
averagePower = mean(abs(non_qamMap).^2);

% Calculate the normalization factor
normalizationFactor = sqrt(1 / averagePower);

% Normalize the constellation points 
% (수신된 신호가 normalized된 16QAM으로 modulation됨)
qamMap = normalizationFactor * non_qamMap;

% Corresponding binary values for each symbol (2D array)
% Consider Gray coding
bitMap = [
    0 0 0 0;   % 1
    0 0 0 1;   % 2
    0 0 1 1;   % 3
    0 0 1 0;   % 4
    0 1 0 0;   % 5
    0 1 0 1;   % 6
    0 1 1 1;   % 7
    0 1 1 0;   % 8
    1 1 0 0;   % 9
    1 1 0 1;   % 10
    1 1 1 1;   % 11
    1 1 1 0;   % 12
    1 0 0 0;   % 13
    1 0 0 1;   % 14
    1 0 1 1;   % 15
    1 0 1 0    % 16
];

% output bitstream as an empty array
bitstream = zeros(1, length(X_hat)*4);

for k = 1:length(X_hat)
    % Find the closest constellation point
    [~, idx] = min(abs(qamMap - X_hat(k)));
    
    % Append the corresponding bits to the bitstream
    bitstream((k-1)*4 + 1:k*4) = bitMap(idx, :);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               3. Error Correction & Decoding                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parity-check matrix H for Hamming (7,4) code
H = [
    1 0 0 0 1 1 1;
    0 1 0 1 1 0 1;
    0 0 1 1 0 1 1
];

% Transpose of H
H_T = H';

% % Syndrome lookup table for error correction
syndromeTable = [
    0 0 0 0; % No error
    1 0 0 1; % Error in bit 1
    0 1 0 2; % Error in bit 2
    0 0 1 3; % Error in bit 3
    0 1 1 4; % Error in bit 4
    0 1 1 5; % Error in bit 5
    1 0 1 6; % Error in bit 6
    1 1 1 7  % Error in bit 7
];


% Number of codewords in the bitstream
numCodewords = floor(length(bitstream) / 7);

% Initialize corrected bitstream
corrected = zeros(1, length(bitstream));

% calculate syndrome & correct bitstream
for i = 1:numCodewords
    codeword = bitstream((i-1)*7 + 1:i*7);
    s = mod(codeword * H_T, 2);
    errorPos = 0;
    for j = 1:size(syndromeTable, 1)
        if isequal(s, syndromeTable(j, 1:3))
            errorPos = syndromeTable(j, 4);
            break;
        end
    end
    if errorPos ~= 0
        codeword(errorPos) = mod(codeword(errorPos) + 1, 2);
    end
    corrected((i-1)*7 + 1:i*7) = codeword;
end

% elliminate parity bits
decodedData = zeros(1, numCodewords * 4);
for i = 1:numCodewords
    correctedCodeword = corrected((i-1)*7 + 1:i*7);
    decodedData((i-1)*4 + 1:i*4) = correctedCodeword(4:7);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        4. Dequantization                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of 5-bit samples
numSamples = floor(length(decodedData) / 5);

% Quantization look up table
quantize_power = [
    0 0 0 0 0 0; 
    0 0 0 0 1 1; 
    0 0 0 1 0 2; 
    0 0 0 1 1 3; 
    0 0 1 0 0 4; 
    0 0 1 0 1 5; 
    0 0 1 1 0 6; 
    0 0 1 1 1 7; 
    0 1 0 0 0 8; 
    0 1 0 0 1 9; 
    0 1 0 1 0 10; 
    0 1 0 1 1 11; 
    0 1 1 0 0 12; 
    0 1 1 0 1 13; 
    0 1 1 1 0 14; 
    0 1 1 1 1 15; 
    1 0 0 0 0 16; 
    1 0 0 0 1 17; 
    1 0 0 1 0 18; 
    1 0 0 1 1 19; 
    1 0 1 0 0 20; 
    1 0 1 0 1 21; 
    1 0 1 1 0 22; 
    1 0 1 1 1 23; 
    1 1 0 0 0 24; 
    1 1 0 0 1 25; 
    1 1 0 1 0 26; 
    1 1 0 1 1 27; 
    1 1 1 0 0 28; 
    1 1 1 0 1 29; 
    1 1 1 1 0 30; 
    1 1 1 1 1 31;
];

quantized = zeros(1, numSamples);

% calculate quantized to analog
for i = 1:numSamples
    q_levels = decodedData((i-1)*5 + 1:i*5);

    for j = 1:size(quantize_power, 1)
        if isequal(q_levels, quantize_power(j, 1:5))
            q_power = quantize_power(j, 6);
            break;
        end
    end
    quantized(i) = q_power/31;
end

% power = -1 ~ 1
analogSignal = -1 + 2 * quantized;

% Play the sound
% sound(analogSignal, fs);

% % audiowrite
% filename = 'project_2_a_sound.wav';
% audiowrite(filename,analogSignal,44100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  5. Reconstruction(Filtering)                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 주파수 벡터를 조정하여 길이를 맞춤
D = fft(analogSignal);

% 주파수 도메인 축을 만든다.
omega_s = 2*pi*fs;
FS_sampling = omega_s/length(D); % 샘플링 주파수 계산
omega = -omega_s/2 : FS_sampling : omega_s/2-FS_sampling;

% 음악의 Low frequency 대역을 통과시키는 Low Pass Filter 설계
low_pass_cutoff = 15000; % 첫 번째 음악의 cut-off frequency
low_pass_filter = (abs(omega) >= (44100-low_pass_cutoff)*pi);

% 원본 음악 파일 필터링
D_filtered = zeros(size(D));

for k = 1:length(D)
    D_filtered(k) = fftshift(D(k)) * low_pass_filter(k);
end

% 주파수 도메인에서의 역변환
filteredSignal = ifft(D_filtered);

% Play the filtered sound
sound(filteredSignal, fs);

% audiowrite
filename = 'project_2_a_sound.wav';
audiowrite(filename,filteredSignal,44100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LPF 플로팅
figure;
plot(omega/pi, fftshift(low_pass_filter));
title('LPF');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot the analog signal for visualization
t = (0:numSamples-1) / fs;
figure;
plot(t, filteredSignal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Reconstructed Analog Filtered Signal');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Perform Fourier Transform using fft
N = length(filteredSignal);  % Number of samples
Y = fft(filteredSignal);     % Compute the FFT
Y_shifted = fftshift(Y);   % Shift zero frequency component to center

% Frequency vector for plotting
f = (-N/2:N/2-1)*(2*fs/N);

% Plot the magnitude of the FFT
figure;
plot(f, abs(Y_shifted)/N);
title('Magnitude Spectrum of Analog filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
