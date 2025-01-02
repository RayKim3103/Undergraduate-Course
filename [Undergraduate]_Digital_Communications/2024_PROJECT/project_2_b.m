clear;
clc;
close all;

% Parameters
fs = 44100;  % Sampling frequency
EbN0_dB = 10;  % Eb/N0 in dB

% Generator matrix for Hamming (7,4) code
G = [
    0 1 1 1 0 0 0;
    1 1 0 0 1 0 0;
    1 0 1 0 0 1 0;
    1 1 1 0 0 0 1
];  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Load the original signal (same with 2.(a))            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
originalSignal = ifft(D_filtered);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      start transmitting                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             1. Quantization (5-bit, 32 levels)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
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

% Preallocate the quantizedSignal array
quantizedSignal = zeros(1, length(originalSignal));

% Quantization
for i = 1:length(originalSignal)
    [~, idx] = min(abs(originalSignal(i) - (-1 + 2*(quantize_power(:,6)/31))));
    quantizedSignal(i) = idx;
end

% Preallocate the quantizedBits array
quantizedBits = zeros(1, length(quantizedSignal) * 5);

% Convert quantized levels to binary
% quantization에서 기본적으로 올림 연산을 했기 때문에, 전체적으로
% 5bit의 값을 decimal로 봤을 때 +1이 되는 경향을 보임
for i = 1:length(quantizedSignal)
    value = quantizedSignal(i);
    for bit = 5:-1:1
        quantizedBits((i-1)*5 + bit) = mod(value, 2);
        value = floor(value / 2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                2. Hamming (7,4) code encoding                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Preallocate the encodedBits array
encodedBits = zeros(1, ceil(length(quantizedBits) / 4) * 7);

% Hamming (7,4) code encoding
encodedBitsIdx = 1;
for i = 1:4:length(quantizedBits)
    if i+3 <= length(quantizedBits)
        block = quantizedBits(i:i+3);
        encodedBlock = mod(block * G, 2);
        encodedBits(encodedBitsIdx:encodedBitsIdx+6) = encodedBlock;
        encodedBitsIdx = encodedBitsIdx + 7;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      3. 16 QAM modulation                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 16-QAM constellation points
qamMap = [-3-3i, -3-1i, -3+1i, -3+3i, ...
          -1-3i, -1-1i, -1+1i, -1+3i, ...
           1-3i,  1-1i,  1+1i,  1+3i, ...
           3-3i,  3-1i,  3+1i,  3+3i];

% Normalize the constellation points
averagePower = mean(abs(qamMap).^2);
normalizationFactor = sqrt(1 / averagePower);
qamMap = normalizationFactor * qamMap;

% Preallocate the symbols array
symbols = zeros(1, ceil(length(encodedBits) / 4));
% 16-QAM modulation
symbolsIdx = 1;
for j = 1:4:length(encodedBits)
    if j+3 <= length(encodedBits)
        bits = encodedBits(j:j+3);
        for k = 1:size(bitMap, 1)
            if isequal(bits, bitMap(k, :))
                symbolIndex = k;
                break;
            end
        end
        symbols(symbolsIdx) = qamMap(symbolIndex);
        symbolsIdx = symbolsIdx + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               4. add noises               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Transmit through Rayleigh channel and add noise
h = 0.1938 + 0.7159i;
noisySymbols = symbols * h;

% Calculate noise power
% 사용한 식: Eb/N0 = (Es/N0)*(1/log2(M)) -> N0 = Es/(log2(M)*Eb/N0)
% AWGN Channel, Es=1 이고, N0: noisepower는 심볼에 섞이는 노이즈이다

EbN0 = 10^(EbN0_dB/10);
noisePower = 1/(4*EbN0);

% Add noise to the symbols
noisySymbols = noisySymbols + sqrt(noisePower) * 1/sqrt(2) * (randn(size(noisySymbols)) + 1i*randn(size(noisySymbols)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                start receiving  (same with 2.(a))                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Receiver processing (Assuming the same code as provided in 2.a)
% noise is amplified (quantize noise + channel noise + AWGN)
% also |h| = 0.7465, which amplifies quantize noise & AWGN
receivedSignal = noisySymbols / h;

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
qamMap = normalizationFactor * non_qamMap;

% Corresponding binary values for each symbol (2D array)
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

% Preallocate bitstream array
bitstream = zeros(1, length(receivedSignal)*4);

for k = 1:length(receivedSignal)
    % Find the closest constellation point
    [~, idx] = min(abs(qamMap - receivedSignal(k)));
    
    % Append the corresponding bits to the bitstream
    bitstream((k-1)*4 + 1:k*4) = bitMap(idx, :);
end

% Parity-check matrix H for Hamming (7,4) code
H = [
    1 0 0 0 1 1 1;
    0 1 0 1 1 0 1;
    0 0 1 1 0 1 1
];

% Transpose of H
H_T = H';

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

decodedData = zeros(1, numCodewords * 4);
for i = 1:numCodewords
    correctedCodeword = corrected((i-1)*7 + 1:i*7);
    decodedData((i-1)*4 + 1:i*4) = correctedCodeword(4:7);
end


% Number of 5-bit samples
numSamples = floor(length(decodedData) / 5);

quantized = zeros(1, numSamples);

% calculate quantized levels
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

analogSignal = -1 + 2 * quantized;

% Reconstruction(Filtering)
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
filename = 'project_2_b_sound.wav';
audiowrite(filename,filteredSignal,44100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot the original and received signals for comparison
t = (0:length(originalSignal)-1) / fs;
figure;
subplot(2,1,1);
plot(t, originalSignal(1:length(t)));
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(t, filteredSignal);
title('Received Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Perform Fourier Transform using fft
N1 = length(originalSignal);  % Number of samples
Y1 = fft(originalSignal);     % Compute the FFT
Y1_shifted = fftshift(Y1);   % Shift zero frequency component to center

% Frequency vector for plotting
f1 = (-N1/2:N1/2-1)*(2*fs/N1);

% Plot the magnitude of the FFT
figure;
subplot(2,1,1);
plot(f1, abs(Y1_shifted)/max(abs(Y1)));
title('Magnitude Spectrum of Original Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Perform Fourier Transform using fft
N2 = length(filteredSignal);  % Number of samples
Y2 = fft(filteredSignal);     % Compute the FFT
Y2_shifted = fftshift(Y2);   % Shift zero frequency component to center

% Frequency vector for plotting
f2 = (-N2/2:N2/2-1)*(2*fs/N2);

% Plot the magnitude of the FFT
subplot(2,1,2);
plot(f2, abs(Y2_shifted)/max(abs(Y2)));
title('Magnitude Spectrum of Received Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;