clear;
clc;
close all;

% QPSK BER Simulation in AWGN and Rayleigh Fading Channel with 1-tap Equalizer

% Parameters
EbN0_dB = 0:10; % Eb/N0 range in dB
N = 10^6; % Number of bits or symbols
EbN0 = 10.^(EbN0_dB/10);

% QPSK symbol generation
data = randi([0 1], 1, N); % Generate random data


G = [
    1 1 0 1 0 0 0;
    0 1 1 0 1 0 0;
    1 1 1 0 0 1 0;
    1 0 1 0 0 0 1
];

H = [
    1 0 0 1 0 1 1;
    0 1 0 1 1 1 0;
    0 0 1 0 1 1 1
];

% Transpose of H
H_T = H';

syndromeTable = [
    0 0 0 0; % No error
    1 0 0 1; % Error in bit 1
    0 1 0 2; % Error in bit 2
    0 0 1 3; % Error in bit 3
    1 1 0 4; % Error in bit 4
    0 1 1 5; % Error in bit 5
    1 1 1 6; % Error in bit 6
    1 0 1 7  % Error in bit 7
];

% Preallocate the encodedBits array
encodedBits = zeros(1, ceil(length(data) / 4) * 7);

% Hamming (7,4) code encoding
encodedBitsIdx = 1;
for i = 1:4:length(data)
    if i+3 <= length(data)
        block = data(i:i+3);
        encodedBlock = mod(block * G, 2);
        encodedBits(encodedBitsIdx:encodedBitsIdx+6) = encodedBlock;
        encodedBitsIdx = encodedBitsIdx + 7;
    end
end

% QPSK constellation points, Eb = 1
qpskMap = [-1-1i, -1+1i, 1+1i,  1-1i];

% % Normalize the constellation points
averagePower = mean(abs(qpskMap).^2);
normalizationFactor = sqrt(1 / averagePower);
qpskMap = normalizationFactor * qpskMap;

% gray coding
bitMap = [
    0 0;   % 1
    0 1;   % 2
    1 1;   % 3
    1 0   % 4
];

% Preallocate the symbols array
symbols = zeros(1, ceil(length(data) / 2));

% QPSK modulation
symbolsIdx = 1;
symbolIndex = 0;
for j = 1:2:length(encodedBits)
    if j+1 <= length(encodedBits)
        bits = encodedBits(j:j+1);
        for k = 1:size(bitMap, 1)
            if isequal(bits, bitMap(k, :))
                symbolIndex = k;
                break;
            end
        end
        symbols(symbolsIdx) = qpskMap(symbolIndex);
        symbolsIdx = symbolsIdx + 1;
    end
end

% Preallocation
BER_AWGN = zeros(1, length(EbN0_dB));
BER_Rayleigh = zeros(1, length(EbN0_dB));

% Simulation
for i = 1:length(EbN0_dB)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                         Add noise                         %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % AWGN Channel, Es=1 이고, noise는 심볼에 섞이는 노이즈이다
    % 따라서, 여기서 noise는 bit에 섞이는 것이 아니기에
    % noise power를 2*N0로 생각해줘야한다
    noise_len = length(symbols);
    noise =(randn(1, noise_len) + 1i*randn(1, noise_len)); %noise
    averagePower = mean(abs(noise).^2);
    normalizationFactor = sqrt(1 / averagePower);
    noise = normalizationFactor*noise;

    noisepower = 1/(2*EbN0(i));
    noise = sqrt(noisepower)*noise;
    r_awgn = symbols + noise;

    % Rayleigh Fading Channel
    % h ~ CN(0,1), 즉, real 부분 ~ N(0,0.5), imaginary 부분 ~ N(0,0.5)
    % 분산을 0.5로 하기 위해, 1/sqrt(2)로 scaling
    h = (1/sqrt(2))*(randn(1, noise_len) + 1i*randn(1, noise_len)); % Rayleigh fading

    r_rayleigh = h.*symbols + noise;
    
    
    % r_awgn
    % r_rayleigh
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                       Demodulation                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Equalization for Rayleigh channel
    r_rayleigh_eq = r_rayleigh ./ h;

    % Preallocate bitstream array
    bitstream_awgn = zeros(1, length(encodedBits));
    bitstream_rayleigh_eq = zeros(1, length(encodedBits));

    for k = 1:length(r_awgn)
        % Find the closest constellation point
        [~, idx] = min(abs(qpskMap - r_awgn(k)));
        
        % Append the corresponding bits to the bitstream
        bitstream_awgn((k-1)*2 + 1:k*2) = bitMap(idx, :);
    end

    for k = 1:length(r_rayleigh_eq)
        % Find the closest constellation point
        [~, idx] = min(abs(qpskMap - r_rayleigh_eq(k)));
        
        % Append the corresponding bits to the bitstream
        bitstream_rayleigh_eq((k-1)*2 + 1:k*2) = bitMap(idx, :);
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                        Decoding                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Number of codewords in the bitstream
    numCodewords = floor(length(bitstream_awgn) / 7);
    
    % Initialize corrected bitstream
    corrected_awgn = zeros(1, length(bitstream_awgn));
    corrected_rayleigh = zeros(1, length(bitstream_rayleigh_eq));
    
    % calculate syndrome & correct bitstream
    for k = 1:numCodewords
        codeword_awgn = bitstream_awgn((k-1)*7 + 1:k*7);
        codeword_rayleigh = bitstream_rayleigh_eq((k-1)*7 + 1:k*7);
        s_awgn = mod(codeword_awgn * H_T, 2);
        s_rayleigh = mod(codeword_rayleigh * H_T, 2);

        errorPos_awgn = 0;
        errorPos_rayleigh = 0;

        for j = 1:size(syndromeTable, 1)
            if isequal(s_awgn, syndromeTable(j, 1:3))
                errorPos_awgn = syndromeTable(j, 4);
                break;
            end
        end

        for j = 1:size(syndromeTable, 1)
            if isequal(s_rayleigh, syndromeTable(j, 1:3))
                errorPos_rayleigh = syndromeTable(j, 4);
                break;
            end
        end
        if errorPos_awgn ~= 0
            codeword_awgn(errorPos_awgn) = mod(codeword_awgn(errorPos_awgn) + 1, 2);
            
        end
        if errorPos_rayleigh ~= 0
            codeword_rayleigh(errorPos_rayleigh) = mod(codeword_rayleigh(errorPos_rayleigh) + 1, 2);
        end
        corrected_awgn((k-1)*7 + 1:k*7) = codeword_awgn;
        corrected_rayleigh((k-1)*7 + 1:k*7) = codeword_rayleigh;
    end
    
    decodedData_awgn = zeros(1, numCodewords * 4);
    decodedData_rayleigh = zeros(1, numCodewords * 4);
    for k = 1:numCodewords
        correctedCodeword_awgn = corrected_awgn((k-1)*7 + 1:k*7);
        decodedData_awgn((k-1)*4 + 1:k*4) = correctedCodeword_awgn(4:7);
        correctedCodeword_rayleigh = corrected_rayleigh((k-1)*7 + 1:k*7);
        decodedData_rayleigh((k-1)*4 + 1:k*4) = correctedCodeword_rayleigh(4:7);
    end


    bitErrors_awgn = sum(data ~= decodedData_awgn);
    BER_AWGN(i) = bitErrors_awgn / N;
    bitErrors_rayleigh = sum(data ~= decodedData_rayleigh);
    BER_Rayleigh(i) = bitErrors_rayleigh / N;
end


% Plotting
figure;
semilogy(EbN0_dB, BER_AWGN, 'b.-', EbN0_dB, BER_Rayleigh, 'r.-');
legend('AWGN Channel', 'Rayleigh Fading Channel with 1-tap Equalizer');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs. Eb/N0 for QPSK in AWGN and Rayleigh Fading Channel');


% hold on;
% semilogy(EbN0_dB, BER_QPSK, 'm-v', 'DisplayName', 'theo QPSK');

grid on;