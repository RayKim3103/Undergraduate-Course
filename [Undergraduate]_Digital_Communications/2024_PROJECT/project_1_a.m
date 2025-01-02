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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      QPSK modulation                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    1 0;   % 4
];

% Preallocate the symbols array
symbols = zeros(1, ceil(length(data) / 2));

% QPSK modulation
symbolsIdx = 1;
for j = 1:2:length(data)
    if j+1 <= length(data)
        bits = data(j:j+1);
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
    noise =(randn(1, N/2) + 1i*randn(1, N/2)); %noise
    averagePower = mean(abs(noise).^2);
    normalizationFactor = sqrt(1 / averagePower);
    noise = normalizationFactor*noise;
    
    % Eb = 0.5, N0 = 1/2EbN0, noisepower = 1/EbN0
    noisepower = 1/(2*EbN0(i));
    noise = sqrt(noisepower)*noise;
    r_awgn = symbols + noise;

    % Rayleigh Fading Channel
    % h ~ CN(0,1), 즉, real 부분 ~ N(0,0.5), imaginary 부분 ~ N(0,0.5)
    % 분산을 0.5로 하기 위해, 1/sqrt(2)로 scaling
    h = (1/sqrt(2))*(randn(1, N/2) + 1i*randn(1, N/2)); % Rayleigh fading
    r_rayleigh = h.*symbols + noise;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                       Demodulation                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Equalization for Rayleigh channel
    r_rayleigh_eq = r_rayleigh ./ h;

    % Preallocate bitstream array
    bitstream_awgn = zeros(1, length(data));
    bitstream_rayleigh_eq = zeros(1, length(data));

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


        bitErrors_awgn = sum(data ~= bitstream_awgn);
        BER_AWGN(i) = bitErrors_awgn / N;
        bitErrors_rayleigh = sum(data ~= bitstream_rayleigh_eq);
        BER_Rayleigh(i) = bitErrors_rayleigh / N;
end

% % BER 계산
% BER_QPSK = berawgn(EbN0_dB, 'psk', 4, 'nondiff');

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
