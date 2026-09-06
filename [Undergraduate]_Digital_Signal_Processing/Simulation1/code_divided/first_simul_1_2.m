% 1-2. frequency domain에서의 Spectrum Magnitude (Log scale)

clear;      
clc;        
close all;  

% x1.wav 파일 불러와 읽고, Fs(Sampling Rate)를 가져온다.
[x1, Fs] = audioread('x1.wav');  % Fs = 48000 Hz
N = length(x1);                  % 96000 point (총 샘플 수)

% FFT 계산
X = fft(x1, N);                  % N-point FFT (DFT 대신 사용) (N = 96000)

% Magnitude 계산 및 log scale 적용
X_mag = abs(X);                  % Magnitude
X_log = log(1 + X_mag);          % Log scale

% fftshift로 -π ~ π 범위로 바꿔줌
X_log_shifted = fftshift(X_log);

% ω-domain 범위를 정해준다. (ω는 [-π, π]까지)
w = linspace(-pi, pi, N);

% ω-domain에서 F{x1[n]} = X(ω)을 plot
figure;
plot(w, X_log_shifted);
xlabel('\omega (rad/sample)');
ylabel('log(1 + |X(\omega)|)');
title('Log-Magnitude Spectrum of x_1[n]');
grid on;
