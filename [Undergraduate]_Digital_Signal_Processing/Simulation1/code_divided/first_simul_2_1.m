% ===============================
% 2-1. Discrete convolution 직접 구현
% ===============================

clear;
clc;
close all;

%%%%%%%%%% x2.wav 및 LPF, HPF 읽기 %%%%%%%%%%
[x2, Fs] = audioread('x2.wav');   % x2[n], Fs = 44100 Hz
hl = importdata('LPF.txt');       % LPF impulse response
hh = importdata('HPF.txt');       % HPF impulse response



%%%%%%%%%% 직접 구현한 컨볼루션 %%%%%%%%%%
x_l_custom = self_conv(x2', hl');   % LPF 적용
x_h_custom = self_conv(x2', hh');   % HPF 적용

%%%%%%%%%% n축의 범위를 정해줌  %%%%%%%%%%
n_l = 0:length(x_l_custom)-1;
n_h = 0:length(x_h_custom)-1;

%%%%%%%%%% 각각의 filter의 n-domain에서 x2[n]*h[n]을 plot %%%%%%%%%%
figure;

subplot(2, 1, 1);
plot(n_l, x_l_custom);
xlabel('n');
ylabel('Amplitude');
title('x_l[n] = h_l[n] * x_2[n] (Self Convolution)');
grid on;

subplot(2, 1, 2);
plot(n_h, x_h_custom);
xlabel('n');
ylabel('Amplitude');
title('x_h[n] = h_h[n] * x_2[n] (Self Convolution)');
grid on;

%============================================================%
% %%%%%%%%%% 추가: x2[n], hl[n], hh[n] n-domain plot %%%%%%%%%%
% figure;
% 
% subplot(3, 1, 1);
% n_x2 = 0:length(x2)-1;
% plot(n_x2, x2);
% xlabel('n');
% ylabel('Amplitude');
% title('x_2[n]');
% grid on;
% 
% subplot(3, 1, 2);
% n_hl = 0:length(hl)-1;
% stem(n_hl, hl);
% xlabel('n');
% ylabel('Amplitude');
% title('h_L[n]');
% grid on;
% 
% subplot(3, 1, 3);
% n_hh = 0:length(hh)-1;
% stem(n_hh, hh);
% xlabel('n');
% ylabel('Amplitude');
% title('h_H[n]');
% grid on;
% 
% %%%%%%%%%% 추가: Spectrum Magnitude (Log scale) %%%%%%%%%%
% N = length(x2);
% X2 = fft(x2, N);
% HL = fft(hl, N);
% HH = fft(hh, N);
% 
% % log scale 적용
% X2_mag = abs(X2);                  % Magnitude
% X2_log = log(1 + X2_mag);          % Log scale
% HL_mag = abs(HL);
% HL_log = log(1 + HL_mag);
% HH_mag = abs(HH);
% HH_log = log(1 + HH_mag);
% 
% % fftshift로 -π ~ π 범위로 바꿔줌
% X2_shifted = fftshift(X2_log);
% HL_shifted = fftshift(HL_log);
% HH_shifted = fftshift(HH_log);
% 
% 
% % ω-domain 범위를 정해준다. (ω는 [-π, π]까지)
% w = linspace(-pi, pi, N);
% 
% % Plot
% figure;
% 
% subplot(3, 1, 1);
% plot(w, X2_shifted);
% xlabel('\omega (rad/sample)');
% ylabel('Log Scale Magnitude');
% title('|X_2(\omega)|');
% grid on;
% 
% subplot(3, 1, 2);
% plot(w, HL_shifted);
% xlabel('\omega (rad/sample)');
% ylabel('Log Scale Magnitude');
% title('|H_L(\omega)|');
% grid on;
% 
% subplot(3, 1, 3);
% plot(w, HH_shifted);
% xlabel('\omega (rad/sample)');
% ylabel('Log Scale Magnitude');
% title('|H_H(\omega)|');
% grid on;
% 
% %%%%%%%%%% 추가: x_l_custom, x_h_custom Frequency Domain %%%%%%%%%%
% 
% % FFT, M-point에서 point수는 똑같이 맞춤
% N_conv = length(x_l_custom);
% X_L_custom = fft(x_l_custom, N_conv);
% X_H_custom = fft(x_h_custom, N_conv);
% 
% % log scale 적용
% X_L_mag_log = log(1 + abs(X_L_custom));
% X_H_mag_log = log(1 + abs(X_H_custom));
% 
% % fftshift로 -π ~ π 범위로 바꿔줌
% X_L_shifted = fftshift(X_L_mag_log);
% X_H_shifted = fftshift(X_H_mag_log);
% 
% % ω-domain 범위를 정해준다. (ω는 [-π, π]까지)
% w_conv = linspace(-pi, pi, N_conv);
% 
% % plot
% figure;
% 
% subplot(2, 1, 1);
% plot(w_conv, X_L_shifted);
% xlabel('\omega (rad/sample)');
% ylabel('Log Scale Magnitude');
% title('|X_L(\omega)| of x_l[n] = h_L[n] * x_2[n]');
% grid on;
% 
% subplot(2, 1, 2);
% plot(w_conv, X_H_shifted);
% xlabel('\omega (rad/sample)');
% ylabel('Log Scale Magnitude');
% title('|X_H(\omega)| of x_h[n] = h_H[n] * x_2[n]');
% grid on;

%============================================================%

%%%%%%%%%% Convolution 직접 구현한 함수 %%%%%%%%%%
% y[n] = sum{x[m] * h[n - m]} (m=0 ~ Nx+Nh-1까지)
% discrete한 신호 convolution 시 신호의 길이 = input 2개 신호의 길이 합 - 1
function y = self_conv(x, h)
    Nx = length(x);
    Nh = length(h);
    Ny = Nx + Nh - 1;
    y = zeros(1, Ny);

    % input 2개의 신호의 x[m] * h[n - m]가 y[n]에 accumulate됨
    % 0 < n-m+1 < Nx일 때가, 두 신호(x[m], h[n - m])가 겹치는 구간
    % 겹치는 구간의 곱을 계산하여 y[n]에 합해줌
    for n = 1:Ny
        for m = 1:Nh
            if (n - m + 1 > 0) && (n - m + 1 <= Nx)
                y(n) = y(n) + x(n - m + 1) * h(m);
            end
        end
    end
end



