% 1-1. x1[n]을 n-domain에서 plot & sound 듣기 (x축 = n-domain)

% clear;
% clc;
% close all;
% 
% % x1.wav 파일 불러와 읽고, Fs(Sampling Rate)를 가져온다.
% [x1, Fs] = audioread('x1.wav');  % Fs = 48000 (Hz)
% N = length(x1);                  % 96000 point (총 샘플 수)
% 
% % n-domain에 plot해야 하므로, n의 범위를 정해준다.
% n = 0:N-1;
% 
% % n-domain에서 x1[n]을 plot
% figure;
% plot(n, x1);
% xlabel('n');
% ylabel('Amplitude');
% title('n Domain Signal x_1[n]');
% grid on;
% 
% % sound 재생
% sound(x1, Fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-1. x1[n]을 t-domain에서 plot & sound 듣기 (x축 = t-domain)

clear;
clc;
close all;

% x1.wav 파일 불러와 읽고, Fs(Sampling Rate)를 가져온다.
[x1, Fs] = audioread('x1.wav');  % Fs = 48000 (Hz)
Ts = 1 / Fs;                     % Ts는 샘플링 주기 (F=1/T)
N = length(x1);                  % 96000 point (총 샘플 수)

% t-domain의 범위를 정해준다. 
% 샘플 수에 샘플링 주기를 곱하면, 몇 sec동안 음악이 나오는지 알 수 있다.
% x축을 통해 몇 sec동안 파형이 지속되는지 표기하는 것은
% n-domain을 t-domain으로 바꾸는 것과 동일하다
t = (0:N-1) * Ts;

% t-domain에서 x1[n]을 plot
figure;
plot(t, x1);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Time Domain Signal x_1[n]');
grid on;

% sound 재생
sound(x1, Fs);

%====================================================================%
% 1-2. frequency domain에서의 Spectrum Magnitude (Log scale)

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

%====================================================================%
% ===============================
% 2-1. Discrete convolution 직접 구현
% ===============================

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
%====================================================================%
% ================================
% 2-2. conv() 함수 사용
% ================================

%%%%%%%%%% x2.wav 및 LPF, HPF 읽기 %%%%%%%%%%
[x2, Fs] = audioread('x2.wav');   % x2[n], Fs = 44100 Hz
hl = importdata('LPF.txt');       % LPF impulse response h_l[n]
hh = importdata('HPF.txt');       % HPF impulse response h_h[n]

%%%%%%%%%% conv() 함수로 convolution %%%%%%%%%%
x_l_conv = conv(x2, hl);          % x_l[n] = h_l[n] * x2[n]
x_h_conv = conv(x2, hh);          % x_h[n] = h_h[n] * x2[n]

%%%%%%%%%% n축의 범위를 정해줌 %%%%%%%%%%
n_l = 0:length(x_l_conv)-1;
n_h = 0:length(x_h_conv)-1;

%%%%%%%%%% 각각의 filter의 n-domain에서 x2[n]*h[n]을 plot %%%%%%%%%%
figure;

subplot(2, 1, 1);
plot(n_l, x_l_conv);
xlabel('n');
ylabel('Amplitude');
title('x_l[n] = h_l[n] * x_2[n] (conv)');
grid on;

subplot(2, 1, 2);
plot(n_h, x_h_conv);
xlabel('n');
ylabel('Amplitude');
title('x_h[n] = h_h[n] * x_2[n] (conv)');
grid on;
%====================================================================%
% ====================================
% 2-1 직접 구현 vs 2-2 conv 결과 비교
% ====================================

%%%%% x2.wav 및 LPF, HPF 읽기 %%%%%
[x2, Fs] = audioread('x2.wav');
hl = importdata('LPF.txt');
hh = importdata('HPF.txt');

%%%%% 결과 계산 %%%%%
x_l_custom = self_conv(x2', hl');
x_l_conv   = conv(x2, hl);

x_h_custom = self_conv(x2', hh');
x_h_conv   = conv(x2, hh);

%%%%% 길이 맞추기, 최소값으로 맞춤 %%%%%
% 최댓값으로 맞추면, 두 신호의 길이가 안 맞을 때, 오차가 매우 커질 수 있음
min_len_l = min(length(x_l_custom), length(x_l_conv));
min_len_h = min(length(x_h_custom), length(x_h_conv));

diff_l = x_l_custom(1:min_len_l) - x_l_conv(1:min_len_l)';
diff_h = x_h_custom(1:min_len_h) - x_h_conv(1:min_len_h)';

%%%%% 오차 측정 %%%%%
fprintf('=== Low-pass filtering 차이 ===\n');
fprintf('평균 오차: %.6e\n', mean(abs(diff_l))); % 평균 절대 오차
fprintf('최대 오차: %.6e\n\n', max(abs(diff_l)));

fprintf('=== High-pass filtering 차이 ===\n');
fprintf('평균 오차: %.6e\n', mean(abs(diff_h))); % 평균 절대 오차
fprintf('최대 오차: %.6e\n', max(abs(diff_h)));

%%%%% Low-pass filtering 차이 plot %%%%%
figure;
subplot(2, 1, 1);
plot(0:min_len_l-1, diff_l);
xlabel('n');
ylabel('Difference');
title('Difference: x_l[n] (self\_conv - conv)');
grid on;

%%%%% High-pass filtering 차이 시각화 %%%%%
subplot(2, 1, 2);
plot(0:min_len_h-1, diff_h);
xlabel('n');
ylabel('Difference');
title('Difference: x_h[n] (self\_conv - conv)');
grid on;
%====================================================================%
% ================================
% 2-3. 필터링 결과 청취 및 비교
% ================================

%%%%%%%%%% x2.wav 및 LPF, HPF 읽기 %%%%%%%%%%
[x2, Fs] = audioread('x2.wav');   % 원본 음성, Fs = 44100 Hz
hl = importdata('LPF.txt');       % LPF
hh = importdata('HPF.txt');       % HPF

%%%%%%%%%% conv() 함수로 convolution %%%%%%%%%%
x_l = conv(x2, hl);               % Low-pass filtered signal
x_h = conv(x2, hh);               % High-pass filtered signal

%%%%%%%%%% sound 재생 %%%%%%%%%%
disp('x_2[n]');
sound(x2, Fs);
pause(length(x2)/Fs + 1);

disp('x_l[n]');
sound(x_l, Fs);
pause(length(x_l)/Fs + 1);

disp('x_h[n]');
sound(x_h, Fs);
pause(length(x_h)/Fs + 1);


% % ===============================================================
% % 직접 구현한 convolution 사용
% 
% clear;
% clc;
% close all;
% 
% %%%%%%%%%% x2.wav 및 LPF, HPF 읽기 %%%%%%%%%%
% [x2, Fs] = audioread('x2.wav');   % 원본 음성, Fs = 44100 Hz
% hl = importdata('LPF.txt');       % LPF
% hh = importdata('HPF.txt');       % HPF
% 
% %%%%%%%%%% self_conv() 사용 %%%%%%%%%%
% x_l_custom = self_conv(x2', hl');  % Low-pass 필터링 (직접 구현한 컨볼루션)
% x_h_custom = self_conv(x2', hh');  % High-pass 필터링 (직접 구현한 컨볼루션)
% 
% %%%%%%%%%% sound 재생 %%%%%%%%%%
% disp('x_2[n]');
% sound(x2, Fs);
% pause(length(x2)/Fs + 1); 
% 
% disp('x_l[n]');
% sound(x_l_custom, Fs);
% pause(length(x_l_custom)/Fs + 1);
% 
% disp('x_h[n]');
% sound(x_h_custom, Fs);
% pause(length(x_h_custom)/Fs + 1);
% 
% %%%%%%%%%% Convolution 직접 구현한 함수 %%%%%%%%%%
% % y[n] = sum{x[m] * h[n - m]} (m=0 ~ Nx+Nh-1까지)
% % discrete한 신호 convolution 시 신호의 길이 = input 2개 신호의 길이 합 - 1
% function y = self_conv(x, h)
%     Nx = length(x);
%     Nh = length(h);
%     Ny = Nx + Nh - 1;
%     y = zeros(1, Ny);
% 
%     % input 2개의 신호의 x[m] * h[n - m]가 y[n]에 accumulate됨
%     % 0 < n-m+1 < Nx일 때가, 두 신호(x[m], h[n - m])가 겹치는 구간
%     % 겹치는 구간의 곱을 계산하여 y[n]에 합해줌
%     for n = 1:Ny
%         for m = 1:Nh
%             if (n - m + 1 > 0) && (n - m + 1 <= Nx)
%                 y(n) = y(n) + x(n - m + 1) * h(m);
%             end
%         end
%     end
% end
% % ===============================================================
%%%%% Convolution 직접 구현한 함수 %%%%%
% y[n] = sum{x[m] * h[n - m]} (m=0 ~ Nx+Nh-1까지)
% discrete한 신호 convolution 시 신호의 길이 = input 2개 신호의 길이 합 - 1
function y = self_conv(x, h)
    Nx = length(x);
    Nh = length(h);
    Ny = Nx + Nh - 1;
    y = zeros(1, Ny);
    for n = 1:Ny
        for m = 1:Nh
            if (n - m + 1 > 0) && (n - m + 1 <= Nx)
                y(n) = y(n) + x(n - m + 1) * h(m);
            end
        end
    end
end

