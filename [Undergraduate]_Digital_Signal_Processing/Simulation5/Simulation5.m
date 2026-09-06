clear;      % 모든 변수 지우기
clc;        % 명령창 지우기
close all;  % 모든 figure 창 닫기


%% Problem 1

% 1. input signal
[x, Fs] = audioread('input.wav');  % Fs = 8000 Hz

%%%%%%%%%%%%%%%%%%%%%%%%%% Playing Sound %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("playing x[n]: input.wav")
sound(x, Fs);
pause(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make Time domain
t = (0:length(x)-1) / Fs;

% 2. compute PSD
X_shift = fftshift(fft(x));     % FFT shift
N = length(x);
omega = linspace(-pi, pi, N);   % omega domain (-π ~ π)
PSD_X_shift = abs(X_shift).^2;          % compute power spectrum
PSD_log = log(1 + PSD_X_shift);         % log scale
% PSD_log = log10(1 + PSD_X_shift);         % log scale

% 3. plot x in time domain
figure;
subplot(2,1,1);
plot(t, x);
xlabel('Time [s]');
ylabel('Amplitude');
title('입력 신호 x[n]');
grid on

% 4. plot PSD (log scale)
subplot(2,1,2);
plot(omega, PSD_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('입력 신호 x[n]의 PSD (log scale)');
xlim([-pi, pi]);
grid on;


%% Problem 2

% 1. filter of system
h = readmatrix('h.txt');

% 2. Compute Frequency Response H(ω)
N = length(h);              % FFT length, 보기 좋게 1024로 바꿔도 됨
H_shift = fftshift(fft(h, N));
w = linspace(-pi, pi, N);   % ω-domain

% 3. plot h in n domain
figure;
subplot(2,1,1);
stem(0:length(h)-1, h);
xlabel('n');
ylabel('h[n]');
title('filter h[n] in n domain');
grid on;

% 4. Magnitude plot (linear scale)
subplot(2,1,2);
plot(w, abs(H_shift));
xlabel('\omega (rad/sample)');
ylabel('|H(\omega)|');
title('Frequency Response Magnitude of h[n]');
xlim([-pi pi]);
grid on;


%% Problem 3

% 1. noise
[u, fs_noise] = audioread('noise.wav');        % u[n] : 

%%%%%%%%%%%%%%%%%%%%%%% NOISE FFT PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 푸리에 변환 (FFT)
% N = length(u);                   
% U = fftshift(fft(u));             % 푸리에 변환 + DC 성분 중앙 정렬
% omega = linspace(-pi, pi, N);     % 디지털 주파수 축 [-π, π]
% 
% % 스펙트럼 시각화
% figure;
% plot(omega, abs(U));
% xlabel('\omega (rad/sample)');
% ylabel('|U(\omega)|');
% title('Fourier Transform of Noise Signal (Digital Frequency)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Playing Sound %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("playing noise[n]: noise.wav")
sound(u, Fs);
pause(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2-1. If we use convolution for computing y
y = conv(x,h,'same') + u;

%%%%%%%%%%%%%%%%%%%%%%%%%% Playing Sound %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("playing y[n]: Degraded input.wav")
sound(y, Fs);
pause(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-2. If we use convolution property for computing y
% need to make Length same for convolution property (Zero-padding)
% N_conv = length(x) + length(h) - 1;
% X = fft(x, N_conv);
% H = fft(h, N_conv);
% U = fft(u, N_conv);
% 
% convolution in frequency domain -> make Y(w) & y[n]
% Y_freq = X .* H + U;                   % Y(ω) = X(ω)·H(ω) + U(ω)
% y = real(ifft(Y_freq));                % 시간 영역 신호로 변환
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3. time domain
t = (0:length(y)-1) / Fs;

% 4. PSD (log scale)
Y_shift = fftshift(fft(y));

N_conv = length(y);
w = linspace(-pi, pi, N_conv);

PSD_Y_shift = abs(Y_shift).^2;
PSD_y_log = log(1 + PSD_Y_shift);
% PSD_y_log = log10(1 + PSD_Y_shift);

% 5. plot y[n]
figure;
subplot(2,1,1);
plot(t, y);
xlabel('Time [s]');
ylabel('Amplitude');
title('열화된 신호 y[n]');
grid on;

subplot(2,1,2);
plot(w, PSD_y_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('열화된 신호 y[n]의 PSD (log scale)');
xlim([-pi pi]);
grid on;


%% Problem 4 - Wiener Filter (Use: S_yy)

% 1. ReCompute Signals need for Wiener Filter -> need to make length same
Y_shift = fftshift(fft(y, N_conv));        % 열화된 신호 y[n]의 FFT
H_shift = fftshift(fft(h, N_conv));        % 시스템 필터 FFT
H_conj_shift = conj(H_shift);              % H* (complex conjugate)

% 2. Power Spectrum of noise & y[n]
S_uu = 5e-3;                     % S_uu(ω) = 5 × 10^-3 (문제 조건)
S_yy = abs(Y_shift).^2;          % S_yy(ω)

% 3. Compute Wiener Filter
H_wiener = (H_conj_shift) ./ (abs(H_shift).^2 + (S_uu ./ S_yy));  % eq.(2)

% 4. Reconstructed Spectrum of X(ω) -> X̂₁(ω)
X_hat1_shift = H_wiener .* Y_shift;

% 5. Inverse FFT → Reconstruced singnal in time-domain
x_hat1 = real(ifft(ifftshift(X_hat1_shift)));

% 6. make time domain
t = (0:length(x_hat1)-1) / Fs;

% 7. make omega domain
w = linspace(-pi, pi, length(x_hat1));


% 8. PSD (log scale)
PSD_X_hat1_shift = abs(X_hat1_shift).^2;

PSD_X_hat1_shift_log = log(1 + PSD_X_hat1_shift);
% PSD_X_hat1_shift_log = log10(1 + PSD_X_hat1_shift);

%%%%%%%%%%%%%%%%%%%%%%%%%% Playing Sound %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("playing x_hat1[n]: Restored(Syy) input.wav")
sound(x_hat1, Fs);
pause(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 9. plot
figure;
subplot(2,1,1);
plot(t, x_hat1);
xlabel('Time [s]');
ylabel('Amplitude');
title('복원된 신호 X1(w) (Wiener 필터)');

subplot(2,1,2);
plot(w, PSD_X_hat1_shift_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('복원된 신호 x1[n]의 PSD (log scale)');
xlim([-pi pi]);
grid on;

%% Problem 5 - 원본, 열화, 복원된 신호의 파형 및 PSD 비교

% time domain & w domain should be same to compare
len_min = min([length(x), length(y), length(x_hat1)]);
t_comp = (0:len_min-1) / Fs;
w_comp = linspace(-pi, pi, len_min);  % 공통 주파수 축

% fft of x, y, x1
X_comp = fftshift(fft(x(1:len_min)));
Y_comp = fftshift(fft(y(1:len_min)));
Xhat1_comp = fftshift(fft(x_hat1(1:len_min)));

% Compute PSD
PSD_x = abs(X_comp).^2;
PSD_y = abs(Y_comp).^2;
PSD_xhat1 = abs(Xhat1_comp).^2;

% Compute log scale
PSD_x_log = log(1 + PSD_x);
PSD_y_log = log(1 + PSD_y);
PSD_xhat1_log = log(1 + PSD_xhat1);
% PSD_x_log = log10(1 + PSD_x);
% PSD_y_log = log10(1 + PSD_y);
% PSD_xhat1_log = log10(1 + PSD_xhat1);

% === 1. plot x, y, x1 ===
figure;
subplot(3,1,1);
plot(t_comp, x(1:len_min));
xlabel('Time [s]');
ylabel('Amplitude');
title('원본 신호 x[n]');
grid on;

subplot(3,1,2);
plot(t_comp, y(1:len_min));
xlabel('Time [s]');
ylabel('Amplitude');
title('열화된 신호 y[n]');
grid on;

subplot(3,1,3);
plot(t_comp, x_hat1(1:len_min));
xlabel('Time [s]');
ylabel('Amplitude');
title('복원된 신호 x1[n] (Wiener)');
grid on;


% === 2. PSD of x, y, x1 ===
figure;
subplot(3,1,1);
plot(w_comp, PSD_x_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 원본 신호 x[n]');
xlim([-pi pi]);
grid on;

subplot(3,1,2);
plot(w_comp, PSD_y_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 열화된 신호 y[n]');
xlim([-pi pi]);
grid on;

subplot(3,1,3);
plot(w_comp, PSD_xhat1_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 복원된 신호 x1[n]');
xlim([-pi pi]);
grid on;

%% Problem5 - detail comparison

% time domain & w domain should be same to compare
len_min = min([length(x), length(y), length(x_hat1)]);
t_comp = (0:len_min-1) / Fs;
w_comp = linspace(-pi, pi, len_min);

% cut signals for comparison
x_cut = x(1:len_min);
y_cut = y(1:len_min);
xhat1_cut = x_hat1(1:len_min);

% Compute PSD
X_cut = fftshift(fft(x_cut));
Y_cut = fftshift(fft(y_cut));
Xhat1_cut = fftshift(fft(xhat1_cut));

PSD_x = abs(X_cut).^2;
PSD_y = abs(Y_cut).^2;
PSD_xhat1 = abs(Xhat1_cut).^2;

PSD_x_log = log(1 + PSD_x);
PSD_y_log = log(1 + PSD_y);
PSD_xhat1_log = log(1 + PSD_xhat1);
% PSD_x_log = log10(1 + PSD_x);
% PSD_y_log = log10(1 + PSD_y);
% PSD_xhat1_log = log10(1 + PSD_xhat1);

% === 1. plot in time domain ===
figure;
plot(t_comp, x_cut, 'b-'); 
hold on;
plot(t_comp, y_cut, 'r--');
plot(t_comp, xhat1_cut, 'k:', 'LineWidth', 1.2);
xlabel('Time [s]');
ylabel('Amplitude');
title('x[n], y[n], x_1[n] Time Domain 비교');
legend('원본 x[n]', '열화 y[n]', '복원 x_1[n]');
grid on;

% === 2. PSD (log scale) - plot in freq domain ===
figure;
plot(w_comp, PSD_x_log, 'b-'); 
hold on;
plot(w_comp, PSD_y_log, 'r--');
plot(w_comp, PSD_xhat1_log, 'k:', 'LineWidth', 1.1);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('x[n], y[n], x_1[n] PSD 비교 (log scale)');
legend('원본 x[n]', '열화 y[n]', '복원 x_1[n]');
xlim([-pi pi]);
grid on;


%% Problem 6 - Wiener 필터 (Use: S_xx instead of S_yy)

% 1. ReCompute Signals need for Wiener Filter -> need to make length same
X_shift = fftshift(fft(x, N_conv));       % 원본 신호 x[n]의 FFT
Y_shift = fftshift(fft(y, N_conv));       % 열화된 신호 y[n]의 FFT
H_shift = fftshift(fft(h, N_conv));       % 필터
H_conj_shift = conj(H_shift);             % H*(ω)

% 2. Compute PSD
S_xx = abs(X_shift).^2;                   % S_xx(ω)
S_uu = 5e-3;                              % 문제에서 주어진 S_uu

% 3. Compute Wiener Filter (eq.2)
H_wiener2 = (H_conj_shift) ./ (abs(H_shift).^2 + (S_uu ./ S_xx));

% 4. X̂₂(ω) = H_wiener2 · Y(ω)
X_hat2_shift = H_wiener2 .* Y_shift;

% 5. IFFT로 x̂₂[n] 얻기
x_hat2 = real(ifft(ifftshift(X_hat2_shift)));

% 6. time domain & omega domain
t = (0:length(x_hat2)-1) / Fs;
w = linspace(-pi, pi, length(x_hat2));

% 7. Compute PSD
PSD_X_hat2 = abs(X_hat2_shift).^2;
PSD_X_hat2_log = log(1 + PSD_X_hat2);  % log scale
% PSD_X_hat2_log = log10(1 + PSD_X_hat2);  % log scale


%%%%%%%%%%%%%%%%%%%%%%%%%% Playing Sound %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("playing x_hat2[n]: Restored(Sxx) input.wav")
sound(x_hat2, Fs);
pause(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 8. plot
figure;
subplot(2,1,1);
plot(t, x_hat2);
xlabel('Time [s]');
ylabel('Amplitude');
title('복원된 신호 x_2[n] (Wiener 필터 - S_{xx} 사용)');
grid on;

subplot(2,1,2);
plot(w, PSD_X_hat2_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('복원된 신호 x_2[n]의 PSD (log scale)');
xlim([-pi pi]);
grid on;

%% Problem 7 - 원본, 열화, 복원(x1, x2) 신호 비교

% time domain & w domain should be same to compare
len_min = min([length(x), length(y), length(x_hat1), length(x_hat2)]);
t_comp = (0:len_min-1) / Fs;
w_comp = linspace(-pi, pi, len_min);

% cut signals for comparison
x_cut = x(1:len_min);
y_cut = y(1:len_min);
xhat1_cut = x_hat1(1:len_min);
xhat2_cut = x_hat2(1:len_min);

% Compute PSD
X_cut = fftshift(fft(x_cut));
Y_cut = fftshift(fft(y_cut));
Xhat1_cut = fftshift(fft(xhat1_cut));
Xhat2_cut = fftshift(fft(xhat2_cut));

PSD_x = abs(X_cut).^2;
PSD_y = abs(Y_cut).^2;
PSD_xhat1 = abs(Xhat1_cut).^2;
PSD_xhat2 = abs(Xhat2_cut).^2;

% log scale
PSD_x_log = log(1 + PSD_x);
PSD_y_log = log(1 + PSD_y);
PSD_xhat1_log = log(1 + PSD_xhat1);
PSD_xhat2_log = log(1 + PSD_xhat2);
% PSD_x_log = log10(1 + PSD_x);
% PSD_y_log = log10(1 + PSD_y);
% PSD_xhat1_log = log10(1 + PSD_xhat1);
% PSD_xhat2_log = log10(1 + PSD_xhat2);

%%%%%%%%%% 1. plot in time domain %%%%%%%%%%
figure;
plot(t_comp, x_cut, 'k', 'LineWidth', 1.1); 
hold on;
plot(t_comp, y_cut, 'r--', 'LineWidth', 1.1);
plot(t_comp, xhat1_cut, 'b:', 'LineWidth', 1.1);
plot(t_comp, xhat2_cut, 'g-.');
xlabel('Time [s]');
ylabel('Amplitude');
title('x[n], y[n], x̂₁[n], x̂₂[n] 시간 파형 비교');
legend('원본 x[n]', '열화 y[n]', '복원 x̂₁[n]', '복원 x̂₂[n]');
grid on;

%%%%%%%%%% 2. PSD Comparison (log scale) %%%%%%%%%%
figure;
plot(w_comp, PSD_x_log, 'k', 'LineWidth', 1.1); 
hold on;
plot(w_comp, PSD_y_log, 'r--', 'LineWidth', 1.1);
plot(w_comp, PSD_xhat1_log, 'b:', 'LineWidth', 1.1);
plot(w_comp, PSD_xhat2_log, 'g-.');
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('x[n], y[n], x̂₁[n], x̂₂[n] PSD 비교 (log scale)');
legend('원본 x[n]', '열화 y[n]', '복원 x̂₁[n]', '복원 x̂₂[n]');
xlim([-pi pi]);
grid on;

%% Problem 7 - 원본, 열화, 복원(x̂₁, x̂₂) 신호 비교 하나 씩

% 1. 길이 정렬 (가장 짧은 길이 기준 자르기)
len_min = min([length(x), length(y), length(x_hat1), length(x_hat2)]);
t_comp = (0:len_min-1) / Fs;
w_comp = linspace(-pi, pi, len_min);

% 2. 신호 자르기
x_cut = x(1:len_min);
y_cut = y(1:len_min);
xhat1_cut = x_hat1(1:len_min);
xhat2_cut = x_hat2(1:len_min);

% 3. FFT 및 PSD 계산
X_cut = fftshift(fft(x_cut));
Y_cut = fftshift(fft(y_cut));
Xhat1_cut = fftshift(fft(xhat1_cut));
Xhat2_cut = fftshift(fft(xhat2_cut));

PSD_x = abs(X_cut).^2;
PSD_y = abs(Y_cut).^2;
PSD_xhat1 = abs(Xhat1_cut).^2;
PSD_xhat2 = abs(Xhat2_cut).^2;

% log scale 변환
PSD_x_log = log(1 + PSD_x);
PSD_y_log = log(1 + PSD_y);
PSD_xhat1_log = log(1 + PSD_xhat1);
PSD_xhat2_log = log(1 + PSD_xhat2);
% PSD_x_log = log10(1 + PSD_x);
% PSD_y_log = log10(1 + PSD_y);
% PSD_xhat1_log = log10(1 + PSD_xhat1);
% PSD_xhat2_log = log10(1 + PSD_xhat2);

%%%%%%%%%% 1. 시간 영역 파형 비교 %%%%%%%%%%
figure;

% 1. 원본 신호 x[n]
subplot(4, 1, 1);
plot(t_comp, x_cut);
xlabel('Time [s]');
ylabel('Amplitude');
title('원본 신호 x[n]');
grid on;

% 2. 열화된 신호 y[n]
subplot(4, 1, 2);
plot(t_comp, y_cut);
xlabel('Time [s]');
ylabel('Amplitude');
title('열화된 신호 y[n]');
grid on;

% 3. 복원된 신호 x̂₁[n]
subplot(4, 1, 3);
plot(t_comp, xhat1_cut);
xlabel('Time [s]');
ylabel('Amplitude');
title('복원된 신호 x̂₁[n]');
grid on;

% 4. 복원된 신호 x̂₂[n]
subplot(4, 1, 4);
plot(t_comp, xhat2_cut);
xlabel('Time [s]');
ylabel('Amplitude');
title('복원된 신호 x̂₂[n]');
grid on;

%%%%%%%%%% 2. PSD 로그 스케일 비교 (각각 subplot) %%%%%%%%%%
figure;

subplot(4, 1, 1);
plot(w_comp, PSD_x_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 원본 신호 x[n]');
xlim([-pi pi]);
grid on;

subplot(4, 1, 2);
plot(w_comp, PSD_y_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 열화된 신호 y[n]');
xlim([-pi pi]);
grid on;

subplot(4, 1, 3);
plot(w_comp, PSD_xhat1_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 복원된 신호 x̂₁[n]');
xlim([-pi pi]);
grid on;

subplot(4, 1, 4);
plot(w_comp, PSD_xhat2_log);
xlabel('\omega (rad/sample)');
ylabel('log(1 + PSD)');
title('PSD of 복원된 신호 x̂₂[n]');
xlim([-pi pi]);
grid on;