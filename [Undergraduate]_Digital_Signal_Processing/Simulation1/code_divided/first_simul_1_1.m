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

