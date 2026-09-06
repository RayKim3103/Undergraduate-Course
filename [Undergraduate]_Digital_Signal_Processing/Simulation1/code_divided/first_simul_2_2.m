% ================================
% 2-2. conv() 함수 사용
% ================================

clear;
clc;
close all;

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
