% ================================
% 2-3. 필터링 결과 청취 및 비교
% ================================

clear;
clc;
close all;

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
