% ====================================
% 2-1 직접 구현 vs 2-2 conv 결과 비교
% ====================================

clear;
clc;

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

%%%%% Convolution 직접 구현한 함수 %%%%%
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