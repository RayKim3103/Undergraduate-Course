clear;      % 모든 변수 지우기
clc;        % 명령창 지우기
close all;  % 모든 figure 창 닫기

%% problem 3 & 4

% read x.wav & set FT of x
[x, fs] = audioread('x.wav');
x = x(:);  % column vector
X_f = fftshift(fft(x, fs));                 % shift to w=0, FFT
X_PowerSpec = abs(X_f).^2;                  % |X(w)|^2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% beta = 1e-2;
beta_squared = 1e-4;            % variance of signal
omega = linspace(-pi, pi, fs);  % [-pi, pi]

% list of how many poles we are using, AR
p_list = [3, 7, 13];

figure;
hold on;
legend_names = {};

for idx = 1:length(p_list)
    p = p_list(idx);

    % compute r
    r0 = mean(x .* conj(x));
    r_tail = r_xx(x, p);
    r = [r0; r_tail];

    % Toeplitz Matrix
    R = toeplitz(r(1:p));
    r_vec = r(2:p+1);
    a_coeff = -inv(R) * r_vec;
    a_final = [1; a_coeff];     % note: a[0] = 1
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % Varify whether it is ok to us Toeplitz function
    % fprintf('\nr(1:%d) as row vector:\n\n', p);
    % row_r = r(1:p).';  % 열 벡터를 행 벡터로 변환 (transpose)
    % 
    % for i = 1:p
    %     fprintf('%10.4e  ', row_r(i));  % 지수 형식으로 출력
    % end
    % fprintf('\n');
    % 
    % % Display header
    % fprintf('\nToeplitz Matrix R (%d x %d):\n\n', p, p);
    % fprintf('%6s', '');  % 빈 칸 (열 인덱스용)
    % for j = 1:p
    %     fprintf('%10d  ', j);  % 열 인덱스 출력
    % end
    % fprintf('\n');
    % 
    % % Display matrix with row indices
    % for i = 1:p
    %     fprintf('%6d    ', i);  % 행 인덱스 출력
    %     for j = 1:p
    %         fprintf('%10.4e  ', R(i,j));  % 지수형으로 값 출력
    %     end
    %     fprintf('\n');
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % print
    fprintf('AR(%d) coefficients:\n', p);
    disp(a_final);

    % compute A(w)
    A_w = zeros(size(omega));
    for k = 0:p
        A_w = A_w + a_final(k + 1) * exp(-1j * omega * k);
    end

    % compute PSD
    PSD = beta_squared ./ abs(A_w).^2;
    
    a = 1;
    % a = max(X_PowerSpec) / max(PSD);

    % Log scale
    PSD_log = log10(a*PSD);
    % PSD_log = log(a*PSD);
    % PSD_log = log(1+PSD);

    % Plot
    plot(omega, PSD_log, 'DisplayName', sprintf('AR(%d)', p));
    legend_names{end+1} = sprintf('AR(%d)', p);
end

title('Power Spectral Density (Log Scale) \alpha = 1');
xlabel('\omega (rad/sample)');
ylabel('log_1_0(Magnitude)');
legend show;
grid on;
xlim([-pi, pi]);

%% Problem 5

% settings
p = 13;           % number of poles
M = 2 * p;        % m = 1 ~ 2p
N = length(x);

% --- compute auto crrelation r[1] ~ r[2p] ---
r_vec = r_xx(x, M);    % Make Vector (SIZE: [2p x 1])

% --- R' Matrix (SIZE: [2p x p]) ---
R_prime = zeros(M, p);
r_full = [mean(x .* conj(x)); r_vec];  % r[0] is auto correlation with self

for m = 1:M
    for k = 1:p
        R_prime(m, k) = r_full(abs(m - k) + 1);  % lag |m-k|
    end
end

% --- use Least Squares to compute a ---
a_coeff = -inv(R_prime' * R_prime) * R_prime' * r_vec;

% --- use a[0] = 1 to find final a_coefficents ---
a_final = [1; a_coeff];

% --- print ---
fprintf('AR(13) coefficients (Use Least Square solution):\n');
disp(a_final);

%% Problem 5
% --- compute PSD & log scale plot ---
beta_squared = 1e-4;
omega = linspace(-pi, pi, fs);
% a = 3000;
a = 1;

% A(w)
A_w = zeros(size(omega));
for k = 0:p
    A_w = A_w + a_final(k+1) * exp(-1j * omega * k);
end

% PSD
PSD = beta_squared ./ abs(A_w).^2;

% log scale
% PSD_log = log(a*PSD);
PSD_log = log10(a*PSD);

% --- plot ---
figure;
plot(omega, PSD_log);
title('AR(13) Power Spectral Density (Log Scale) \alpha = 1');
xlabel('\omega (rad/sample)');
ylabel('log_1_0(Magnitude)');
xlim([-pi, pi]);
grid on;

%% Problem 7

% --- 1. compute Power Spectrum |X(w)|^2 ---
X_f = fftshift(fft(x, fs));    % FT of x           
X_PowerSpec = abs(X_f).^2;     % |X(w)|^2

% log scale power spectrum
X_PowerSpec_log = log10(X_PowerSpec);
% X_PowerSpec_log = log(X_PowerSpec);

% n = 0:(length(x)-1);
% t = n / fs;
% disp(length(n));
% disp(length(x));
% figure;
% plot(t, x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
legend_names = {};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for idx = 1:length(p_list)
    p = p_list(idx);

    % compute r
    r0 = mean(x .* conj(x));
    r_tail = r_xx(x, p);
    r = [r0; r_tail];

    % Toeplitz Matrix
    R = toeplitz(r(1:p));
    r_vec = r(2:p+1);
    a_coeff = -inv(R) * r_vec;
    a_final = [1; a_coeff];     % note: a[0] = 1

    % print
    % fprintf('AR(%d) coefficients:\n', p);
    % disp(a_final);

    % compute A(w)
    A_w = zeros(size(omega));
    for k = 0:p
        A_w = A_w + a_final(k + 1) * exp(-1j * omega * k);
    end

    % compute PSD
    PSD = beta_squared ./ abs(A_w).^2;
    
    % a = 1;
    a = max(X_PowerSpec) / max(PSD);

    % Log scale
    PSD_log = log10(a*PSD);
    % PSD_log = log(a*PSD);
    % PSD_log = log(1+PSD);

    % Plot
    plot(omega, PSD_log, 'LineWidth', 1.5, 'DisplayName', sprintf('AR(%d)', p));
    legend_names{end+1} = sprintf('AR(%d)', p);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega = linspace(-pi, pi, fs);

% settings
p = 13;           % number of poles
M = 2 * p;        % m = 1 ~ 2p
N = length(x);

% --- compute auto crrelation r[1] ~ r[2p] ---
r_vec = r_xx(x, M);    % Make Vector (SIZE: [2p x 1])

% --- R' Matrix (SIZE: [2p x p]) ---
R_prime = zeros(M, p);
r_full = [mean(x .* conj(x)); r_vec];  % r[0] is auto correlation with self

for m = 1:M
    for k = 1:p
        R_prime(m, k) = r_full(abs(m - k) + 1);  % lag |m-k|
    end
end

% --- use Least Squares to compute a ---
a_coeff = -inv(R_prime' * R_prime) * R_prime' * r_vec;

% --- use a[0] = 1 to find final a_coefficents ---
a_final = [1; a_coeff];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- compute PSD & log scale plot ---
beta_squared = 1e-4;
omega = linspace(-pi, pi, fs);
a = max(X_PowerSpec) / max(PSD);

% A(w)
A_w = zeros(size(omega));
for k = 0:p
    A_w = A_w + a_final(k+1) * exp(-1j * omega * k);
end

% PSD
PSD = beta_squared ./ abs(A_w).^2;

% log scale
% PSD_log = log(a*PSD);
PSD_log = log10(a*PSD);

% --- plot ---
plot(omega, PSD_log, 'LineWidth', 1.5, 'DisplayName', sprintf('AR_L_S(%d)', 13));
title('AR(13) Power Spectral Density (Log Scale)');
xlabel('\omega (rad/sample)');
ylabel('log_1_0(Magnitude)');
xlim([-pi, pi]);
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- plot ---
plot(omega, X_PowerSpec_log, 'DisplayName', sprintf('X(w)^2(%d)', 13));
title('Square Magnitude |X(W)|^2');
xlabel('\omega (rad/sample)');
ylabel('log_1_0(Magnitude)');
legend;
grid on;
xlim([-pi, pi]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% poles = roots(a_final);
% disp("poles");
% disp(poles);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
subplot(2,1,1);
plot(omega, abs(log10(1./A_w)), 'LineWidth', 1.5);
title('1/A(W)');
xlabel('\omega (rad/sample)');
ylabel('log_1_0(Magnitude)');
xlim([-pi, pi]);
subplot(2,1,2);
plot(omega, angle(1./A_w), 'LineWidth', 1.5);
title('1/A(W) Phase');
xlabel('\omega (rad/sample)');
ylabel('angle(radian)');
xlim([-pi, pi]);



function output = r_xx(x,p)
    N = length(x);
    for i=1:p
         n=1:1:N-i;
         a=x(n+i).*conj(x(n));
         output(i,1)=mean(a);
    end
end