clear; % 모든 변수 지우기
clc; % 명령창 지우기
close all; % 모든 figure 창 닫기

%% problem 1

% 1. Define the system parameters
theta = 2 * pi / 5; % Angle of the poles
p1 = 1.4 * exp(-1j * theta);    % Pole 1
p2 = 1.4 * exp(1j * theta);     % Pole 2 (complex conjugate)

% 2. Set the samples for frequency response
N = 1024; % Number of frequency points
omega = linspace(-pi, pi, N); % Frequency range from -pi to pi

% 3. Make H(z)
% H(z) = 1 / [(1 - p1*z^-1)(1 - p2*z^-1)]
H = zeros(1, N);
for k = 1:N
    z = exp(1j * omega(k));         % z = e^(jω)
    denom1 = 1 - p1 * (1/z);        % 1 - p1*z^-1
    denom2 = 1 - p2 * (1/z);        % 1 - p2*z^-1
    H(k) = 1 / (denom1 * denom2);   % H(e^jω)
end

% Step 4: Plot the magnitude response (linear scale)
figure;
subplot(2, 1, 1);
plot(omega, abs(H));
grid on;
title('Spectral Magnitude Response of H(z)');
xlabel('\omega (radians)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});


% 5. 위상 응답 플로팅 (선형 스케일)
subplot(2, 1, 2);
plot(omega, angle(H));
grid on;
title('Spectral Phase Response of H(z)');
xlabel('\omega (radians)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

% figure;
% subplot(2,1,1);
% plot(omega, real(H), 'LineWidth', 1.5);
% title('real part of H(z)');
% grid on;
% 
% subplot(2,1,2);
% plot(omega, imag(H), 'LineWidth', 1.5);
% title('imaginary part of H(z)');
% grid on;


%% problem 2

%===== 1. Define parameters =====%
N = 1024; % Number of points
% n = (0:N-1); % n from 0 to 1023
n = 1:N;

%===== 2. Generate x[n] = cos(pi/4 * n) + cos(pi/6 * n) + sin(2*pi/3 * n) =====%
x = cos(pi/4 * n) + cos(pi/6 * n) + sin(2*pi/3 * n);

%===== 3. Compute FFT for frequency domain =====%
% X = fft(x, N);                  % Compute FFT
X = fft(x);                  % Compute FFT
X_shifted = fftshift(X);        % Shift to center at 0
omega = linspace(-pi, pi, N);   % Omega from -pi to pi

%===== 4. Compute magnitude and phase =====%
x_magnitude = abs(X_shifted);             % Magnitude
x_phase = angle(X_shifted);               % Phase
real_part = real(X_shifted);            % Real part of X
imag_part = imag(X_shifted);            % Imaginary part of X
% phase = atan2(imag_part, real_part);    % Phase using atan2

%===== 5. Plot x[n] in time domain =====%
figure;
subplot(3, 1, 1);
plot(n, x);
grid on;
title('Input Signal x[n]');
xlabel('n');
ylabel('x[n]');
xlim([0, N-1]);

%===== 6. Plot magnitude & phase in frequency domain (linear scale) =====%
% figure;
subplot(3, 1, 2);
% subplot(2,1,1);
plot(omega, x_magnitude);
grid on;
title('Magnitude |X(e^jw)|');
xlabel('w (radians)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(3, 1, 3);
% subplot(2,1,2);
plot(omega, x_phase);
grid on;
title('Phase <X(e^jw)');
xlabel('w (radians)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

figure;
% subplot(2,1,1);
% plot(omega, real_part, 'LineWidth', 1.5);
% title('real part of X(e^jw)');
% grid on;
% 
% subplot(2,1,2);
% plot(omega, imag_part, 'LineWidth', 1.5);
% title('imaginary part of X(e^jw)');
% grid on;

%% problem 3

%===== 1. Compute y1[n] = x[n] through H(z) =====%
Y1_shifted = X_shifted .* H;        % Y1(e^jw) = X(e^jw) * H(e^jw)
y1 = ifft(ifftshift(Y1_shifted, N));   % y1[n] = IFFT of Y1(e^jw)

%===== 2. Compute magnitude and phase of Y1(e^jw) =====%
y1_magnitude = abs(Y1_shifted);     % Magnitude
% y1_phase = atan2(imag(Y1_shifted), real(Y1_shifted)); % Phase using atan2
y1_phase = angle(Y1_shifted);       % Phase

%===== 3. Plot y1[n] & Y(w) magnitude, phase (linear scale) =====%
figure;
subplot(3, 1, 1);
% plot(n, y1);
plot(n, real(y1)); % Use real part to avoid small imaginary residuals
grid on;
title('Output Signal y1[n]');
xlabel('n');
ylabel('y1[n]');
xlim([0, N-1]);

subplot(3, 1, 2);
plot(omega, y1_magnitude);
grid on;
title('Magnitude |Y1(e^jw)|');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(3, 1, 3);
plot(omega, y1_phase);
grid on;
title('Phase <Y1(e^jw)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

% figure;
% subplot(2,1,1);
% plot(omega, real(Y1_shifted), 'LineWidth', 1.5);
% title('real part of Y1(e^jw)');
% grid on;
% 
% subplot(2,1,2);
% plot(omega, imag(Y1_shifted), 'LineWidth', 1.5);
% title('imaginary part of Y1(e^jw)');
% grid on;

%% problem 4

%===== 1. Define the system parameters (same as H(z)) =====%
% theta = 2 * pi / 5; % Angle of the poles
% p1 = 1.4 * exp(-1j * theta); % Pole 1
% p2 = 1.4 * exp(1j * theta); % Pole 2 (complex conjugate)
% 
% % 2. Set frequency range
% N = 1024; % Number of points
% omega = linspace(-pi, pi, N); % Omega from -pi to pi (radians/sample)
% 
% % 3. Compute Hi(z) = 1 / H(z)
% Hi = zeros(1, N);
% for k = 1:N
%     z = exp(1j * omega(k)); % z = e^(jw)
%     num1 = 1 - p1 * (1/z); % 1 - p1*z^-1
%     num2 = 1 - p2 * (1/z); % 1 - p2*z^-1
%     Hi(k) = num1 * num2; % Hi(e^jw) = (1 - p1*z^-1)(1 - p2*z^-1)
% end

%===== 1. Just taking 1/H for every elements =====%
Hi = 1 ./ H;
% Hi = zeros(1, N);
% for k = 1:N
%     Hi(k) = 1/H(k);
% end

%===== 2. Plot magnitude & phase response (linear scale) =====%
figure;
subplot(2, 1, 1);
plot(omega, abs(Hi));
grid on;
title('Magnitude Response of Hi(z)');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(2, 1, 2);
plot(omega, angle(Hi));
grid on;
title('Phase Response of Hi(z)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

%% Problem 5

%===== 1. Compute x'[n] by passing y1[n] through Hi(z) =====%
X_prime_shifted = Y1_shifted .* Hi;             % X'(e^jw) = Y1(e^jw) * Hi(e^jw)
x_prime = ifft(ifftshift(X_prime_shifted));     % x'[n] = IFFT of X'(e^jw)

%===== 2. Plot x'[n], magnitude, phase =====%
figure;
subplot(3, 1, 1);
% plot(n, x_prime);
plot(n, real(x_prime)); % Use real part to avoid small imaginary residuals
grid on;
title('Output Signal x''[n]');
xlabel('n');
ylabel('x''[n]');
xlim([0, N-1]);

subplot(3, 1, 2);
plot(omega, abs(X_prime_shifted));
grid on;
title('Magnitude |X''(e^jw)|');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(3, 1, 3);
plot(omega, angle(X_prime_shifted));
grid on;
title('Phase <X''(e^jw)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

%===== 3. Compare x'[n] with x[n] =====%
% figure;
% plot(n, abs(real(x) - real(x_prime)));
% xlabel('n');
% ylabel('Amplitude');
% plot(n, real(x), 'b-');           % Original x[n]
% hold on;
% plot(n, real(x_prime), 'r--');    % Recovered x'[n]
% grid on;
% title('Comparison of x[n] and x''[n]');
% xlabel('n');
% ylabel('Amplitude');
% xlim([0, N-1]);
% legend('show');

% 8. Compute and display the error between x[n] and x'[n]
error = max(abs(real(x) - real(x_prime))); % Maximum absolute error
disp(['Maximum error between x[n] and x''[n]: ', num2str(error)]);

% 9. Compute and display the error between X(w) and X'(w)
error = max(abs(abs(X_shifted) - abs(X_prime_shifted))); % Maximum absolute error
disp(['Maximum error between X(w) and X''(w): ', num2str(error)]);

%% Problem 6

%===== 1. Define the system parameters (same as H(z)) =====%
theta = 2 * pi / 5;             % Angle of the poles
p1 = 1.4 * exp(-1j * theta);    % Pole 1
p2 = 1.4 * exp(1j * theta);     % Pole 2 (complex conjugate)

%===== 2. Set frequency range =====%
N = 1024;                       % Number of points
omega = linspace(-pi, pi, N);   % Omega from -pi to pi (radians/sample)

%===== 3. Compute Hmin(z) (minimum phase system) =====%
p1_conj = (1.4) * exp(-1j * theta);
p2_conj = (1.4) * exp(1j * theta);
Hmin = zeros(1, N);
for k = 1:N
    z = exp(1j * omega(k));             % z = e^(jw)
    denom1 = (1/z) - p1_conj;           % z^-1 - p1_conj
    denom2 = (1/z) - p2_conj;           % z^-1 - p1_conj
    Hmin(k) = 1 / (denom1 * denom2);    % Hmin(e^jw)
end

% 4. Compute Hap(z) (all-pass filter) = Hmin(z) / H(z)
Hap = Hmin ./ H;                        % Hap(e^jw) = Hmin(e^jw) / H(e^jw)

% 5. Plot magnitude and phase response of Hmin(z) (linear scale)
figure;
subplot(2, 1, 1);
plot(omega, abs(Hmin));
grid on;
title('Magnitude Response of Hmin(z)');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(2, 1, 2);
plot(omega, angle(Hmin));
grid on;
title('Phase Response of Hmin(z)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

% 6. Plot magnitude and phase response of Hap(z) (linear scale)
figure;
subplot(2, 1, 1);
plot(omega, abs(Hap));
grid on;
title('Magnitude Response of Hap(z)');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(2, 1, 2);
plot(omega, angle(Hap));
grid on;
title('Phase Response of Hap(z)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

% figure;
% subplot(2,1,1);
% plot(omega, real(Hap), 'LineWidth', 1.5);
% title('real part of Hap(e^jw)');
% grid on;
% 
% subplot(2,1,2);
% plot(omega, imag(Hap), 'LineWidth', 1.5);
% title('imaginary part of Hap(e^jw)');
% grid on;

%% Problem 7

%===== 1. Compute y2[n] = x[n] through Hmin(z) =====%
Y2_shifted = X_shifted .* Hmin;             % Y2(e^jw) = X(e^jw) * Hmin(e^jw)
y2 = ifft(ifftshift(Y2_shifted));           % y2[n] = IFFT of Y2(e^jw)

%===== 2. Plot y2[n], magnitude, phase (linear scale) =====%
figure;
subplot(3, 1, 1);
plot(n, real(y2)); % Use real part to avoid small imaginary residuals
grid on;
title('Output Signal y2[n]');
xlabel('n');
ylabel('y2[n]');
xlim([0, N-1]);

subplot(3, 1, 2);
plot(omega, abs(Y2_shifted));
grid on;
title('Magnitude |Y2(e^jw)|');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(3, 1, 3);
plot(omega, angle(Y2_shifted));
grid on;
title('Phase <Y2(e^jw)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

%===== 3. Compare y1[n] with y2[n] in time domain =====%
% figure;
% plot(n, abs(real(y1)-real(y2))); % y1[n] from H(z) / % y2[n] from Hmin(z)
% grid on;
% title('Comparison of y1[n] and y2[n]');
% xlabel('n');
% ylabel('Amplitude');
% xlim([0, N-1]);
% legend('show');

%===== 4. Compute and display the error between y1[n], y[n] responses =====%
error_magnitude = max(abs(real(y1) - real(y2))); % Maximum absolute error in magnitude
disp(['Maximum error between y1[n] and y2[n]: ', num2str(error_magnitude)]);

%===== 5. Compute and display the error between magnitude responses =====%
error_magnitude = max(abs(abs(Y1_shifted) - abs(Y2_shifted))); % Maximum absolute error in magnitude
disp(['Maximum error between Y1(w) and Y2(w): ', num2str(error_magnitude)]);

%% Problem 8

Hap_book = H ./ Hmin;

% Plot magnitude and phase response of Hap(z) (linear scale)
figure;
subplot(2, 1, 1);
plot(omega, abs(Hap_book));
grid on;
title('Magnitude Response of Hap(z)');
xlabel('w (radians/sample)');
ylabel('Magnitude');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

subplot(2, 1, 2);
plot(omega, angle(Hap_book));
grid on;
title('Phase Response of Hap(z)');
xlabel('w (radians/sample)');
ylabel('Phase (radians)');
xlim([-pi, pi]);
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
