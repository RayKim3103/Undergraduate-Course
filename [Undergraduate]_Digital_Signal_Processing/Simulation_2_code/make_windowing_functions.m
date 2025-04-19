

%%%%%%%%%%%%%%%%%%%% 사용 예시 %%%%%%%%%%%%%%%%%%%%
% %========================= 설정 =========================%
% [x, fs] = audioread('x.wav');
% % L = 4;                % upsampling 계수
% % gain = L;             % H_r(w)의 gain
% % pass_ratio = 1 / L;   % LPF passband 비율

% M = 14;               % downsampling 계수
% gain = 1;             % H_r(w)의 gain
% pass_ratio = 1 / M;   % LPF passband 비율

% %==================== H_r(w) 생성 및 플롯 ====================%
% H_r = create_Hr_filter(x, gain, pass_ratio);           % 주파수 영역 필터 생성
% H_r_length = length(H_r);
% omega = linspace(-pi, pi, H_r_length);                 % 주파수 축

% % Magnitude plot (linear scale)
% figure;
% plot(omega, abs(H_r), 'LineWidth', 1.5);
% xlabel('\omega (rad/sample)');
% ylabel('Magnitude');
% title('Magnitude Spectrum of H\_r(\omega) (Linear Scale)');
% grid on;

% %==================== h_r[n] 생성 및 windowing ====================%
% h_r_3 = create_windowed_LPF(H_r, gain, 3);
% h_r_5 = create_windowed_LPF(H_r, gain, 5);
% h_r_13 = create_windowed_LPF(H_r, gain, 13);

% %==================== 파형 plot ====================%
% n_3 = -1:1;
% n_5 = -2:2;
% n_13 = -6:6;

% figure;
% subplot(3, 1, 1);
% stem(n_3, h_r_3, 'filled', 'LineWidth', 1.5);
% title('Windowed Filter h\_r\_3[n] with w\_3[n]');
% xlabel('n'); ylabel('Amplitude'); grid on;

% subplot(3, 1, 2);
% stem(n_5, h_r_5, 'filled', 'LineWidth', 1.5);
% title('Windowed Filter h\_r\_5[n] with w\_5[n]');
% xlabel('n'); ylabel('Amplitude'); grid on;

% subplot(3, 1, 3);
% stem(n_13, h_r_13, 'filled', 'LineWidth', 1.5);
% title('Windowed Filter h\_r\_13[n] with w\_13[n]');
% xlabel('n'); ylabel('Amplitude'); grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% 사용 예시 %%%%%%%%%%%%%%%%%%%%
[x, fs] = audioread('x.wav');

L = 4; M = 14;

y = interpolate_then_decimate(x, L, M);

sound(x, fs);
pause(5);
sound(y, fs * L / M);  % 재생 속도 조정
pause(5);

% 시각화
figure;
subplot(2, 1, 1);
plot(0:length(x)-1, x);
xlabel('n'); ylabel('Amplitude');
title('Input x[n] before Rational Sample Rate Conversion');

subplot(2, 1, 2);
plot(0:length(y)-1, y);
xlabel('n'); ylabel('Amplitude');
title('Final output y[n] after Rational Sample Rate Conversion');

Y = fft(y, length(y));
omega_y = linspace(-pi, pi, length(Y));

figure;
plot(omega_y, fftshift(abs(Y)));
xlabel('\omega (rad/sample)');
ylabel('Magnitude');
title('Magnitude Spectrum of y[n]');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Interpolate -> Decimate 예시 %%%%%%%%%%%%%%%%%%%%
function y = interpolate_then_decimate(x, L, M)
    % 보간 -> 공통 LPF -> 다운샘플링 처리

    % Step 1: Interpolation (Zero 삽입)
    x_e = create_x_e(x, L);

    % Step 2: 공통 LPF 생성
    H_r = create_integrated_LPF(x, L, M);

    % Step 3: 시간영역 필터 생성 (예: 13-point 윈도우 적용)
    h = create_windowed_LPF(H_r, L, 13);  % L이 gain

    H = fft(h, length(x_e));
    omega_h = linspace(-pi, pi, length(H));

    %=========================%
    % Ideal 하지 않은 LPF가 생겨, Aliasing 발생가능
    figure;
    plot(omega_h, fftshift(abs(H)));
    xlabel('\omega (rad/sample)');
    ylabel('Magnitude');
    title('Magnitude Spectrum of H_r_windowed');
    grid on;
    %=========================%

    % Step 4: Convolution (필터링)
    x_i = conv(x_e, h, 'same');  % 중간 신호

    %===== Interpolation complete ======%

    % Step 5: Decimation
    y = create_xd_from_xp(x_i, M);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%% windwing 된 LPF 생성 %%%%%%%%%%%%%%%%%%%%
% ex. h_r_13 = create_windowed_LPF(H_r, gain, 13);
function h_r_windowed = create_windowed_LPF(H_r, gain, window_length)
    % create_windowed_LPF - 주파수 응답 H_r(w)를 시간영역 h[n]으로 변환 후,
    %                       길이 window_length의 윈도우를 적용하고 gain에 맞게 정규화
    %
    % 입력:
    %   H_r  : 주파수 영역 필터
    %   gain : 필터 통과대역의 이득값 (정규화를 위한 기준값, ex: L)
    %   window_length : window 길이 (홀수만 허용)
    %
    % 출력:
    %   h_r_windowed : 윈도우가 적용되고 정규화된 시간영역 필터

    % 홀수 조건 확인
    if mod(window_length, 2) == 0
        error('윈도우 길이 N은 반드시 홀수여야 합니다.');
    end

    H_r_length = length(H_r);
    middle_index = floor(H_r_length / 2) + 1;
    half_N = floor(window_length / 2);

    %=================== t-domain filter 계산 ===================%
    H_r_original = ifftshift(H_r);       % 중심이 0인 주파수 정렬
    h_r = ifftshift(ifft(H_r_original)); % t-domain 변환 + 중심 정렬
    %==========================================================%

    %=================== 윈도우 생성 ===================%
    w = create_window(window_length);  % 현재: rectangular window
    %==================================================%

    %=================== windowing 적용 ===================%
    h_r_windowed = h_r(middle_index - half_N : middle_index + half_N) .* w;

    % Normalize: 계수의 총합이 gain 값이 되도록 조정
    h_r_windowed = h_r_windowed * gain / sum(h_r_windowed);
    %=====================================================%

end

%%%%%%%%%%%%%%%%%%%% create_integrated_LPF - 주파수 영역 필터 H_r(w) 생성 %%%%%%%%%%%%%%%%%%%%
function H_r = create_integrated_LPF(x, L, M)
    % integrated sample rate conversion을 위한 공통 LPF 설계
    % 차단 주파수: wc = min(pi/L, pi/M)

    gain = L;
    pass_ratio = min(1/L, 1/M);  % π 기준 정규화된 비율

    H_r = create_Hr_filter(x, gain, pass_ratio);  % 기존 함수 활용
end


%%%%%%%%%%%%%%%%%%%% create_Hr_filter - 주파수 영역 필터 H_r(w) 생성 %%%%%%%%%%%%%%%%%%%%
function H_r = create_Hr_filter(x, gain, pass_ratio)
    % create_Hr_filter - gain과 passband 비율로 H_r(w) 생성
    %
    % 입력:
    %   x          : 입력 신호
    %   gain       : 필터의 pass band에서의 gain 값
    %   pass_ratio : 전체 주파수 중 pass band의 비율 (0 ~ 1)
    %
    % 출력:
    %   H_r : 설계된 주파수 영역 필터 H_r(w)

    % 전체 주파수 길이 설정
    H_r_length = gain * length(x);

    % 필터 초기화
    H_r = zeros(H_r_length, 1);

    % middle_index 계산
    middle_index = floor(H_r_length / 2) + 1;

    % passband의 길이 계산
    passband_width = floor(pass_ratio * H_r_length / 2);

    % 인덱스 범위 설정
    start_index = middle_index - passband_width;
    end_index   = middle_index + passband_width;

    % 지정된 구간에 gain 할당
    H_r(start_index:end_index) = gain;

    figure;
    omega = linspace(-pi, pi, H_r_length);
    plot(omega, abs(H_r), 'LineWidth', 1.5);
    xlabel('\omega (rad/sample)');
    ylabel('Magnitude');
    title('Magnitude Spectrum of H\_r(\omega)');
    grid on;
end

%%%%%%%%%%%%%%%%%%%% window making function %%%%%%%%%%%%%%%%%%%%
% ex. w_5 = create_window(5);
function w = create_window(N)
    w = ones(N, 1);  % 필터 크기만큼 1으로 할당
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%========== From. make_xp_xd_xe_xi_functions.m ==========%

% x_p: M간격으로 원래 신호 할당, 나머지는 0으로 초기화 (zero-padding은 아님!!)
function x_p = create_x_p(x, M)
    % create_x_p - M 간격으로 x[n]을 샘플링하여 x_p[n] 생성
    %
    % 입력:
    %   x : 원본 신호
    %   M : downsampling 계수 (양의 정수)
    %
    % 출력:
    %   x_p : M 간격으로 downsampling된 신호

    % 입력 검증
    if M <= 0 || floor(M) ~= M
        error('M은 양의 정수여야 합니다.');
    end

    % x와 동일한 길이의 0으로 초기화
    x_p = zeros(size(x));

    % M 간격으로 값 복사
    x_p(1:M:end) = x(1:M:end);
end

function x_d = create_xd_from_xp(x_p, M)
    % create_xd_from_xp - M 간격으로 x_p[n]의 유효 샘플을 추출하여 x_d[n] 생성
    %
    % 입력:
    %   x_p : 간헐적으로 샘플이 존재하는 입력 신호
    %   M   : downsampling 계수
    %
    % 출력:
    %   x_d : 추출된 유효 샘플들로 구성된 downsampled 신호

    % 입력 검증
    if M <= 0 || floor(M) ~= M
        error('M은 양의 정수여야 합니다.');
    end

    % x_d 초기화 및 값 추출
    x_d = zeros(ceil(length(x_p)/M), 1);  % 크기 지정
    x_d(1:end) = x_p(1:M:end);            % M 간격으로 추출
end


function x_e = create_x_e(x, L)
    % create_x_e - 신호 x[n]을 L배로 확장하면서 0을 삽입하여 interpolation 준비
    %
    % 입력:
    %   x : 원본 신호 벡터
    %   L : Interpolation 계수 (삽입할 0의 개수 + 1)
    %
    % 출력:
    %   x_e : 0이 삽입되어 L배 확장된 신호

    % 입력 검증
    if L <= 0 || floor(L) ~= L
        error('L은 양의 정수여야 합니다.');
    end

    % 크기 확장 및 0 삽입
    x_e = zeros(L * length(x), 1);
    x_e(1:L:end) = x;
end

% x_i는 x_e[n]과 h_r_windowed[n]의 convolution 결과임 그냥 convolution 하면 됨
% ex. x_i = conv(x_e, h_r_windowed, 'same');