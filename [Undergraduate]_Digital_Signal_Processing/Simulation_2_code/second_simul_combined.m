clear; % 모든 변수 지우기
clc; % 명령창 지우기
close all; % 모든 figure 창 닫기

%%%%%%%%%%%%%%%%%%%%%%%%%% 1-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-1. 신호 x[n]의 파형을 그리기 & sound

% x.wav 파일 읽기
[x, fs] = audioread('x.wav');                   % x: 신호 데이터, fs: 샘플링 주파수

%========== 변수 setting ==========%
n_domain = (0:length(x)-1);                     % n축 생성
t_domain = (0:length(x)-1)*(1/fs);              % t축 생성

n = length(x);                                  % 신호 길이
%==================================%

% x_zero(1:1:length(x)) = 0;

% % 파형 그리기
% figure;
% plot(n_domain, x);
% % plot(t_domain, x);
% xlabel('n domain');
% % xlabel('time domain (sec)');
% ylabel('Amplitude');
% title('Waveform of x[n]');
% grid on;

% 신호 청취
disp("1-1. playing x")
sound(x, fs);
pause(5);                                           % 5초 대기 (신호 재생 시간)

% 오디오 파일로 저장
audiowrite('original_sound_x.wav', x, fs);

%%%%%%%%%%%%%%%%%%%%%%%%%% 1-2 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% FFT 계산

X = fft(x); % FFT 계산
X_shifted = fftshift(X);                            % FFT 결과를 중앙 정렬

%========== 변수 setting ==========%
X_omega = linspace(-pi, pi, n);                     % w 축 생성 (주파수 범위: -π ~ π)
%==================================%

% Magnitude 계산 및 로그 스케일 변환
X_magnitude = abs(X_shifted);                       % Magnitude 계산
X_magnitude_log = log(1 + X_magnitude);             % 로그 스케일


% % Magnitude 그래프 그리기
% figure;
% plot(X_omega, X_magnitude_log);
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('Magnitude Spectrum of X(ω) in Log Scale');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 1-3 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-3. Interpolation을 하기 전에, 이전 신호 x[n] 사이사이에 0을 삽입하는 과정 (Zero-padding)을 진행

L = 4;                                                  % Interpolation (L=4)

% Matrix size: 4*n 행, 1열
x_e = zeros(L*n, 1);                                    % 0으로 채워진 신호 생성

% x_e의 1, 5, 9, ... 번째에 x[n]의 값을 넣음
x_e(1:L:end) = x;                                       % 원래 신호를 4배 길이로 확장
% for k = 1:n
%     x_e((k-1)*L + 1) = x(k);                            % x[n]의 값을 x_e[n]에 삽입
% end

%========== 변수 setting ==========%
n_domain_e = (0:length(x_e)-1);                         % n_e축 생성
t_domain_e = (0:length(x_e)-1)/(L*fs);                  % t_e축 생성
%==================================%

% for i = 1:length(x_e)
%     if x_e(i) ~= 0
%         disp('x_e[i]');
%         disp(i);
%         disp((i-1)/4 + 1);
%         disp(x_e(i));
%         disp(x_e(i+1));
%         disp(x_e(i+2));
%         disp(x_e(i+3));
%         disp(x_e(i+4));
%         break;
%     end
% end

% for i = 1:length(x)
%     if x(i) ~= 0
%         if mod(i, L) == 1
%             disp('x[i]');
%             disp(i);
%             disp(x(i));
%             disp(x(i+1));
%             disp(x(i+2));
%             disp(x(i+3));
%             disp(x(i+4)); 
%             break;  
%         end
       
%     end
% end


% figure;
% % stem(n_domain_e, x_e); 
% plot(t_domain_e, x_e);
% title('Waveform of x_e[n] (= x[n] with Zero-padding)');
% % xlabel('n domain (0 ~ 4n-1)');
% xlabel('time domain (sec)');
% ylabel('Amplitude');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 1-4 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% disp("1-4. playing x_e")
% sound(x_e, fs*L);                                       % sound signal (fs는 4배 증가)
% pause(5);                                               % 5초 대기 (신호 재생 시간)

% 오디오 파일로 저장
audiowrite('x_e.wav', x_e, L*fs);
%%%%%%%%%%%%%%%%%%%%%%%%%% 1-5 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
X_e = fft(x_e);                                         % FFT 계산
X_e_shifted = fftshift(X_e);                            % FFT 결과를 중앙 정렬
X_e_magnitude = abs(X_e_shifted);                       % Magnitude 계산
X_e_magnitude_log = log(1 + X_e_magnitude);             % 로그 스케일 변환
upsample_omega = linspace(-pi, pi, L*n);                % 주파수 벡터 생성

% figure;
% subplot(2, 1, 1);                                       % 2행 1열의 첫 번째 subplot
% plot(X_omega, X_magnitude_log);                         % X(ω) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('Magnitude Spectrum of X(ω) in Log Scale');
% grid on;
% subplot(2, 1, 2);                                       % 2행 1열의 두 번째 subplot
% plot(upsample_omega, X_e_magnitude_log);                % X_e(ω) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('Magnitude Spectrum of X_e(ω) in Log Scale');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 2-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-1. H_r(w)를 알맞게 설계한 뒤, H_r(w)의 Magnitude 그래프를 log scale이 아닌 일반적인 linear scale로 그리고, 설명

upsample_n = L * length(x);                             % 신호 길이

upsample_omega = linspace(-pi, pi, upsample_n);         % 주파수 벡터 생성

H_r = zeros(upsample_n, 1);                             % H_r(w) 필터 초기화

middle_index = floor(upsample_n / 2) + 1;               % 필터의 중심 인덱스

% H_r의 middle에 L 값을 할당 (주파수 대역에 해당하는 부분)
start_index = middle_index - floor(upsample_n / L / 2); 
end_index = middle_index + floor(upsample_n / L / 2);

H_r(start_index:end_index) = L;

% % Magnitude 그래프 그리기
% figure;
% plot(upsample_omega, H_r);                              % H_r(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (linear scale)');
% title('Magnitude Spectrum of H_r(ω) in Linear Scale');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 2-2 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-2. 주파수 축에서 구현한 H(w)를 시간축에서의 h[n]로 변환한 뒤 아래와 같이 w5[n], w13[n]로 각각 windowing한 h5[n], h13[n]을 구하세요. 
% 그리고 둘의 파형을 그리세요. 
% 단, h5[n], h13[n]의 모든 coefficient 합은 L이 되도록 normalize 해야 합니다.

%=================== h_r[n] ===================%
H_r_original = ifftshift(H_r);              % H_r(w)의 0 rad/sec를 샘플 0에서 시작하도록 변환
h_r = ifftshift(ifft(H_r_original));        % H_r(w)를 n domain으로 변환 후 다시 중심을 이동
%================================================%

%=================== w_5[n]  ===================%
w_5 = create_filter(5);
%================================================%

%=================== w_13[n]  ===================%
w_13 = create_filter(13);
%================================================%

%=================== h_r_5[n]  ===================%
h_r_5 = h_r(middle_index - 2 : middle_index + 2) .* w_5;    % H_r(w)를 시간 영역으로 변환
%================================================%

%=================== h_r_13[n]  ===================%
h_r_13 = h_r(middle_index - 6 : middle_index + 6) .* w_13;  % H_r(w)를 시간 영역으로 변환
%================================================%

%==== Power 보정 : 모든 coefficient 합은 이 되도록 normalize ====%
% disp(sum(h_r))
h_r_5   = h_r_5 * L / sum(h_r_5);
h_r_13  = h_r_13 * L / sum(h_r_13);
% power는 이렇게 아닌가?? 원래??
% h_r_5   = h_r_5  * sqrt(sum(abs(h_r).^2) / sum(abs(h_r_5).^2)); 
% h_r_13  = h_r_13 * sqrt(sum(abs(h_r).^2) / sum(abs(h_r_13).^2));
%================================================%

%================== 변수 setting ================%
n_5_domain  = linspace(-2, 2, length(h_r_5));                               % h_r_5[n]의 n 축
n_13_domain = linspace(-6, 6, length(h_r_13));                              % h_r_13[n]의 n 축
upsample_n_domain = linspace(-upsample_n / 2, upsample_n / 2, length(h_r)); % 전체 upsample_n 축
%================================================%

figure;
stem(upsample_n_domain, h_r);                       % h_r[n] 파형 그래프
xlabel('upsameple n');
ylabel('Amplitude');
title('Waveform of h_r[n]');
grid on;

figure;
subplot(2, 1, 1);
stem(n_5_domain, h_r_5);                            % h_r_5[n] 파형 그래프
xlabel('n_5 domain');
ylabel('Amplitude');
title('Waveform of h_r_5[n]');
grid on;

subplot(2, 1, 2);
stem(n_13_domain, h_r_13);                          % h_r_13[n] 파형 그래프
xlabel('n_1_3 domain');
ylabel('Amplitude');
title('Waveform of h_r_1_3[n]');
grid on;
%================================================%

%%%%%%%%%%%%%%%%%%%%%%%%%% 2-3 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-3. h5[n], h13[n]의 Frequency Response 
% H5(w), H13(w)의 Magnitude를 
% log scale이 아닌 일반적인 linear scale로 그리고, H(w)와 비교하세요.

%== H_r_5의 주파수 응답 및 H_r_13의 주파수 응답을 계산 ==%
H_r_2 = fftshift(fft(h_r, length(H_r)));       
H_r_5 = fftshift(fft(h_r_5, length(H_r)));      
H_r_13 = fftshift(fft(h_r_13, length(H_r)));   
% H_r_5 = fftshift(fft(h_r_5));                 
% H_r_13 = fftshift(fft(h_r_13));               
% H_r_5_omega = linspace(-pi, pi, length(H_r_5));
% H_r_13_omega = linspace(-pi, pi, length(H_r_13));
%================================================%

% figure;
% subplot(3, 1, 1);
% plot(upsample_omega, abs(H_r_2));                 % H_r(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (linear scale)');
% title('Magnitude Spectrum of H_r(ω) in Linear Scale');
% grid on;
% 
% subplot(3, 1, 2); 
% plot(upsample_omega, abs(H_r_5));                 % H_r(w) Magnitude 그래프
% % stem(H_r_5_omega, abs(H_r_5));                  % H_r(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (linear scale)');
% title('Magnitude Spectrum of H_r_5(ω) in Linear Scale');
% grid on;
% 
% subplot(3, 1, 3);
% plot(upsample_omega, abs(H_r_13));                % H_r(w) Magnitude 그래프
% % stem(H_r_13_omega, abs(H_r_13));                % H_r(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (linear scale)');
% title('Magnitude Spectrum of H_r_1_3(ω) in Linear Scale');
% grid on;
%=================================================%

%%%%%%%%%%%%%%%%%%%%%%%%%% 3-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 3-1. 신호 xe[n]을 Reconstruction Filter hr5[n], hr13[n]을 이용하여 
% time domain에서 filtering을 진행한 신호 x5[n], x13[n]을 구하세요. 그리고 둘의 파형을 그리세요.

%========== x_i_5 및 x_i_13을 convolution 계산 ==========%
x_i_5 = conv(x_e, h_r_5, 'same');                           % x[n]을 h_r_5[n]으로 convolution
x_i_13 = conv(x_e, h_r_13, 'same');                         % x[n]을 h_r_13[n]으로 convolution
%=====================================================%

% disp(length(x_i_5));
% disp(length(x_i_13));

%=================== n축 생성 =========================%
% 1-3번에서 정의함
%=====================================================%

% figure;
% subplot(3, 1, 1);
% % plot(n_domain_e, x_e);                      % x_e[n] plot
% stem(t_domain_e, x_e);                        % x_e[n] plot
% % xlabel('n domain (0 ~ 4n-1)');
% xlabel('time domain (sec)');
% ylabel('Amplitude');
% title('Waveform of x_e[n]');
% grid on;

% subplot(3, 1, 2); 
% % plot(n_domain_e, x_i_5);                    % x_i_5[n] plot
% stem(t_domain_e, x_i_5);                      % x_i_5[n] plot
% % xlabel('n domain (0 ~ 4n-1)');
% xlabel('time domain (sec)');
% ylabel('Amplitude');
% title('Waveform of x_i_5[n]');
% grid on;

% subplot(3, 1, 3); 
% % plot(n_domain_e, x_i_13);                   % x_i_13[n] plot
% stem(t_domain_e, x_i_13);                     % x_i_13[n] plot
% % xlabel('n domain (0 ~ 4n-1)');
% xlabel('time domain (sec)');
% ylabel('Amplitude');
% title('Waveform of x_i_1_3[n]');
% grid on;
%=====================================================%


%%%%%%%%%%%%%%%%%%%%%%%%%% 3-2 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
%=================== X_e(w)  ===================%
X_e = fftshift(fft(x_e, upsample_n));                              % x_e[n]의 주파수 응답
%================================================%

%=================== X_i_5(w)  ===================%
X_i_5 = fftshift(fft(x_i_5, upsample_n));                        % x_i_5[n]의 주파수 응답
%================================================%

%=================== X_i_13(w)  ===================%
X_i_13 = fftshift(fft(x_i_13, upsample_n));                      % x_i_13[n]의 주파수 응답
%================================================%

% figure;
% subplot(3, 1, 1); 
% plot(upsample_omega, log(1+ abs(X_e)));               % X(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('Magnitude Spectrum of X_e(ω) in Log Scale');
% grid on;
% 
% subplot(3, 1, 2);
% plot(upsample_omega, log(1+abs(X_i_5)));              % X_i_5(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('Magnitude Spectrum of X_i_5(ω) in Log Scale');
% grid on;
% 
% subplot(3, 1, 3);
% plot(upsample_omega, log(1+abs(X_i_13)));             % X_i_13(w) Magnitude 그래프
% xlabel('ω (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('Magnitude Spectrum of X_i_1_3(ω) in Log Scale');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 3-3 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 3-3. x5[n], x13[n]을 청취하고 신호 xe[n]과 어떤 차이가 있는지 서술하세요.

disp("3-3. playing x_e")
sound(x_e, L*fs);                           % x_e[n] 청취
pause(5); % 5초 대기

disp("3-3. playing x_i_5")
sound(x_i_5, L*fs);                         % x_i_5[n] 청취
pause(5); % 5초 대기

disp("3-3. playing x_i_13")
sound(x_i_13, L*fs);                        % x_i_13[n] 청취
pause(5); % 5초 대기

% 오디오 파일로 저장
audiowrite('x_i_5.wav', x_i_5, L*fs);
% 오디오 파일로 저장
audiowrite('x_i_13.wav', x_i_13, L*fs);

%%%%%%%%%%%%%%%%%%%%%%%%%% 4-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%

% M=14인 경우 xp[n]을 구하고 파형을 그리기
x_p = zeros(size(x_i_13));                  % xp[n] 초기화
M = 14; % M 값 설정
x_p(1:M:end) = x_i_13(1:M:end);             % M 간격으로 x_i_13[n]의 값을 할당

downsample_n = length(x_p);                 % xp[n]의 길이

%========== n_down 축 생성 ==========%
downsample_n_domain = (0:downsample_n-1); 
downsample_t_domain = (0:downsample_n-1)/(L*fs); 
%=================================================%

% figure;
% stem(downsample_n_domain, x_p);               % xp[n] plot
% % plot(downsample_t_domain, x_p);             % xp[n] plot
% xlabel('n domain'); 
% % xlabel('t domain');
% ylabel('Amplitude'); 
% title('xp[n] with M=14'); 

%%%%%%%%%%%%%%%%%%%%%%%%%% 4-2 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 4-2. xp[n]의 Spectrum Xp(w)의 magnitude 그래프를 그리고 설명하세요.

%================= Xp(w) 계산 ==================%
% Xp(w) 계산을 위한 FFT 수행
Xp = fft(x_p); % xp[n]의 FFT
%=================================================%

% figure;
% plot(upsample_omega, abs(Xp));                                % xp[n]의 파형
% xlabel('w (rad/sample)');
% ylabel('Magnitude');                                          
% title('X_p(w)');                                              
% grid on;
%=================================================% 

%%%%%%%%%%%%%%%%%%%%%%%%%% 4-3 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 4-3. xp[n]을 청취하고, 원래 신호 x13[n]과 어떤 차이가 있는지 서술하세요.
% xp[n] =   x_i_13[n] for n = Mk
%           0 otherwise

%========== sound x_i_13 및 x_p =========%
% disp("4-3. playing x_i_13")
% sound(x_i_13, L*fs);                                          % x_i_13[n] 재생
% pause(5);                                                     % 5초 대기

% disp("4-3. playing x_p")
% sound(x_p, L*fs);                                             % xp[n] 재생
% pause(5);                                                     % 5초 대기

% 오디오 파일로 저장
audiowrite('x_p.wav', x_p, L*fs);
%%%%%%%%%%%%%%%%%%%%%%%%%% 4-4 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 4-4. Down-sampling을 하기 위해 문제 3에서 얻은 신호 xp[n]에 대해, 
% 아래의 식을 사용하여 xd[n]의 파형을 그리세요. (M=14)

%============================= xd[n] =============================%
x_d = zeros(ceil(size(x_p)/M));                                 % xd[n] 초기화
x_d(1:1:end) = x_p(1:M:end);                                    % M 간격으로 x_p[n]의 값을 할당

% disp(length(x_d));

decimation_n_domain = (0:length(x_d)-1);                        % n_downsample 축 생성
decimation_t_domain = (0:length(x_d)-1)/(L*fs/M);               % n_downsample 축 생성
%=====================================================% 
 
% figure;
% subplot(2, 1, 1); 
% % plot(downsample_n_domain, x_p);                                 % xp[n] 파형
% plot(downsample_t_domain, x_p);                                 % xp[n] 파형
% % xlabel('n domain'); 
% xlabel('t domain');
% ylabel('Amplitude'); 
% title('xp[n]'); 
% grid on;

% subplot(2, 1, 2); 
% % plot(decimation_n_domain, x_d);                                 % xd[n] 파형
% plot(decimation_t_domain, x_d);                                 % xd[n] 파형
% % xlabel('n downsample domain'); 
% xlabel('t domain');
% ylabel('Amplitude'); 
% title('xd[n]'); 
% grid on;
%=====================================================%

%%%%%%%%%%%%%%%%%%%%%%%%%% 4-5 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 4-5. xd[n]의 Spectrum Xd(w)의 magnitude 그래프를 그리세요.

%============================= Xd(w) =============================%
Xd = fftshift(fft(x_d));                                        % xd[n]의 FFT 계산
%=====================================================%

%=================== Xd(w) magnitude plot ===================%
downsample_omega = linspace(-pi, pi, length(Xd));               % w 축 생성
%==================================================%

% figure;
% plot(downsample_omega, abs(Xd));                              % Xd(w)의 magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('Xd(w)');
% grid on;
%==================================================%

%%%%%%%%%%%%%%%%%%%%%%%%%% 4-6 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
%========== sound(x_d, L*fs/M) 및 sound(x_p, L*fs) ==========%
% disp("4-6. playing xd")
% sound(x_d, L*fs/M);                                           % xd[n]을 재생
% pause(5); % 5초 대기

% disp("4-6. playing xp")
% sound(x_p, L*fs);                                             % xp[n]을 재생
% pause(5);

% 오디오 파일로 저장
audiowrite('x_d.wav', x_d, L*fs/M);
%==================================================%

%%%%%%%%%%%%%%%%%%%%%%%%%% 5-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
%=============== Ha(w) 디자인 =================%
H_a = zeros(upsample_n, 1);                                     % H_r(w) 필터 초기화

middle_index = floor(upsample_n / 2) + 1;                       % 필터의 중심 인덱스

% H_a의 중앙 부분에 L 값을 할당 (주파수 대역에 해당하는 부분)
start_index = middle_index - floor(upsample_n / M / 2); 
end_index = middle_index + floor(upsample_n / M / 2);

% disp(middle_index);
% disp(start_index);
% disp(end_index);

H_a(start_index:end_index) = 1;                                 % 중앙 대역에 L 값 할당
%==================================================%
% 
% figure;
% plot(upsample_omega, H_a);                                    % Ha(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('Ha(w) Magnitude Plot');
% grid on;
%==================================================%

%%%%%%%%%%%%%%%%%%%%%%%%%% 5-2 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
X_a = H_a .* X_i_13;                % Xa(w) = multiply(Ha(w), X_i_13(w))
x_a = (ifft(ifftshift(X_a)));       % xa[n] = IDFT(Xa(w))

% n_a = (0:upsample_n-1);           % n_a 축 생성

% figure;
% % plot(n_domain_e, x_a);
% plot(t_domain_e, x_a);
% % xlabel('n domain (upsampled)');
% xlabel('t domain');
% ylabel('Amplitude');
% title('x_a[n]');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 5-3 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% X_a = H_a .* X_i_13;                                  % Xa(w) = multiply(Ha(w), X_i_13(w))

% figure;
% subplot(3,1,1);
% plot(upsample_omega, log(1+abs(X_a)));                % Ha(w) magnitude plot
% % plot(upsample_omega, abs(X_a));                     % Ha(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('X_a(w) magnitude plot');
% grid on;
% 
% subplot(3,1,2);
% plot(X_omega, X_magnitude_log);                       % X(w) magnitude plot
% % plot(X_omega, X_magnitude);                         % X(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('X(w) magnitude plot');
% grid on;
% 
% subplot(3,1,3);
% plot(upsample_omega, log(1+abs(X_i_13)));             % X_i_13(w) magnitude plot
% % plot(upsample_omega, abs(X_i_13));                  % X_i_13(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude (log scale)');
% title('X_i_1_3(w) magnitude plot');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 5-4 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
x_f = zeros(ceil(size(x_a)/M));                 % xf[n] 초기화
x_f(1:1:end) = x_a(1:M:end);                    % M 간격으로 x_a[n]의 값을 할당
%=====================================================%

% figure;
% % plot(decimation_n_domain, x_f);                 % xf[n] plot
% plot(decimation_t_domain, x_f);                   % xf[n]의 t-domain plot
% % xlabel('n domain (decimation)');
% xlabel('t domain (sec)');
% ylabel('Amplitude');
% title('xf[n] (Downsampled Signal)');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 5-5 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
X_f = fftshift(fft(x_f));                       % xf[n]의 FFT 계산
magnitude_Xf = abs(X_f);                        % xf[n]의 magnitude 계산
log_magnitude_Xf = log(1 + magnitude_Xf);       % log scale로 변환
%=====================================================%

% figure;
% subplot(2, 1, 1);
% plot(downsample_omega, log(1+abs(Xd)));       % Xd(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('Xd(w) Magnitude');
% grid on;

% subplot(2, 1, 2);
% plot(downsample_omega, log_magnitude_Xf);     % Xf(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('Xf(w) Magnitude');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%% 5-6 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("5-6. playing xd[n]");
sound(x_d, L*fs/M);                           % xd[n] 청취
pause(5);                                     % 5초 대기

disp("5-6. playing xf[n]");
sound(x_f, L*fs/M);                           % xf[n] 청취
pause(5);

% 오디오 파일로 저장
audiowrite('x_f.wav', x_f, L*fs/M);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Discussion 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% 6-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 만약 Interpolation을 선행하지 않고 decimation 진행

% Anti-Alias Filter 사용하여 design
%=============== Ha2(w) 디자인 =================%
H_a2 = zeros(n, 1);                                    % H_r(w) 필터 초기화

middle_index = floor(n / 2) + 1;                       % 필터의 중심 인덱스

% H_a의 중앙 부분에 L 값을 할당 (주파수 대역에 해당하는 부분)
start_index = middle_index - floor(n / M / 2); 
end_index = middle_index + floor(n / M / 2);

% disp(middle_index);
% disp(start_index);
% disp(end_index);

H_a2(start_index:end_index) = 1;                                 % 중앙 대역에 L 값 할당
%==================================================%

%================== X_a2(w) 계산 ==================%
X_a2 = H_a2 .* fftshift(X);                         % Xa(w) = multiply(Ha(w), X_i_13(w))
x_a2 = (ifft(ifftshift(X_a2)));                     % xa[n] = IDFT(Xa(w))

% figure;
% plot(X_omega, log(1+abs(fftshift(X))));
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('X(w) magnitude plot');
% grid on;

% figure;
% subplot(2, 1, 1);
% % plot(n_domain, x_a2);
% plot(t_domain, x_a2);
% xlabel('t domain');
% ylabel('Amplitude');
% title('x_a2[n] (After Decimation)');
% grid on;

% subplot(2, 1, 2);
% plot(X_omega, log(1+abs(X_a2)));                  % Ha(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('X_a2(w) magnitude plot');
% grid on;
%=====================================================%

%================== x_f2[n] 계산 ==================%
x_f2 = zeros(ceil(size(x_a2)/M));                   % xf[n] 초기화 ************************ why ceil??
x_f2(1:1:end) = x_a2(1:M:end);                      % M 간격으로 x_a[n]의 값을 할당
%=====================================================%

%================== x_f2(w) 계산 ==================%
X_f2 = fftshift(fft(x_f2));                         % xf[n]의 FFT 계산
magnitude_Xf2 = abs(X_f2);                          % xf[n]의 magnitude 계산
log_magnitude_Xf2 = log(1 + magnitude_Xf2);         % log scale로 변환
%=====================================================%

%================== x_f2[n] 및 X_f2(w)의 축 생성 ===============%
nointerpolation_n_domain = (0:length(x_f2)-1);                  % n_f2 축 생성
nointerpolation_t_domain = (0:length(x_f2)-1)/(fs/M);           % t_f2 축 생성
noiterpolation_omega = linspace(-pi, pi, length(X_f2));         % omega 축 생성
%========================================================%

% figure;
% subplot(2, 1, 1);
% % plot(nointerpolation_n_domain, x_f2);                       % xf2[n]의 n-domain plot
% plot(nointerpolation_t_domain, x_f2);                         % xf2[n]의 t-domain plot
% xlabel('t domain (decimation)');
% ylabel('Amplitude');
% title('xf2[n] (Downsampled Signal)');
% grid on;

% subplot(2, 1, 2);
% plot(noiterpolation_omega, log_magnitude_Xf2);                % Xf(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('X_f_2(w) Magnitude');
% grid on;
%==========================================================%

% disp("6-1. playing xf2[n]");
% sound(x_f2, fs/M);                                              % xf2[n] 청취fs/M
% pause(5);                                                       % 5초 대기

%%%%%%%%%%%%%%%%%%%%%%%%%% 6-2 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% 만약 Interpolation을 선행하지 않고 decimation 진행

% Anti-Alias Filter 사용 안하고 design
%=============== x_p3(n) 디자인 ==================%
x_p3 = zeros(size(x));
x_p3(1:M:end) = x(1:M:end);                                     % x[n]을 M 간격으로 할당
%=====================================================%

%=================== X_p3(w) 계산 ==================%
X_p3 = fftshift(fft(x_p3));                                     % Xp(w) 계산
%==================================================%

% figure;
% subplot(2, 1, 1);
% % plot(n_domain, x_p3);
% plot(t_domain, x_p3);
% xlabel('t domain');
% ylabel('Amplitude');
% title('x_p_3[n] (Downsampled Signal)');
% grid on;

% subplot(2, 1, 2);
% plot(X_omega, log(1+abs(X_p3)));                              % Xp(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('X_p_3(w) magnitude plot');
% grid on;

%================== x_f3[n] 계산 ==================%
x_f3 = zeros(ceil(size(x)/M));                                  % xf[n] 초기화 ************ ceil 안하면 오류남 왜지?
x_f3(1:1:end) = x(1:M:end);                                     % M 간격으로 x_a[n]의 값을 할당
%=====================================================%

%================== x_f3(w) 계산 ==================%
X_f3 = fftshift(fft(x_f3));                                     % xf[n]의 FFT 계산
magnitude_Xf3 = abs(X_f3);                                      % xf[n]의 magnitude 계산
log_magnitude_Xf3 = log(1 + magnitude_Xf3);                     % log scale로 변환
%=====================================================%

%================== x_f3[n] 및 X_f3(w)의 축 생성 ===============%
nointerpolation_n_domain = (0:length(x_f3)-1);                  % n_f2 축 생성
nointerpolation_t_domain = (0:length(x_f3)-1)/(fs/M);           % t_f2 축 생성
noiterpolation_omega = linspace(-pi, pi, length(X_f3));         % 주파수 벡터 생성
%========================================================%

% figure;
% subplot(2, 1, 1);
% % plot(nointerpolation_n_domain, x_f3);
% plot(nointerpolation_t_domain, x_f3); 
% xlabel('t domain (decimation)');
% ylabel('Amplitude');
% title('x_f_3[n] (Downsampled Signal)');
% grid on;

% subplot(2, 1, 2);
% plot(noiterpolation_omega, log_magnitude_Xf3);                % Xf(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('X_f_3(w) Magnitude');
% grid on;
%==========================================================%

% disp("6-2. playing xf3[n]");
% sound(x_f3, fs/M);                                              % xf2[n] 청취fs/M
% pause(5);                                                       % 5초 대기

%%%%%%%%%%%%%%%%%%%%%%%%%% 7-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% L=4, M=4인 경우 원신호로 되돌아가는지 

M2 = 4;

%=============== Ha(w) 디자인 =================%
H_a4 = zeros(upsample_n, 1);                                     % H_r(w) 필터 초기화

middle_index = floor(upsample_n / 2) + 1;                        % 필터의 중심 인덱스

% H_a의 중앙 부분에 L 값을 할당 (주파수 대역에 해당하는 부분)
start_index = middle_index - floor(upsample_n / M2 / 2); 
end_index = middle_index + floor(upsample_n / M2 / 2);

H_a4(start_index:end_index) = 1;                                 % 중앙 대역에 L 값 할당
%==================================================%
% 
% figure;
% plot(upsample_omega, H_a4); % Ha(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('Ha(w) Magnitude Plot');
% grid on;
%==================================================%

X_a = H_a4 .* X_i_13;                                           % Xa(w) = multiply(Ha(w), X_i_13(w))
x_a = (ifft(ifftshift(X_a)));                                   % xa[n] = IDFT(Xa(w))

x_f4 = zeros(ceil(size(x_a)/M2));                               % xf[n] 초기화
x_f4(1:1:end) = x_a(1:M2:end);                                  % M 간격으로 x_a[n]의 값을 할당
%=====================================================%

%====================변수 setting======================%
x_f4_n_domain = (0:length(x_f4)-1);
x_f4_t_domain = (0:length(x_f4)-1)/fs;
X_f4_omega = linspace(-pi, pi, length(x_f4));

X_f4 = fftshift(fft(x_f4));                                     % xf[n]의 FFT 계산
magnitude_Xf4 = abs(X_f4);                                      % xf[n]의 magnitude 계산
log_magnitude_Xf4 = log(1 + magnitude_Xf4);                     % log scale로 변환


% figure;
% subplot(2, 1, 1);
% plot(x_f4_t_domain, x_f4); % xf[n]의 t-domain plot
% xlabel('t domain (decimation)');
% ylabel('Amplitude');
% title('xf4[n] (Downsampled Signal ?= original signal)');
% 
% subplot(2, 1, 2);
% plot(X_f4_omega, log_magnitude_Xf4);                            % Xf(w) magnitude plot
% xlabel('w (rad/sample)');
% ylabel('Magnitude');
% title('Xf4(w) Magnitude');
% grid on;

%==================== 소리 청취 =========================%
% disp("7-1.playing xf4[n]");
% sound(x_f4, L*fs/M2);                                           % xf[n] 청취
% pause(5);

%%%%%%%%%%%%%%%%%%%%%%%%%% 8-1 시작 %%%%%%%%%%%%%%%%%%%%%%%%%%
% Decimation 먼저 하고, Interpolation 진행

clear; % 모든 변수 지우기
clc; % 명령창 지우기
close all; % 모든 figure 창 닫기

% x.wav 파일 읽기
[x, fs] = audioread('x.wav');                       % x: 신호 데이터, fs: 샘플링 주파수

%==================== X(w) 구하기 ====================%
X = fft(x);                                         % FFT 계산
X_shifted = fftshift(X);                            % FFT 결과를 중앙 정렬

%==================== 변수 setting ====================%
n = length(x);                                  % x[n]의 길이
n_domain = (0:length(x)-1);                     % n축 생성
t_domain = (0:length(x)-1)*(1/fs);              % t축 생성
X_omega = linspace(-pi, pi, length(X));    
%==================================================%

%==================== Decimation ====================%
M = 14;
%==================== Ha(w) 디자인 ====================%
H_a = zeros(n, 1);                                     % H_r(w) 필터 초기화

middle_index = floor(n / 2) + 1;                       % 필터의 중심 인덱스

% H_a의 중앙 부분에 L 값을 할당 (주파수 대역에 해당하는 부분)
start_index = middle_index - floor(n / M / 2); 
end_index = middle_index + floor(n / M / 2);

H_a(start_index:end_index) = 1;                                 % 중앙 대역에 L 값 할당
%================= X_a에 LPF 씌위고 IFFT =================%

X_a = H_a .* X_shifted;             % Xa(w) = multiply(Ha(w), X(w))
x_a = (ifft(ifftshift(X_a)));       % xa[n] = IDFT(Xa(w))

%=====================================================%
figure;
plot(X_omega, log(1+abs(X_a))); % X(w) magnitude plot
xlabel('w (rad/sample)');
ylabel('Magnitude');
title('X_a(w) Magnitude');
grid on;

%=============== Down Sampling 된 x_f 생성 ================%
x_f = zeros(ceil(size(x_a)/M));                 % xf[n] 초기화
x_f(1:1:end) = x_a(1:M:end);                    % M 간격으로 x_a[n]의 값을 할당

downsample_n = length(x_f);                     % xp[n]의 길이

%==================== Xf(w) 생성 ======================%
X_f = fft(x_f);                         % xf[n]의 FFT 계산
X_f_shifted = fftshift(X_f);            % FFT 결과를 중앙 정렬

%==================== n_down 축 및 w 축 생성 ====================%
downsample_n_domain = (0:downsample_n-1); 
downsample_t_domain = (0:downsample_n-1)/(fs/M); 
downsample_omega = linspace(-pi, pi, length(x_f)); 
%==================================================%

figure;
subplot(2, 1, 1);
plot(downsample_t_domain, x_f);                         % xp[n] plot
xlabel('t domain');
ylabel('Amplitude');
title('8-1. xf[n] (Downsampled Signal)');
grid on;

subplot(2, 1, 2);
plot(downsample_omega, log(1+abs(X_f_shifted)));        % Xd(w) magnitude plot
xlabel('w (rad/sample)');
ylabel('Magnitude');
title('8-1. Xf(w) Magnitude');
grid on;

%===================== Interpolation =====================%
L = 4;

%=================== Zero Padding x_e[n]생성 =====================%
upsample_n = downsample_n * L;                          % upsampled x[n]의 길이

x_e = zeros(upsample_n, 1);                             % x_e[n] 초기화
x_e(1:L:end) = x_f;                                     % L 간격으로 x_f[n]의 값을 할당

%=================== X_e(w) 생성 =====================%
X_e = fft(x_e);                                         % x_e[n]의 FFT 계산
X_e_shifted = fftshift(X_e);                            % FFT 결과를 중앙 정렬

%=========== gain L인 reconstruct LPF 생성 ===========%
H_r = zeros(upsample_n, 1);                             % H_r(w) 필터 초기화
middle_index = floor(upsample_n / 2) + 1;               % 필터의 중심 인덱스
start_index = middle_index - floor(upsample_n / L / 2); % 시작 인덱스
end_index = middle_index + floor(upsample_n / L / 2);   % 끝 인덱스
H_r(start_index:end_index) = L;                         % 중앙 대역에 L 값 할당

%================== H_r(w)를 IFFT ======================%
H_r_original = ifftshift(H_r);                    
h_r = ifftshift(ifft(H_r_original));                        % h_r[n] = IDFT(H_r(w))

%================== 13point windowing ======================%
w_13 = create_filter(13); % 13point windowing
h_r_13 = h_r(middle_index - 6 : middle_index + 6) .* w_13;  % 13point windowing 적용
h_r_13 = h_r_13 * L / sum(h_r_13); % 정규화
%==================================================%

n_13_domain = linspace(-6, 6, length(h_r_13));
figure;
stem(n_13_domain, h_r_13); % h_r[n] plot
xlabel('n domain (13 point windowing)');
ylabel('Amplitude');
title('h_r[n] (13 point windowing)');
grid on;

%======== convolution [x_e * h_r_13] =============%
x_i_13 = conv(x_e, h_r_13, 'same'); % x[n]을 h_r_13[n]으로 convolution

%================= x_i_13의 FFT =======================%
X_i_13 = fft(x_i_13);                   % x_i_13[n]의 FFT 계산
X_i_13_shifted = fftshift(X_i_13);      % FFT 결과를 중앙 정렬

%=========interpolation 신호 domain 생성 =============%
upsample_n_domain = (0:upsample_n-1);
upsample_t_domain = (0:upsample_n-1)/(L*fs/M);
upsample_omega = linspace(-pi, pi, upsample_n); % 주파수 벡터 생성
%==================================================%
figure;
subplot(2, 1, 1);
plot(upsample_t_domain, x_i_13); % x_i_13[n] plot
xlabel('t domain (upsampled)');
ylabel('Amplitude');
title('8-1. x_i_1_3[n] (Upsampled Signal)');
grid on;

subplot(2, 1, 2);
plot(upsample_omega, log(1+abs(X_i_13_shifted)));
xlabel('w (rad/sample)');
ylabel('Magnitude');
title('8-1. X_i_1_3(w) Magnitude');
grid on;

%================== 소리 청취 ===================%
disp("8-1. playing x_i_13[n]");
sound(x_i_13, L*fs/M);
pause(5);
%==================================================%
%==================================================%



% window making function
function w = create_filter(N)
    w = ones(N, 1);  % 필터 크기만큼 1으로 초기화
end