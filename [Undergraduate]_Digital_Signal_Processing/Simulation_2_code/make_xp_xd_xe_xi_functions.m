


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