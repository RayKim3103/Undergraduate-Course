
.inc '45nm_LP.pm'
.inc 'mobility_check_beta.hsp'

* 노드 정의
v1 vdd 0 vddx
v2 vss 0 vssx

* 입력 신호 정의 (DC Sweep)
vin in 0 dc vssx

* 출력 노드 연결 (테스트 목적)
* 예: vout이 in과 관련된 노드라면, 이를 적절히 정의해야 합니다.
* .subckt 내부에 회로가 있다면 해당 정의를 참조해 vout 설정 필요.

.option post
.param
+vddx = 1.1
+vssx = 0

* DC 분석
.dc vin vssx vddx 0.01

* 출력 로그
.print dc v(in) v(out)

.end

