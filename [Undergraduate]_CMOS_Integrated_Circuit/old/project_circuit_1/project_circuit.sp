
.inc '45nm_LP.pm'
.inc 'project_circuit.hsp'

v1 vdd 0 vddx
v2 vss 0 vssx

vin a0 0 vssx
vin1 a1 0 vddx
vin2 a2 0 vddx
vin3 a3 0 vddx
vin4 a4 0 vddx
vin5 a5 0 vddx
vin6 a6 0 vddx
vin7 a7 0 vddx
vin8 a8 0 vssx
vin9 a9 0 vssx
vin10 a10 0 vssx
vin11 a11 0 vssx
vin12 a12 0 vssx
vin13 a13 0 vssx
vin14 a14 0 vssx
vin15 a15 0 vddx

vin16 c0 0 vddx
vin17 c1 0 vddx
vin18 c2 0 vddx
vin19 c3 0 vddx
vin20 c4 0 vddx
vin21 c5 0 vddx
vin22 c6 0 vddx
vin23 c7 0 vddx
vin24 c8 0 vddx
vin25 c9 0 vddx
vin26 c10 0 vddx
vin27 c11 0 vddx
vin28 c12 0 vddx
vin29 c13 0 vddx
vin30 c14 0 pulse vssx vddx 1n 0.1n 0.1n 0.5n 1n
vin31 c15 0 vddx

vin32 s0 0 vddx
vin33 s1 0 vddx
vin34 s2 0 vddx
vin35 s3 0 vddx


.option post node list

.param
+vddx = 1.1
+vssx = 0

.tran 5p 10n

.end
