
.inc '45nm_LP.pm'
.inc 'repeater3_1st.hsp'

v1 vdd 0 vddx
v2 vss 0 vssx

vin in 0 pulse vssx vddx 1n 0.1n 0.1n 0.5n 1n

.option post node list

.param
+vddx = 1.1
+vssx = 0

.tran 5p 10n

.end

