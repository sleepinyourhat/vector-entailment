pq=''
q='.*test data'

for f in scr/tuning6-e-*f2/
do
	echo $f
	t=$(/bin/grep " pass $1 " $f/stat_log | head -c 16)
	echo \"pass $1 \"
	echo $t
	if [ -n "$t" ]; then
		fq=$pq$t$q
		echo $fq
		egrep "$fq" $f/example_log
	fi
done
