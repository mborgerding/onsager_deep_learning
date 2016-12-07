#!/bin/bash
# A set of functions to be used by bash scripts 
#
#

die() {
    echo $* >&2
    exit 1
}

runlog=data/run$$.log

echolog() {
    echo "$1" |tee -a $runlog
}

#    runcmd cmd base
#
# runs the command `cmd`
# and uses `base` in the following way
# `base`.log : text log file
# `base`.npz : file used for numpy.{savez,load}
# `base`.running : flag file that disappears on successful completion
runcmd() {
    cmd_="$1"
    baseT_="$2"
    logfile_="${baseT_}.log"
    if [ -f ${baseT_}.npz ];then
        if [ -f ${baseT_}.running ];then
            echolog " ${baseT_}.npz exists but so does .running file -- restarting"
        else
            echolog "SKIPPING -- ${baseT_}.npz exists" 
            return
        fi
    fi
    cmd_="$cmd_ --save=${baseT_}.npz --summary=${baseT_}.sum"
    echo "$$" > ${baseT_}.running
    echo "$cmd_" > $logfile_
    echolog "$cmd_" 
    echolog "logfile_=$logfile_" 
    $cmd_ | tee -a $logfile_ || die "died in $cmd_"
    rm ${baseT_}.running
    (echo -n "`date +'%y:%m:%d::%H:%M:%S' ` $baseT_ " && tail -n3 $logfile_) >> $runlog
}

nmse_for() {
    if [ -f ${1}.sum ] ; then
        . ${1}.sum
        echo $nmse_test
    else
        echo 999
    fi
}

less_than() {
    [ `echo "$1 < $2" |bc -l` == 1 ]
}
