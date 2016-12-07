#!/bin/bash
. utils.sh

cat << EOT
This scipt runs a variety of trials of LAMP and VAMP(aka VAMP).
If interrupted, it will restart the most recent trial.

It runs 30*(2*maxT-1) trials. The number 30 comes from the outer product
  {Gaussian,kappa=15}              # linear operator construction
 x{LAMP,LAMP untied,VAMP}           # algorithm
 x{soft threshold,Bernoulli-Gaussian,Exponential,Piecewise Linear,Spline} #  shrinkage function

 Note this could theoretically be restructured to run 30 independent streams.
 This might be useful if anyone needs to rerun all the experiments.

 Approach:
 for t in 2..T
     the previous system is extended greedily,
        ie. the previous layers are frozen
        and the last layer is trained in isolation via back-propagation.
      After the new layer is trained, all t layers are fine-tuned trained.
EOT

# number of layers
#maxT=5
maxT=15

startArgs="--refinements=4 --stopAfter=20"
for kappaarg in '' '--kappa=15';do

    if [ -z "$kappaarg" ]; then
        midname='Giid'
    else
        midname='k15'
    fi

    # test matched VAMP (not learned)
    for T in {1..20};do
        base=data/VAMP_${midname}_matched_T${T}f
        runcmd "./lamp_vamp.py --matched --T $T $kappaarg" $base
    done

    for typearg in '' '--vamp' '--tieS=false';do
        if [ "$typearg" = "--tieS=false" ]; then
            shrinkfuncs="bg pwlin soft"
        elif [ "$typearg" = "--vamp" ]; then
            shrinkfuncs="bg pwlin"
        else
            shrinkfuncs="bg pwlin expo spline soft"
        fi

        tr1=1e-3
        tr2=1e-4

        for shrink in $shrinkfuncs;do
            if [ -z "$typearg" ]; then
                base=data/LAMP_${midname}_$shrink
            elif [ "$typearg" = "--vamp" ]; then
                base=data/LVAMP_${midname}_$shrink
            elif [ "$typearg" = "--tieS=false" ]; then
                base=data/LAMPut_${midname}_$shrink
            fi
            baseArgs="$startArgs --shrink $shrink $typearg $kappaarg"
            # note at each refinement, the trainRate will be divided by 10 and the stopAfter will be doubled at each refinement
            # so the final refinement will occur with trainRate=1e-7,stopAfter=160

            runcmd "./lamp_vamp.py  --trainRate=$tr1 $baseArgs --T 1 " ${base}_T1f
            loadStateArg="--load=${base}_T1f.npz"

            for T in `seq 2 $maxT`;do
                baseT=${base}_T${T}
                # extend the last layer greedily
                runcmd "./lamp_vamp.py $baseArgs $loadStateArg --trainRate=$tr1 --T $T " $baseT

                loadArg=--load=${baseT}.npz

                # fine tune all layers
                if [ "$typearg" = "--tieS=false" ]; then
                    # If we are fine-tuning untied LAMP,
                    # then compare the nmse of the tied case with the greedy extension above.
                    # Choose the better as a starting point.
                    baseOptA=data/LAMP_${midname}_${shrink}_T${T}f
                    baseOptB=$baseT
                    nmseOptA=`nmse_for $baseOptA`
                    nmseOptB=`nmse_for $baseOptB`
                    echolog "$baseOptA nmse $nmseOptA, $baseOptB nmse $nmseOptB"
                    if less_than $nmseOptA $nmseOptB ;then
                        echolog "starting by untying $baseOptA because $nmseOptA < $nmseOptB"
                        loadArg=--load=${baseOptA}.npz
                    fi
                fi

                runcmd "./lamp_vamp.py $baseArgs ${loadArg} --trainRate=$tr2 --T $T " ${baseT}f
                loadStateArg="--load=${baseT}f.npz"

                # report progress
                #echo "progress report (so far) ... "
                #sumfiles="`find data/ -type f -name 'LAMP_*f.sum' -or -name 'LAMPut_*f.sum' -or -name 'LVAMP_*f.sum'| xargs -r ls -rt ` "
                #grep nmse_val= $sumfiles
                #echo
            done
        done
    done
done
