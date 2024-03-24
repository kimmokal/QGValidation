#!/bin/bash
if [[ -z "${COFFEAHOME}" ]]; then
  echo "ERROR: Environment has not been activated!"
  return 1
fi

script_path="${COFFEAHOME}/plotting_and_SFs/scripts"

#### Give the configuration file for the processsing as an argument
if [ $# -eq 1 ]; then
    config_file=$1
    echo "Using config: ${1}"
else
    echo "ERROR: Specify config file as an argument: ./run_scripts.sh config/config_file.toml"
    return 1
fi

#### Set flags to choose the processing steps ######
FLAG_PROCESS_VALIDATION_PLOTS=1
FLAG_PROCESS_VALIDATION_PLOTS_FOR_SCALE_FACTORS=1

FLAG_EXTRACT_SCALE_FACTORS=1
FLAG_COMBINE_SCALE_FACTOR_UNCERTAINTIES=1
FLAG_PLOT_SCALE_FACTORS_WITH_TOTAL_UNC=1

FLAG_PROCESS_REWEIGHTED_VALIDATION_PLOTS=1

FLAG_PROCESS_HISTOGRAMS_FOR_ROC_CURVES=1
FLAG_COMBINE_HISTOGRAMS_FOR_ROC_CURVES=1
FLAG_PLOT_ROC_CURVES=1

FLAG_PROCESS_WP_VALIDATION_PLOTS=1
FLAG_EXTRACT_WP_SCALE_FACTORS=1
FLAG_COMBINE_WP_SCALE_FACTORS=1
FLAG_PLOT_WP_SCALE_FACTORS=1
####################################################

if [ ${FLAG_PROCESS_VALIDATION_PLOTS} -eq 1 ] 
then
    for channel in zmm dijet
    do
        for variable in qgl deepjet particlenet axis2 mult ptd
        do
            python ${script_path}/validation_plots.py --config ${config_file} --binning validation -c ${channel} -v ${variable} -s -r
            for syst in gluon fsr isr pu jes jer
            do
                  python ${script_path}/validation_plots.py --config ${config_file} --binning validation -c ${channel} -v ${variable} --syst ${syst} --syst_up -s -r
                  python ${script_path}/validation_plots.py --config ${config_file} --binning validation -c ${channel} -v ${variable} --syst ${syst} --syst_down -s -r
            done
        done
    done
fi

if [ ${FLAG_PROCESS_VALIDATION_PLOTS_FOR_SCALE_FACTORS} -eq 1 ] 
then
    for channel in zmm dijet
    do
        for variable in qgl deepjet particlenet axis2 mult ptd
        do
            python ${script_path}/validation_plots.py --config ${config_file} --binning scalefactors -c ${channel} -v ${variable} -s -r
            for syst in gluon fsr isr pu jes jer
            do
                  python ${script_path}/validation_plots.py --config ${config_file} --binning scalefactors -c ${channel} -v ${variable} --syst ${syst} --syst_up -s -r
                  python ${script_path}/validation_plots.py --config ${config_file} --binning scalefactors -c ${channel} -v ${variable} --syst ${syst} --syst_down -s -r
            done
        done
    done
fi

if [ ${FLAG_EXTRACT_SCALE_FACTORS} -eq 1 ]
then
    for variable in qgl deepjet particlenet 
    do
        for flavour in '-q' '-g'
        do
            for syst in nominal gluon fsr isr pu jes jer
            do 
                python ${script_path}/extract_scale_factors.py --config ${config_file} -s -v ${variable} --syst ${syst} ${flavour} 
            done
        done
    done
fi

if [ ${FLAG_COMBINE_SCALE_FACTOR_UNCERTAINTIES} -eq 1 ] 
then
    for variable in qgl deepjet particlenet
    do
        python ${script_path}/combine_sf_uncertainties.py --config ${config_file} -v ${variable}
    done
fi

if [ ${FLAG_PLOT_SCALE_FACTORS_WITH_TOTAL_UNC} -eq 1 ] 
then
    for variable in qgl deepjet particlenet
    do
        for flavour in '-q' '-g'
        do
             if [[ ${variable} == "qgl" ]]; then
                for do_fit in '' '-f' # Only do fits to QGL SFs
                do
                    python ${script_path}/plot_sfs_with_total_uncertainty.py --config ${config_file} -s -v ${variable} ${flavour} ${do_fit}
                done
             else
                python ${script_path}/plot_sfs_with_total_uncertainty.py --config ${config_file} -s -v ${variable} ${flavour}
             fi
        done
    done
fi

if [ ${FLAG_PROCESS_REWEIGHTED_VALIDATION_PLOTS} -eq 1 ] 
then
    for channel in zmm dijet
    do
        for variable in qgl deepjet particlenet
        do
            python ${script_path}/reweighted_validation_plots.py --config ${config_file} --binning validation -c ${channel} -v ${variable} -s -r
        done
    done
fi

if [ ${FLAG_PROCESS_HISTOGRAMS_FOR_ROC_CURVES} -eq 1 ] 
then
    for channel in zmm dijet
    do
        for variable in qgl deepjet particlenet
        do
            python ${script_path}/mc_hists_for_ROC.py --config ${config_file} -c ${channel} -v ${variable} -s -r
        done
    done
fi

if [ ${FLAG_COMBINE_HISTOGRAMS_FOR_ROC_CURVES} -eq 1 ] 
then
    for variable in qgl deepjet particlenet
    do
        python ${script_path}/combine_mc_hists_for_ROC.py --config ${config_file} -v ${variable}
    done
fi

if [ ${FLAG_PLOT_ROC_CURVES} -eq 1 ] 
then
    python ${script_path}/roc_curve_plot.py --config ${config_file} -s --uncertainty_band
    python ${script_path}/roc_curve_plot.py --config ${config_file} -s --wp
fi

if [ ${FLAG_PROCESS_WP_VALIDATION_PLOTS} -eq 1 ] 
then
    for channel in zmm dijet
    do
        for variable in qgl deepjet particlenet
        do
            for wp in loose medium tight
            do
                python ${script_path}/wp_validation_plots.py --config ${config_file} -c ${channel} -v ${variable} -w ${wp} -s -r
                for syst in gluon fsr isr pu jes jer
                do
                      python ${script_path}/wp_validation_plots.py --config ${config_file} -c ${channel} -v ${variable} -w ${wp} --syst ${syst} --syst_up -s -r
                      python ${script_path}/wp_validation_plots.py --config ${config_file} -c ${channel} -v ${variable} -w ${wp} --syst ${syst} --syst_down -s -r
                done
            done
        done
    done
fi

if [ ${FLAG_EXTRACT_WP_SCALE_FACTORS} -eq 1 ] 
then
    for variable in qgl deepjet particlenet
    do
        for wp in loose medium tight
        do
            for syst in nominal gluon fsr isr pu jes jer
            do
                python ${script_path}/wp_extract_scale_factors.py --config ${config_file} -v ${variable} -w ${wp} --syst ${syst} -s
            done
        done
    done
fi
if [ ${FLAG_COMBINE_WP_SCALE_FACTORS} -eq 1 ] 

then
    for variable in qgl deepjet particlenet
    do
        for wp in loose medium tight
        do
            python ${script_path}/wp_combine_scale_factors.py --config ${config_file} -v ${variable} -w ${wp}
        done
    done
fi

if [ ${FLAG_PLOT_WP_SCALE_FACTORS} -eq 1 ] 
then
    for variable in qgl deepjet particlenet
    do
        for wp in loose medium tight
        do
            for syst in nominal gluon fsr isr pu jes jer total
            do
                python ${script_path}/wp_scale_factor_plot.py --config ${config_file} -v ${variable} -w ${wp} --syst ${syst} -s
            done
        done
    done
fi
