import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import tomllib
import uproot
import hist
import sys

from pathlib import Path
from optparse import OptionParser
from matplotlib.ticker import FixedLocator

hep.style.use('CMS')
matplotlib.rcParams['font.size'] = 20

def main(config_path, variable, working_point):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n-------------------------------------')
    print(f'COMBINING WORKING POINT SCALE FACTORS')
    print(f'-------------------------------------')
    print(f'VARIABLE:    {variable}')
    print(f'WORKING POINT:  {working_point}')

    input_file_path = Path(config['path']['input'])
    print(f'READING SCALE FACTORS FROM: {input_file_path}/wp_sf_weights/')

    output_path = config['path']['output']
    weight_save_path = Path(f'{output_path}/wp_sf_weights')
    weight_save_path.mkdir(parents=True, exist_ok=True)
    print(f'SAVING COMBINED SCALE FACTORS TO: {weight_save_path}/')

    wp_bins = config['WP_binning']['bins']
    for wp_bin in wp_bins:
        eta_bin = wp_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bins = wp_bin['pT']
        for i in range(len(pt_bins)-1):
            pt_low = pt_bins[i]
            pt_high = pt_bins[i+1]
            pt_range = str(pt_low)+'to'+str(pt_high)
            quark_weights_dict = {}
            gluon_weights_dict = {}

            systematics = ['nominal', 'fsr', 'isr', 'gluon', 'pu', 'jes', 'jer']
            for syst in systematics:
                if syst == 'nominal':
                    quark_weights_dict[f'nominal'] = []
                    gluon_weights_dict[f'nominal'] = []
                    quark_weights_dict[f'stat_up'] = []
                    gluon_weights_dict[f'stat_up'] = []
                    quark_weights_dict[f'stat_down'] = []
                    gluon_weights_dict[f'stat_down'] = []
                else:
                    quark_weights_dict[f'{syst}_syst_up'] = []
                    gluon_weights_dict[f'{syst}_syst_up'] = []
                    quark_weights_dict[f'{syst}_syst_down'] = []
                    gluon_weights_dict[f'{syst}_syst_down'] = []

                campaign = config['campaign']
                weight_path = Path(f'{input_file_path}/wp_sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP_{syst}_syst')

                if syst == 'nominal':
                    with uproot.open(weight_path.with_suffix('.root')) as sf_file:
                        quark_weights_dict['nominal'].append(sf_file[f'{variable}_quark_weights'].to_hist())
                        gluon_weights_dict['nominal'].append(sf_file[f'{variable}_gluon_weights'].to_hist())

                    wp_antiselected_quark_weights = [hist.values()[0] for hist in quark_weights_dict['nominal']][0]
                    wp_antiselected_quark_weights_variances = [hist.variances()[0] for hist in quark_weights_dict['nominal']][0]
                    wp_selected_quark_weights = [hist.values()[1] for hist in quark_weights_dict['nominal']][0]
                    wp_selected_quark_weights_variances = [hist.variances()[1] for hist in quark_weights_dict['nominal']][0]

                    wp_antiselected_gluon_weights = [hist.values()[0] for hist in gluon_weights_dict['nominal']][0]
                    wp_antiselected_gluon_weights_variances = [hist.variances()[0] for hist in gluon_weights_dict['nominal']][0]
                    wp_selected_gluon_weights = [hist.values()[1] for hist in gluon_weights_dict['nominal']][0]
                    wp_selected_gluon_weights_variances = [hist.variances()[1] for hist in gluon_weights_dict['nominal']][0]
                else:
                    with uproot.open(weight_path.with_suffix('.root')) as sf_file:
                        quark_weights_dict[f'{syst}_syst_up'].append(sf_file[f'{variable}_quark_weights_syst_up'].to_hist())
                        gluon_weights_dict[f'{syst}_syst_up'].append(sf_file[f'{variable}_gluon_weights_syst_up'].to_hist())
                        quark_weights_dict[f'{syst}_syst_down'].append(sf_file[f'{variable}_quark_weights_syst_down'].to_hist())
                        gluon_weights_dict[f'{syst}_syst_down'].append(sf_file[f'{variable}_gluon_weights_syst_down'].to_hist())

                    quark_weights_dict[f'{syst}_syst_up_antiselected'] = [hist.values()[0] for hist in quark_weights_dict[f'{syst}_syst_up']][0]
                    quark_weights_dict[f'{syst}_syst_up_selected'] = [hist.values()[1] for hist in quark_weights_dict[f'{syst}_syst_up']][0]
                    quark_weights_dict[f'{syst}_syst_down_antiselected'] = [hist.values()[0] for hist in quark_weights_dict[f'{syst}_syst_down']][0]
                    quark_weights_dict[f'{syst}_syst_down_selected'] = [hist.values()[1] for hist in quark_weights_dict[f'{syst}_syst_down']][0]
                    gluon_weights_dict[f'{syst}_syst_up_antiselected'] = [hist.values()[0] for hist in gluon_weights_dict[f'{syst}_syst_up']][0]
                    gluon_weights_dict[f'{syst}_syst_up_selected'] = [hist.values()[1] for hist in gluon_weights_dict[f'{syst}_syst_up']][0]
                    gluon_weights_dict[f'{syst}_syst_down_antiselected'] = [hist.values()[0] for hist in gluon_weights_dict[f'{syst}_syst_down']][0]
                    gluon_weights_dict[f'{syst}_syst_down_selected'] = [hist.values()[1] for hist in gluon_weights_dict[f'{syst}_syst_down']][0]

            wp_selected_quark_weights_total_up_variation = []
            wp_selected_quark_weights_total_down_variation = []
            wp_selected_gluon_weights_total_up_variation = []
            wp_selected_gluon_weights_total_down_variation = []
            wp_antiselected_quark_weights_total_up_variation = []
            wp_antiselected_quark_weights_total_down_variation = []
            wp_antiselected_gluon_weights_total_up_variation = []
            wp_antiselected_gluon_weights_total_down_variation = []

            wp_selected_quark_weights_total_variances = [[],[]]
            wp_selected_gluon_weights_total_variances = [[],[]]
            wp_antiselected_quark_weights_total_variances = [[],[]]
            wp_antiselected_gluon_weights_total_variances = [[],[]]

            ### SELECTED QUARK WEIGHTS
            wp_selected_quark_weight_nominal_value = wp_selected_quark_weights
            wp_selected_quark_weight_statistical_up_value = np.sqrt(wp_selected_quark_weights_variances)
            wp_selected_quark_weight_statistical_down_value = np.sqrt(wp_selected_quark_weights_variances)

            wp_selected_quark_weight_gluon_syst_up_value = quark_weights_dict['gluon_syst_up_selected']
            wp_selected_quark_weight_gluon_syst_down_value = quark_weights_dict['gluon_syst_down_selected']
            wp_selected_quark_weight_fsr_syst_up_value = quark_weights_dict['fsr_syst_up_selected']
            wp_selected_quark_weight_fsr_syst_down_value = quark_weights_dict['fsr_syst_down_selected']
            wp_selected_quark_weight_isr_syst_up_value = quark_weights_dict['isr_syst_up_selected']
            wp_selected_quark_weight_isr_syst_down_value = quark_weights_dict['isr_syst_down_selected']
            wp_selected_quark_weight_pu_syst_up_value = quark_weights_dict['pu_syst_up_selected']
            wp_selected_quark_weight_pu_syst_down_value = quark_weights_dict['pu_syst_down_selected']
            wp_selected_quark_weight_jes_syst_up_value = quark_weights_dict['jes_syst_up_selected']
            wp_selected_quark_weight_jes_syst_down_value = quark_weights_dict['jes_syst_down_selected']
            wp_selected_quark_weight_jer_syst_up_value = quark_weights_dict['jer_syst_up_selected']
            wp_selected_quark_weight_jer_syst_down_value = quark_weights_dict['jer_syst_down_selected']

            wp_selected_quark_weight_gluon_syst_up_difference = wp_selected_quark_weight_gluon_syst_up_value - wp_selected_quark_weight_nominal_value
            wp_selected_quark_weight_gluon_syst_down_difference = wp_selected_quark_weight_gluon_syst_down_value - wp_selected_quark_weight_nominal_value 
            wp_selected_quark_weight_fsr_syst_up_difference = wp_selected_quark_weight_fsr_syst_up_value - wp_selected_quark_weight_nominal_value
            wp_selected_quark_weight_fsr_syst_down_difference = wp_selected_quark_weight_fsr_syst_down_value - wp_selected_quark_weight_nominal_value 
            wp_selected_quark_weight_isr_syst_up_difference = wp_selected_quark_weight_isr_syst_up_value - wp_selected_quark_weight_nominal_value
            wp_selected_quark_weight_isr_syst_down_difference = wp_selected_quark_weight_isr_syst_down_value - wp_selected_quark_weight_nominal_value 
            wp_selected_quark_weight_pu_syst_up_difference = wp_selected_quark_weight_pu_syst_up_value - wp_selected_quark_weight_nominal_value
            wp_selected_quark_weight_pu_syst_down_difference = wp_selected_quark_weight_pu_syst_down_value - wp_selected_quark_weight_nominal_value 
            wp_selected_quark_weight_jes_syst_up_difference = wp_selected_quark_weight_jes_syst_up_value - wp_selected_quark_weight_nominal_value
            wp_selected_quark_weight_jes_syst_down_difference = wp_selected_quark_weight_jes_syst_down_value - wp_selected_quark_weight_nominal_value 
            wp_selected_quark_weight_jer_syst_up_difference = wp_selected_quark_weight_jer_syst_up_value - wp_selected_quark_weight_nominal_value
            wp_selected_quark_weight_jer_syst_down_difference = wp_selected_quark_weight_jer_syst_down_value - wp_selected_quark_weight_nominal_value 

            wp_selected_quark_weight_gluon_syst_up_variation = np.max([wp_selected_quark_weight_gluon_syst_up_difference, wp_selected_quark_weight_gluon_syst_down_difference, 0])
            wp_selected_quark_weight_gluon_syst_down_variation = np.min([wp_selected_quark_weight_gluon_syst_up_difference, wp_selected_quark_weight_gluon_syst_down_difference, 0])
            wp_selected_quark_weight_fsr_syst_up_variation = np.max([wp_selected_quark_weight_fsr_syst_up_difference, wp_selected_quark_weight_fsr_syst_down_difference, 0])
            wp_selected_quark_weight_fsr_syst_down_variation = np.min([wp_selected_quark_weight_fsr_syst_up_difference, wp_selected_quark_weight_fsr_syst_down_difference, 0])
            wp_selected_quark_weight_isr_syst_up_variation = np.max([wp_selected_quark_weight_isr_syst_up_difference, wp_selected_quark_weight_isr_syst_down_difference, 0])
            wp_selected_quark_weight_isr_syst_down_variation = np.min([wp_selected_quark_weight_isr_syst_up_difference, wp_selected_quark_weight_isr_syst_down_difference, 0])
            wp_selected_quark_weight_pu_syst_up_variation = np.max([wp_selected_quark_weight_pu_syst_up_difference, wp_selected_quark_weight_pu_syst_down_difference, 0])
            wp_selected_quark_weight_pu_syst_down_variation = np.min([wp_selected_quark_weight_pu_syst_up_difference, wp_selected_quark_weight_pu_syst_down_difference, 0])
            wp_selected_quark_weight_jes_syst_up_variation = np.max([wp_selected_quark_weight_jes_syst_up_difference, wp_selected_quark_weight_jes_syst_down_difference, 0])
            wp_selected_quark_weight_jes_syst_down_variation = np.min([wp_selected_quark_weight_jes_syst_up_difference, wp_selected_quark_weight_jes_syst_down_difference, 0])
            wp_selected_quark_weight_jer_syst_up_variation = np.max([wp_selected_quark_weight_jer_syst_up_difference, wp_selected_quark_weight_jer_syst_down_difference, 0])
            wp_selected_quark_weight_jer_syst_down_variation = np.min([wp_selected_quark_weight_jer_syst_up_difference, wp_selected_quark_weight_jer_syst_down_difference, 0])

            ### SELECTED GLUON WEIGHTS
            wp_selected_gluon_weight_nominal_value = wp_selected_gluon_weights
            wp_selected_gluon_weight_statistical_up_value = np.sqrt(wp_selected_gluon_weights_variances)
            wp_selected_gluon_weight_statistical_down_value = np.sqrt(wp_selected_gluon_weights_variances)

            wp_selected_gluon_weight_gluon_syst_up_value = gluon_weights_dict['gluon_syst_up_selected']
            wp_selected_gluon_weight_gluon_syst_down_value = gluon_weights_dict['gluon_syst_up_selected']
            wp_selected_gluon_weight_fsr_syst_up_value = gluon_weights_dict['fsr_syst_up_selected']
            wp_selected_gluon_weight_fsr_syst_down_value = gluon_weights_dict['fsr_syst_up_selected']
            wp_selected_gluon_weight_isr_syst_up_value = gluon_weights_dict['isr_syst_up_selected']
            wp_selected_gluon_weight_isr_syst_down_value = gluon_weights_dict['isr_syst_up_selected']
            wp_selected_gluon_weight_pu_syst_up_value = gluon_weights_dict['pu_syst_up_selected']
            wp_selected_gluon_weight_pu_syst_down_value = gluon_weights_dict['pu_syst_up_selected']
            wp_selected_gluon_weight_jes_syst_up_value = gluon_weights_dict['jes_syst_up_selected']
            wp_selected_gluon_weight_jes_syst_down_value = gluon_weights_dict['jes_syst_up_selected']
            wp_selected_gluon_weight_jer_syst_up_value = gluon_weights_dict['jer_syst_up_selected']
            wp_selected_gluon_weight_jer_syst_down_value = gluon_weights_dict['jer_syst_up_selected']

            wp_selected_gluon_weight_gluon_syst_up_difference = wp_selected_gluon_weight_gluon_syst_up_value - wp_selected_gluon_weight_nominal_value
            wp_selected_gluon_weight_gluon_syst_down_difference = wp_selected_gluon_weight_gluon_syst_down_value - wp_selected_gluon_weight_nominal_value 
            wp_selected_gluon_weight_fsr_syst_up_difference = wp_selected_gluon_weight_fsr_syst_up_value - wp_selected_gluon_weight_nominal_value
            wp_selected_gluon_weight_fsr_syst_down_difference = wp_selected_gluon_weight_fsr_syst_down_value - wp_selected_gluon_weight_nominal_value 
            wp_selected_gluon_weight_isr_syst_up_difference = wp_selected_gluon_weight_isr_syst_up_value - wp_selected_gluon_weight_nominal_value
            wp_selected_gluon_weight_isr_syst_down_difference = wp_selected_gluon_weight_isr_syst_down_value - wp_selected_gluon_weight_nominal_value 
            wp_selected_gluon_weight_pu_syst_up_difference = wp_selected_gluon_weight_pu_syst_up_value - wp_selected_gluon_weight_nominal_value
            wp_selected_gluon_weight_pu_syst_down_difference = wp_selected_gluon_weight_pu_syst_down_value - wp_selected_gluon_weight_nominal_value 
            wp_selected_gluon_weight_jes_syst_up_difference = wp_selected_gluon_weight_jes_syst_up_value - wp_selected_gluon_weight_nominal_value
            wp_selected_gluon_weight_jes_syst_down_difference = wp_selected_gluon_weight_jes_syst_down_value - wp_selected_gluon_weight_nominal_value 
            wp_selected_gluon_weight_jer_syst_up_difference = wp_selected_gluon_weight_jer_syst_up_value - wp_selected_gluon_weight_nominal_value
            wp_selected_gluon_weight_jer_syst_down_difference = wp_selected_gluon_weight_jer_syst_down_value - wp_selected_gluon_weight_nominal_value 

            wp_selected_gluon_weight_gluon_syst_up_variation = np.max([wp_selected_gluon_weight_gluon_syst_up_difference, wp_selected_gluon_weight_gluon_syst_down_difference, 0])
            wp_selected_gluon_weight_gluon_syst_down_variation = np.min([wp_selected_gluon_weight_gluon_syst_up_difference, wp_selected_gluon_weight_gluon_syst_down_difference, 0])
            wp_selected_gluon_weight_fsr_syst_up_variation = np.max([wp_selected_gluon_weight_fsr_syst_up_difference, wp_selected_gluon_weight_fsr_syst_down_difference, 0])
            wp_selected_gluon_weight_fsr_syst_down_variation = np.min([wp_selected_gluon_weight_fsr_syst_up_difference, wp_selected_gluon_weight_fsr_syst_down_difference, 0])
            wp_selected_gluon_weight_isr_syst_up_variation = np.max([wp_selected_gluon_weight_isr_syst_up_difference, wp_selected_gluon_weight_isr_syst_down_difference, 0])
            wp_selected_gluon_weight_isr_syst_down_variation = np.min([wp_selected_gluon_weight_isr_syst_up_difference, wp_selected_gluon_weight_isr_syst_down_difference, 0])
            wp_selected_gluon_weight_pu_syst_up_variation = np.max([wp_selected_gluon_weight_pu_syst_up_difference, wp_selected_gluon_weight_pu_syst_down_difference, 0])
            wp_selected_gluon_weight_pu_syst_down_variation = np.min([wp_selected_gluon_weight_pu_syst_up_difference, wp_selected_gluon_weight_pu_syst_down_difference, 0])
            wp_selected_gluon_weight_jes_syst_up_variation = np.max([wp_selected_gluon_weight_jes_syst_up_difference, wp_selected_gluon_weight_jes_syst_down_difference, 0])
            wp_selected_gluon_weight_jes_syst_down_variation = np.min([wp_selected_gluon_weight_jes_syst_up_difference, wp_selected_gluon_weight_jes_syst_down_difference, 0])
            wp_selected_gluon_weight_jer_syst_up_variation = np.max([wp_selected_gluon_weight_jer_syst_up_difference, wp_selected_gluon_weight_jer_syst_down_difference, 0])
            wp_selected_gluon_weight_jer_syst_down_variation = np.min([wp_selected_gluon_weight_jer_syst_up_difference, wp_selected_gluon_weight_jer_syst_down_difference, 0])

            ### ANTISELECTED QUARK WEIGHTS
            wp_antiselected_quark_weight_nominal_value = wp_antiselected_quark_weights
            wp_antiselected_quark_weight_statistical_up_value = np.sqrt(wp_antiselected_quark_weights_variances)
            wp_antiselected_quark_weight_statistical_down_value = np.sqrt(wp_antiselected_quark_weights_variances)

            wp_antiselected_quark_weight_gluon_syst_up_value = quark_weights_dict['gluon_syst_up_antiselected']
            wp_antiselected_quark_weight_gluon_syst_down_value = quark_weights_dict['gluon_syst_down_antiselected']
            wp_antiselected_quark_weight_fsr_syst_up_value = quark_weights_dict['fsr_syst_up_antiselected']
            wp_antiselected_quark_weight_fsr_syst_down_value = quark_weights_dict['fsr_syst_down_antiselected']
            wp_antiselected_quark_weight_isr_syst_up_value = quark_weights_dict['isr_syst_up_antiselected']
            wp_antiselected_quark_weight_isr_syst_down_value = quark_weights_dict['isr_syst_down_antiselected']
            wp_antiselected_quark_weight_pu_syst_up_value = quark_weights_dict['pu_syst_up_antiselected']
            wp_antiselected_quark_weight_pu_syst_down_value = quark_weights_dict['pu_syst_down_antiselected']
            wp_antiselected_quark_weight_jes_syst_up_value = quark_weights_dict['jes_syst_up_antiselected']
            wp_antiselected_quark_weight_jes_syst_down_value = quark_weights_dict['jes_syst_down_antiselected']
            wp_antiselected_quark_weight_jer_syst_up_value = quark_weights_dict['jer_syst_up_antiselected']
            wp_antiselected_quark_weight_jer_syst_down_value = quark_weights_dict['jer_syst_down_antiselected']

            wp_antiselected_quark_weight_gluon_syst_up_difference = wp_antiselected_quark_weight_gluon_syst_up_value - wp_antiselected_quark_weight_nominal_value
            wp_antiselected_quark_weight_gluon_syst_down_difference = wp_antiselected_quark_weight_gluon_syst_down_value - wp_antiselected_quark_weight_nominal_value 
            wp_antiselected_quark_weight_fsr_syst_up_difference = wp_antiselected_quark_weight_fsr_syst_up_value - wp_antiselected_quark_weight_nominal_value
            wp_antiselected_quark_weight_fsr_syst_down_difference = wp_antiselected_quark_weight_fsr_syst_down_value - wp_antiselected_quark_weight_nominal_value 
            wp_antiselected_quark_weight_isr_syst_up_difference = wp_antiselected_quark_weight_isr_syst_up_value - wp_antiselected_quark_weight_nominal_value
            wp_antiselected_quark_weight_isr_syst_down_difference = wp_antiselected_quark_weight_isr_syst_down_value - wp_antiselected_quark_weight_nominal_value 
            wp_antiselected_quark_weight_pu_syst_up_difference = wp_antiselected_quark_weight_pu_syst_up_value - wp_antiselected_quark_weight_nominal_value
            wp_antiselected_quark_weight_pu_syst_down_difference = wp_antiselected_quark_weight_pu_syst_down_value - wp_antiselected_quark_weight_nominal_value 
            wp_antiselected_quark_weight_jes_syst_up_difference = wp_antiselected_quark_weight_jes_syst_up_value - wp_antiselected_quark_weight_nominal_value
            wp_antiselected_quark_weight_jes_syst_down_difference = wp_antiselected_quark_weight_jes_syst_down_value - wp_antiselected_quark_weight_nominal_value 
            wp_antiselected_quark_weight_jer_syst_up_difference = wp_antiselected_quark_weight_jer_syst_up_value - wp_antiselected_quark_weight_nominal_value
            wp_antiselected_quark_weight_jer_syst_down_difference = wp_antiselected_quark_weight_jer_syst_down_value - wp_antiselected_quark_weight_nominal_value 

            wp_antiselected_quark_weight_gluon_syst_up_variation = np.max([wp_antiselected_quark_weight_gluon_syst_up_difference, wp_antiselected_quark_weight_gluon_syst_down_difference, 0])
            wp_antiselected_quark_weight_gluon_syst_down_variation = np.min([wp_antiselected_quark_weight_gluon_syst_up_difference, wp_antiselected_quark_weight_gluon_syst_down_difference, 0])
            wp_antiselected_quark_weight_fsr_syst_up_variation = np.max([wp_antiselected_quark_weight_fsr_syst_up_difference, wp_antiselected_quark_weight_fsr_syst_down_difference, 0])
            wp_antiselected_quark_weight_fsr_syst_down_variation = np.min([wp_antiselected_quark_weight_fsr_syst_up_difference, wp_antiselected_quark_weight_fsr_syst_down_difference, 0])
            wp_antiselected_quark_weight_isr_syst_up_variation = np.max([wp_antiselected_quark_weight_isr_syst_up_difference, wp_antiselected_quark_weight_isr_syst_down_difference, 0])
            wp_antiselected_quark_weight_isr_syst_down_variation = np.min([wp_antiselected_quark_weight_isr_syst_up_difference, wp_antiselected_quark_weight_isr_syst_down_difference, 0])
            wp_antiselected_quark_weight_pu_syst_up_variation = np.max([wp_antiselected_quark_weight_pu_syst_up_difference, wp_antiselected_quark_weight_pu_syst_down_difference, 0])
            wp_antiselected_quark_weight_pu_syst_down_variation = np.min([wp_antiselected_quark_weight_pu_syst_up_difference, wp_antiselected_quark_weight_pu_syst_down_difference, 0])
            wp_antiselected_quark_weight_jes_syst_up_variation = np.max([wp_antiselected_quark_weight_jes_syst_up_difference, wp_antiselected_quark_weight_jes_syst_down_difference, 0])
            wp_antiselected_quark_weight_jes_syst_down_variation = np.min([wp_antiselected_quark_weight_jes_syst_up_difference, wp_antiselected_quark_weight_jes_syst_down_difference, 0])
            wp_antiselected_quark_weight_jer_syst_up_variation = np.max([wp_antiselected_quark_weight_jer_syst_up_difference, wp_antiselected_quark_weight_jer_syst_down_difference, 0])
            wp_antiselected_quark_weight_jer_syst_down_variation = np.min([wp_antiselected_quark_weight_jer_syst_up_difference, wp_antiselected_quark_weight_jer_syst_down_difference, 0])

            ### ANTISELECTED GLUON WEIGHTS
            wp_antiselected_gluon_weight_nominal_value = wp_antiselected_gluon_weights
            wp_antiselected_gluon_weight_statistical_up_value = np.sqrt(wp_antiselected_gluon_weights_variances)
            wp_antiselected_gluon_weight_statistical_down_value = np.sqrt(wp_antiselected_gluon_weights_variances)

            wp_antiselected_gluon_weight_gluon_syst_up_value = gluon_weights_dict['gluon_syst_up_antiselected']
            wp_antiselected_gluon_weight_gluon_syst_down_value = gluon_weights_dict['gluon_syst_down_antiselected']
            wp_antiselected_gluon_weight_fsr_syst_up_value = gluon_weights_dict['fsr_syst_up_antiselected']
            wp_antiselected_gluon_weight_fsr_syst_down_value = gluon_weights_dict['fsr_syst_down_antiselected']
            wp_antiselected_gluon_weight_isr_syst_up_value = gluon_weights_dict['isr_syst_up_antiselected']
            wp_antiselected_gluon_weight_isr_syst_down_value = gluon_weights_dict['isr_syst_down_antiselected']
            wp_antiselected_gluon_weight_pu_syst_up_value = gluon_weights_dict['pu_syst_up_antiselected']
            wp_antiselected_gluon_weight_pu_syst_down_value = gluon_weights_dict['pu_syst_down_antiselected']
            wp_antiselected_gluon_weight_jes_syst_up_value = gluon_weights_dict['jes_syst_up_antiselected']
            wp_antiselected_gluon_weight_jes_syst_down_value = gluon_weights_dict['jes_syst_down_antiselected']
            wp_antiselected_gluon_weight_jer_syst_up_value = gluon_weights_dict['jer_syst_up_antiselected']
            wp_antiselected_gluon_weight_jer_syst_down_value = gluon_weights_dict['jer_syst_down_antiselected']

            wp_antiselected_gluon_weight_gluon_syst_up_difference = wp_antiselected_gluon_weight_gluon_syst_up_value - wp_antiselected_gluon_weight_nominal_value
            wp_antiselected_gluon_weight_gluon_syst_down_difference = wp_antiselected_gluon_weight_gluon_syst_down_value - wp_antiselected_gluon_weight_nominal_value 
            wp_antiselected_gluon_weight_fsr_syst_up_difference = wp_antiselected_gluon_weight_fsr_syst_up_value - wp_antiselected_gluon_weight_nominal_value
            wp_antiselected_gluon_weight_fsr_syst_down_difference = wp_antiselected_gluon_weight_fsr_syst_down_value - wp_antiselected_gluon_weight_nominal_value 
            wp_antiselected_gluon_weight_isr_syst_up_difference = wp_antiselected_gluon_weight_isr_syst_up_value - wp_antiselected_gluon_weight_nominal_value
            wp_antiselected_gluon_weight_isr_syst_down_difference = wp_antiselected_gluon_weight_isr_syst_down_value - wp_antiselected_gluon_weight_nominal_value 
            wp_antiselected_gluon_weight_pu_syst_up_difference = wp_antiselected_gluon_weight_pu_syst_up_value - wp_antiselected_gluon_weight_nominal_value
            wp_antiselected_gluon_weight_pu_syst_down_difference = wp_antiselected_gluon_weight_pu_syst_down_value - wp_antiselected_gluon_weight_nominal_value 
            wp_antiselected_gluon_weight_jes_syst_up_difference = wp_antiselected_gluon_weight_jes_syst_up_value - wp_antiselected_gluon_weight_nominal_value
            wp_antiselected_gluon_weight_jes_syst_down_difference = wp_antiselected_gluon_weight_jes_syst_down_value - wp_antiselected_gluon_weight_nominal_value 
            wp_antiselected_gluon_weight_jer_syst_up_difference = wp_antiselected_gluon_weight_jer_syst_up_value - wp_antiselected_gluon_weight_nominal_value
            wp_antiselected_gluon_weight_jer_syst_down_difference = wp_antiselected_gluon_weight_jer_syst_down_value - wp_antiselected_gluon_weight_nominal_value 

            wp_antiselected_gluon_weight_gluon_syst_up_variation = np.max([wp_antiselected_gluon_weight_gluon_syst_up_difference, wp_antiselected_gluon_weight_gluon_syst_down_difference, 0])
            wp_antiselected_gluon_weight_gluon_syst_down_variation = np.min([wp_antiselected_gluon_weight_gluon_syst_up_difference, wp_antiselected_gluon_weight_gluon_syst_down_difference, 0])
            wp_antiselected_gluon_weight_fsr_syst_up_variation = np.max([wp_antiselected_gluon_weight_fsr_syst_up_difference, wp_antiselected_gluon_weight_fsr_syst_down_difference, 0])
            wp_antiselected_gluon_weight_fsr_syst_down_variation = np.min([wp_antiselected_gluon_weight_fsr_syst_up_difference, wp_antiselected_gluon_weight_fsr_syst_down_difference, 0])
            wp_antiselected_gluon_weight_isr_syst_up_variation = np.max([wp_antiselected_gluon_weight_isr_syst_up_difference, wp_antiselected_gluon_weight_isr_syst_down_difference, 0])
            wp_antiselected_gluon_weight_isr_syst_down_variation = np.min([wp_antiselected_gluon_weight_isr_syst_up_difference, wp_antiselected_gluon_weight_isr_syst_down_difference, 0])
            wp_antiselected_gluon_weight_pu_syst_up_variation = np.max([wp_antiselected_gluon_weight_pu_syst_up_difference, wp_antiselected_gluon_weight_pu_syst_down_difference, 0])
            wp_antiselected_gluon_weight_pu_syst_down_variation = np.min([wp_antiselected_gluon_weight_pu_syst_up_difference, wp_antiselected_gluon_weight_pu_syst_down_difference, 0])
            wp_antiselected_gluon_weight_jes_syst_up_variation = np.max([wp_antiselected_gluon_weight_jes_syst_up_difference, wp_antiselected_gluon_weight_jes_syst_down_difference, 0])
            wp_antiselected_gluon_weight_jes_syst_down_variation = np.min([wp_antiselected_gluon_weight_jes_syst_up_difference, wp_antiselected_gluon_weight_jes_syst_down_difference, 0])
            wp_antiselected_gluon_weight_jer_syst_up_variation = np.max([wp_antiselected_gluon_weight_jer_syst_up_difference, wp_antiselected_gluon_weight_jer_syst_down_difference, 0])
            wp_antiselected_gluon_weight_jer_syst_down_variation = np.min([wp_antiselected_gluon_weight_jer_syst_up_difference, wp_antiselected_gluon_weight_jer_syst_down_difference, 0])

            wp_selected_quark_weight_combined_up_variation = np.sqrt(
                    wp_selected_quark_weight_statistical_up_value**2 +
                    wp_selected_quark_weight_gluon_syst_up_variation**2 +
                    wp_selected_quark_weight_fsr_syst_up_variation**2 +
                    wp_selected_quark_weight_isr_syst_up_variation**2 +
                    wp_selected_quark_weight_pu_syst_up_variation**2 +
                    wp_selected_quark_weight_jes_syst_up_variation**2 +
                    wp_selected_quark_weight_jer_syst_up_variation**2)
            wp_selected_quark_weight_combined_down_variation = np.sqrt(
                    wp_selected_quark_weight_statistical_down_value**2 +
                    wp_selected_quark_weight_gluon_syst_down_variation**2 +
                    wp_selected_quark_weight_fsr_syst_down_variation**2 +
                    wp_selected_quark_weight_isr_syst_down_variation**2 +
                    wp_selected_quark_weight_pu_syst_down_variation**2 +
                    wp_selected_quark_weight_jes_syst_down_variation**2 +
                    wp_selected_quark_weight_jer_syst_down_variation**2)
            wp_selected_gluon_weight_combined_up_variation = np.sqrt(
                    wp_selected_gluon_weight_statistical_up_value**2 +
                    wp_selected_gluon_weight_gluon_syst_up_variation**2 +
                    wp_selected_gluon_weight_fsr_syst_up_variation**2 +
                    wp_selected_gluon_weight_isr_syst_up_variation**2 +
                    wp_selected_gluon_weight_pu_syst_up_variation**2 +
                    wp_selected_gluon_weight_jes_syst_up_variation**2 +
                    wp_selected_gluon_weight_jer_syst_up_variation**2)
            wp_selected_gluon_weight_combined_down_variation = np.sqrt(
                    wp_selected_gluon_weight_statistical_down_value**2 +
                    wp_selected_gluon_weight_gluon_syst_down_variation**2 +
                    wp_selected_gluon_weight_fsr_syst_down_variation**2 +
                    wp_selected_gluon_weight_isr_syst_down_variation**2 +
                    wp_selected_gluon_weight_pu_syst_down_variation**2 +
                    wp_selected_gluon_weight_jes_syst_down_variation**2 +
                    wp_selected_gluon_weight_jer_syst_down_variation**2)
            wp_antiselected_quark_weight_combined_up_variation = np.sqrt(
                    wp_antiselected_quark_weight_statistical_up_value**2 +
                    wp_antiselected_quark_weight_gluon_syst_up_variation**2 +
                    wp_antiselected_quark_weight_fsr_syst_up_variation**2 +
                    wp_antiselected_quark_weight_isr_syst_up_variation**2 +
                    wp_antiselected_quark_weight_pu_syst_up_variation**2 +
                    wp_antiselected_quark_weight_jes_syst_up_variation**2 +
                    wp_antiselected_quark_weight_jer_syst_up_variation**2)
            wp_antiselected_quark_weight_combined_down_variation = np.sqrt(
                    wp_antiselected_quark_weight_statistical_down_value**2 +
                    wp_antiselected_quark_weight_gluon_syst_down_variation**2 +
                    wp_antiselected_quark_weight_fsr_syst_down_variation**2 +
                    wp_antiselected_quark_weight_isr_syst_down_variation**2 +
                    wp_antiselected_quark_weight_pu_syst_down_variation**2 +
                    wp_antiselected_quark_weight_jes_syst_down_variation**2 +
                    wp_antiselected_quark_weight_jer_syst_down_variation**2)
            wp_antiselected_gluon_weight_combined_up_variation = np.sqrt(
                    wp_antiselected_gluon_weight_statistical_up_value**2 +
                    wp_antiselected_gluon_weight_gluon_syst_up_variation**2 +
                    wp_antiselected_gluon_weight_fsr_syst_up_variation**2 +
                    wp_antiselected_gluon_weight_isr_syst_up_variation**2 +
                    wp_antiselected_gluon_weight_pu_syst_up_variation**2 +
                    wp_antiselected_gluon_weight_jes_syst_up_variation**2 +
                    wp_antiselected_gluon_weight_jer_syst_up_variation**2)
            wp_antiselected_gluon_weight_combined_down_variation = np.sqrt(
                    wp_antiselected_gluon_weight_statistical_down_value**2 +
                    wp_antiselected_gluon_weight_gluon_syst_down_variation**2 +
                    wp_antiselected_gluon_weight_fsr_syst_down_variation**2 +
                    wp_antiselected_gluon_weight_isr_syst_down_variation**2 +
                    wp_antiselected_gluon_weight_pu_syst_down_variation**2 +
                    wp_antiselected_gluon_weight_jes_syst_down_variation**2 +
                    wp_antiselected_gluon_weight_jer_syst_down_variation**2)

            wp_selected_quark_weights_total_variances[0].append(wp_selected_quark_weight_combined_down_variation**2)
            wp_selected_quark_weights_total_variances[1].append(wp_selected_quark_weight_combined_up_variation**2)
            wp_selected_gluon_weights_total_variances[0].append(wp_selected_gluon_weight_combined_down_variation**2)
            wp_selected_gluon_weights_total_variances[1].append(wp_selected_gluon_weight_combined_up_variation**2)
            
            wp_antiselected_quark_weights_total_variances[0].append(wp_antiselected_quark_weight_combined_down_variation**2)
            wp_antiselected_quark_weights_total_variances[1].append(wp_antiselected_quark_weight_combined_up_variation**2)
            wp_antiselected_gluon_weights_total_variances[0].append(wp_antiselected_gluon_weight_combined_down_variation**2)
            wp_antiselected_gluon_weights_total_variances[1].append(wp_antiselected_gluon_weight_combined_up_variation**2)

            hist_quark_syst_up = hist.Hist.new.Regular(2, 0.0, 1.0).Weight()
            hist_gluon_syst_up = hist.Hist.new.Regular(2, 0.0, 1.0).Weight()
            hist_quark_syst_down = hist.Hist.new.Regular(2, 0.0, 1.0).Weight()
            hist_gluon_syst_down = hist.Hist.new.Regular(2, 0.0, 1.0).Weight()

            hist_gluon_syst_down.values()[0] = gluon_weights_dict['nominal'][0].values()[0] - wp_antiselected_gluon_weight_combined_down_variation
            hist_gluon_syst_down.values()[1] = gluon_weights_dict['nominal'][0].values()[1] - wp_selected_gluon_weight_combined_down_variation
            hist_gluon_syst_up.values()[0] = gluon_weights_dict['nominal'][0].values()[0] + wp_antiselected_gluon_weight_combined_up_variation
            hist_gluon_syst_up.values()[1] = gluon_weights_dict['nominal'][0].values()[1] + wp_selected_gluon_weight_combined_up_variation
            hist_quark_syst_down.values()[0] = quark_weights_dict['nominal'][0].values()[0] - wp_antiselected_quark_weight_combined_down_variation
            hist_quark_syst_down.values()[1] = quark_weights_dict['nominal'][0].values()[1] - wp_selected_quark_weight_combined_down_variation
            hist_quark_syst_up.values()[0] = quark_weights_dict['nominal'][0].values()[0] + wp_antiselected_quark_weight_combined_up_variation
            hist_quark_syst_up.values()[1] = quark_weights_dict['nominal'][0].values()[1] + wp_selected_quark_weight_combined_up_variation

            weight_save_name = weight_save_path.joinpath(f'{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP_total_syst')
            with uproot.recreate(weight_save_name.with_suffix('.root')) as root_file:
                root_file[f'{variable}_quark_weights'] = quark_weights_dict['nominal'][0]
                root_file[f'{variable}_gluon_weights'] = gluon_weights_dict['nominal'][0]
                root_file[f'{variable}_quark_weights_syst_up'] = hist_quark_syst_up
                root_file[f'{variable}_gluon_weights_syst_up'] = hist_gluon_syst_up
                root_file[f'{variable}_quark_weights_syst_down'] = hist_quark_syst_down
                root_file[f'{variable}_gluon_weights_syst_down'] = hist_gluon_syst_down


if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--var', dest='variable', default='qgl', help='Variable to plot (qgl, particlenet, deepjet)')
    parser.add_option('-w', '--wp', dest='working_point', default='medium', help='Choose working point: loose, medium, tight.')
    (opt, args) = parser.parse_args()

    if opt.variable.lower() == 'particlenet':
        opt.variable = 'particleNetAK4_QvsG'
    if opt.variable.lower() == 'deepjet':
        opt.variable = 'btagDeepFlavQG'

    if opt.working_point.lower() not in ['loose', 'medium', 'tight']:
        sys.exit('ERROR! Invalid working point: {}\nChoose loose, medium, or tight.'.format(opt.working_point.lower()))

    main(opt.config_path, opt.variable, opt.working_point.lower())
