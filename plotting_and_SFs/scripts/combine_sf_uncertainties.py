import numpy as np
import tomllib
import uproot
import hist
import json
import sys

from pathlib import Path
from optparse import OptionParser

def main(config_path, variable):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n------------------------------------')
    print(f'COMBINING SCALE FACTOR UNCERTAINTIES')
    print(f'------------------------------------')
    print(f'VARIABLE: {variable}')

    input_file_path = Path(config['path']['input'])
    print(f'READING SF WEIGHTS FROM:       {input_file_path}/sf_weights/')
    print(f'SAVING COMBINED SF WEIGHTS TO: {input_file_path}/sf_weights/')

    sf_bins = config['SF_binning']['bins']
    for sf_bin in sf_bins:
        eta_bin = sf_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bins = sf_bin['pT']
        for i in range(len(pt_bins)-1):
            pt_range = f'{pt_bins[i]}to{pt_bins[i+1]}'

            campaign = config['campaign']
            weights_gluon_syst_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_gluon_syst.root')
            weights_fsr_syst_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_fsr_syst.root')
            weights_isr_syst_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_isr_syst.root')
            weights_pu_syst_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_pu_syst.root')
            weights_jes_syst_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_jes_syst.root')
            weights_jer_syst_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_jer_syst.root')

            with uproot.open(weights_gluon_syst_path) as f:
                quark_weights = f[f'{variable}_quark_weights']
                gluon_weights = f[f'{variable}_gluon_weights']
                quark_weights_gluon_syst_up = f[f'{variable}_quark_weights_syst_up'].values()
                gluon_weights_gluon_syst_up = f[f'{variable}_gluon_weights_syst_up'].values()
                quark_weights_gluon_syst_down = f[f'{variable}_quark_weights_syst_down'].values()
                gluon_weights_gluon_syst_down = f[f'{variable}_gluon_weights_syst_down'].values()

            with uproot.open(weights_fsr_syst_path) as f:
                quark_weights_fsr_syst_up = f[f'{variable}_quark_weights_syst_up'].values()
                gluon_weights_fsr_syst_up = f[f'{variable}_gluon_weights_syst_up'].values()
                quark_weights_fsr_syst_down = f[f'{variable}_quark_weights_syst_down'].values()
                gluon_weights_fsr_syst_down = f[f'{variable}_gluon_weights_syst_down'].values()

            with uproot.open(weights_isr_syst_path) as f:
                quark_weights_isr_syst_up = f[f'{variable}_quark_weights_syst_up'].values()
                gluon_weights_isr_syst_up = f[f'{variable}_gluon_weights_syst_up'].values()
                quark_weights_isr_syst_down = f[f'{variable}_quark_weights_syst_down'].values()
                gluon_weights_isr_syst_down = f[f'{variable}_gluon_weights_syst_down'].values()

            with uproot.open(weights_pu_syst_path) as f:
                quark_weights_pu_syst_up = f[f'{variable}_quark_weights_syst_up'].values()
                gluon_weights_pu_syst_up = f[f'{variable}_gluon_weights_syst_up'].values()
                quark_weights_pu_syst_down = f[f'{variable}_quark_weights_syst_down'].values()
                gluon_weights_pu_syst_down = f[f'{variable}_gluon_weights_syst_down'].values()

            with uproot.open(weights_jes_syst_path) as f:
                quark_weights_jes_syst_up = f[f'{variable}_quark_weights_syst_up'].values()
                gluon_weights_jes_syst_up = f[f'{variable}_gluon_weights_syst_up'].values()
                quark_weights_jes_syst_down = f[f'{variable}_quark_weights_syst_down'].values()
                gluon_weights_jes_syst_down = f[f'{variable}_gluon_weights_syst_down'].values()

            with uproot.open(weights_jer_syst_path) as f:
                quark_weights_jer_syst_up = f[f'{variable}_quark_weights_syst_up'].values()
                gluon_weights_jer_syst_up = f[f'{variable}_gluon_weights_syst_up'].values()
                quark_weights_jer_syst_down = f[f'{variable}_quark_weights_syst_down'].values()
                gluon_weights_jer_syst_down = f[f'{variable}_gluon_weights_syst_down'].values()

            quark_weights_nominal = quark_weights.values()
            quark_weights_statistical_up_variations = np.sqrt(quark_weights.variances())
            quark_weights_statistical_down_variations = np.sqrt(quark_weights.variances())
            quark_weights_total_up_variations = []
            quark_weights_total_down_variations = []

            gluon_weights_nominal = gluon_weights.values()
            gluon_weights_statistical_up_variations = np.sqrt(gluon_weights.variances())
            gluon_weights_statistical_down_variations = np.sqrt(gluon_weights.variances())
            gluon_weights_total_up_variations = []
            gluon_weights_total_down_variations = []

            nBins = len(quark_weights.values())
            for i in range(nBins):
                quark_weight_nominal_value = quark_weights_nominal[i]
                quark_weight_statistical_up_value = quark_weights_statistical_up_variations[i]
                quark_weight_statistical_down_value = quark_weights_statistical_down_variations[i]
                quark_weight_gluon_syst_up_value = quark_weights_gluon_syst_up[i]
                quark_weight_gluon_syst_down_value = quark_weights_gluon_syst_down[i]
                quark_weight_fsr_syst_up_value = quark_weights_fsr_syst_up[i]
                quark_weight_fsr_syst_down_value = quark_weights_fsr_syst_down[i]
                quark_weight_isr_syst_up_value = quark_weights_isr_syst_up[i]
                quark_weight_isr_syst_down_value = quark_weights_isr_syst_down[i]
                quark_weight_pu_syst_up_value = quark_weights_pu_syst_up[i]
                quark_weight_pu_syst_down_value = quark_weights_pu_syst_down[i]
                quark_weight_jes_syst_up_value = quark_weights_jes_syst_up[i]
                quark_weight_jes_syst_down_value = quark_weights_jes_syst_down[i]
                quark_weight_jer_syst_up_value = quark_weights_jer_syst_up[i]
                quark_weight_jer_syst_down_value = quark_weights_jer_syst_down[i]

                quark_weight_gluon_syst_up_difference = quark_weight_gluon_syst_up_value - quark_weight_nominal_value 
                quark_weight_gluon_syst_down_difference = quark_weight_gluon_syst_down_value - quark_weight_nominal_value 
                quark_weight_fsr_syst_up_difference = quark_weight_fsr_syst_up_value - quark_weight_nominal_value 
                quark_weight_fsr_syst_down_difference = quark_weight_fsr_syst_down_value - quark_weight_nominal_value 
                quark_weight_isr_syst_up_difference = quark_weight_isr_syst_up_value - quark_weight_nominal_value 
                quark_weight_isr_syst_down_difference = quark_weight_isr_syst_down_value - quark_weight_nominal_value 
                quark_weight_pu_syst_up_difference = quark_weight_pu_syst_up_value - quark_weight_nominal_value 
                quark_weight_pu_syst_down_difference = quark_weight_pu_syst_down_value - quark_weight_nominal_value 
                quark_weight_jes_syst_up_difference = quark_weight_jes_syst_up_value - quark_weight_nominal_value 
                quark_weight_jes_syst_down_difference = quark_weight_jes_syst_down_value - quark_weight_nominal_value 
                quark_weight_jer_syst_up_difference = quark_weight_jer_syst_up_value - quark_weight_nominal_value 
                quark_weight_jer_syst_down_difference = quark_weight_jer_syst_down_value - quark_weight_nominal_value 

                quark_weight_gluon_syst_up_variation = np.max([quark_weight_gluon_syst_up_difference, quark_weight_gluon_syst_down_difference, 0])
                quark_weight_gluon_syst_down_variation = np.min([quark_weight_gluon_syst_up_difference, quark_weight_gluon_syst_down_difference, 0])
                quark_weight_fsr_syst_up_variation = np.max([quark_weight_fsr_syst_up_difference, quark_weight_fsr_syst_down_difference, 0])
                quark_weight_fsr_syst_down_variation = np.min([quark_weight_fsr_syst_up_difference, quark_weight_fsr_syst_down_difference, 0])
                quark_weight_isr_syst_up_variation = np.max([quark_weight_isr_syst_up_difference, quark_weight_isr_syst_down_difference, 0])
                quark_weight_isr_syst_down_variation = np.min([quark_weight_isr_syst_up_difference, quark_weight_isr_syst_down_difference, 0])
                quark_weight_pu_syst_up_variation = np.max([quark_weight_pu_syst_up_difference, quark_weight_pu_syst_down_difference, 0])
                quark_weight_pu_syst_down_variation = np.min([quark_weight_pu_syst_up_difference, quark_weight_pu_syst_down_difference, 0])
                quark_weight_jes_syst_up_variation = np.max([quark_weight_jes_syst_up_difference, quark_weight_jes_syst_down_difference, 0])
                quark_weight_jes_syst_down_variation = np.min([quark_weight_jes_syst_up_difference, quark_weight_jes_syst_down_difference, 0])
                quark_weight_jer_syst_up_variation = np.max([quark_weight_jer_syst_up_difference, quark_weight_jer_syst_down_difference, 0])
                quark_weight_jer_syst_down_variation = np.min([quark_weight_jer_syst_up_difference, quark_weight_jer_syst_down_difference, 0])

                quark_weight_combined_up_variation = np.sqrt(quark_weight_statistical_up_value**2
                                                             + quark_weight_gluon_syst_up_variation**2
                                                             + quark_weight_fsr_syst_up_variation**2
                                                             + quark_weight_isr_syst_up_variation**2
                                                             + quark_weight_pu_syst_up_variation**2
                                                             + quark_weight_jes_syst_up_variation**2
                                                             + quark_weight_jer_syst_up_variation**2)
                
                quark_weight_combined_down_variation = np.sqrt(quark_weight_statistical_down_value**2
                                                               + quark_weight_gluon_syst_down_variation**2
                                                               + quark_weight_fsr_syst_down_variation**2
                                                               + quark_weight_isr_syst_down_variation**2
                                                               + quark_weight_pu_syst_down_variation**2 
                                                               + quark_weight_jes_syst_down_variation**2
                                                               + quark_weight_jer_syst_down_variation**2)

                quark_weights_total_up_variations.append(quark_weight_combined_up_variation)
                quark_weights_total_down_variations.append(quark_weight_combined_down_variation)


                gluon_weight_nominal_value = gluon_weights_nominal[i]
                gluon_weight_statistical_up_value = gluon_weights_statistical_up_variations[i]
                gluon_weight_statistical_down_value = gluon_weights_statistical_down_variations[i]
                gluon_weight_gluon_syst_up_value = gluon_weights_gluon_syst_up[i]
                gluon_weight_gluon_syst_down_value = gluon_weights_gluon_syst_down[i]
                gluon_weight_fsr_syst_up_value = gluon_weights_fsr_syst_up[i]
                gluon_weight_fsr_syst_down_value = gluon_weights_fsr_syst_down[i]
                gluon_weight_isr_syst_up_value = gluon_weights_isr_syst_up[i]
                gluon_weight_isr_syst_down_value = gluon_weights_isr_syst_down[i]
                gluon_weight_pu_syst_up_value = gluon_weights_pu_syst_up[i]
                gluon_weight_pu_syst_down_value = gluon_weights_pu_syst_down[i]
                gluon_weight_jes_syst_up_value = gluon_weights_jes_syst_up[i]
                gluon_weight_jes_syst_down_value = gluon_weights_jes_syst_down[i]
                gluon_weight_jer_syst_up_value = gluon_weights_jer_syst_up[i]
                gluon_weight_jer_syst_down_value = gluon_weights_jer_syst_down[i]

                gluon_weight_gluon_syst_up_difference = gluon_weight_gluon_syst_up_value - gluon_weight_nominal_value 
                gluon_weight_gluon_syst_down_difference = gluon_weight_gluon_syst_down_value - gluon_weight_nominal_value 
                gluon_weight_fsr_syst_up_difference = gluon_weight_fsr_syst_up_value - gluon_weight_nominal_value 
                gluon_weight_fsr_syst_down_difference = gluon_weight_fsr_syst_down_value - gluon_weight_nominal_value 
                gluon_weight_isr_syst_up_difference = gluon_weight_isr_syst_up_value - gluon_weight_nominal_value 
                gluon_weight_isr_syst_down_difference = gluon_weight_isr_syst_down_value - gluon_weight_nominal_value 
                gluon_weight_pu_syst_up_difference = gluon_weight_pu_syst_up_value - gluon_weight_nominal_value 
                gluon_weight_pu_syst_down_difference = gluon_weight_pu_syst_down_value - gluon_weight_nominal_value 
                gluon_weight_jes_syst_up_difference = gluon_weight_jes_syst_up_value - gluon_weight_nominal_value 
                gluon_weight_jes_syst_down_difference = gluon_weight_jes_syst_down_value - gluon_weight_nominal_value 
                gluon_weight_jer_syst_up_difference = gluon_weight_jer_syst_up_value - gluon_weight_nominal_value 
                gluon_weight_jer_syst_down_difference = gluon_weight_jer_syst_down_value - gluon_weight_nominal_value 

                gluon_weight_gluon_syst_up_variation = np.max([gluon_weight_gluon_syst_up_difference, gluon_weight_gluon_syst_down_difference, 0])
                gluon_weight_gluon_syst_down_variation = np.min([gluon_weight_gluon_syst_up_difference, gluon_weight_gluon_syst_down_difference, 0])
                gluon_weight_fsr_syst_up_variation = np.max([gluon_weight_fsr_syst_up_difference, gluon_weight_fsr_syst_down_difference, 0])
                gluon_weight_fsr_syst_down_variation = np.min([gluon_weight_fsr_syst_up_difference, gluon_weight_fsr_syst_down_difference, 0])
                gluon_weight_isr_syst_up_variation = np.max([gluon_weight_isr_syst_up_difference, gluon_weight_isr_syst_down_difference, 0])
                gluon_weight_isr_syst_down_variation = np.min([gluon_weight_isr_syst_up_difference, gluon_weight_isr_syst_down_difference, 0])
                gluon_weight_pu_syst_up_variation = np.max([gluon_weight_pu_syst_up_difference, gluon_weight_pu_syst_down_difference, 0])
                gluon_weight_pu_syst_down_variation = np.min([gluon_weight_pu_syst_up_difference, gluon_weight_pu_syst_down_difference, 0])
                gluon_weight_jes_syst_up_variation = np.max([gluon_weight_jes_syst_up_difference, gluon_weight_jes_syst_down_difference, 0])
                gluon_weight_jes_syst_down_variation = np.min([gluon_weight_jes_syst_up_difference, gluon_weight_jes_syst_down_difference, 0])
                gluon_weight_jer_syst_up_variation = np.max([gluon_weight_jer_syst_up_difference, gluon_weight_jer_syst_down_difference, 0])
                gluon_weight_jer_syst_down_variation = np.min([gluon_weight_jer_syst_up_difference, gluon_weight_jer_syst_down_difference, 0])

                gluon_weight_combined_up_variation = np.sqrt(gluon_weight_statistical_up_value**2
                                                             + gluon_weight_gluon_syst_up_variation**2
                                                             + gluon_weight_fsr_syst_up_variation**2
                                                             + gluon_weight_isr_syst_up_variation**2
                                                             + gluon_weight_pu_syst_up_variation**2
                                                             + gluon_weight_jes_syst_up_variation**2
                                                             + gluon_weight_jer_syst_up_variation**2)

                gluon_weight_combined_down_variation = np.sqrt(gluon_weight_statistical_down_value**2
                                                               + gluon_weight_gluon_syst_down_variation**2
                                                               + gluon_weight_fsr_syst_down_variation**2
                                                               + gluon_weight_isr_syst_down_variation**2
                                                               + gluon_weight_pu_syst_down_variation**2 
                                                               + gluon_weight_jes_syst_down_variation**2
                                                               + gluon_weight_jer_syst_down_variation**2)

                gluon_weights_total_up_variations.append(gluon_weight_combined_up_variation)
                gluon_weights_total_down_variations.append(gluon_weight_combined_down_variation)

            quark_weights_total_up_variations = np.squeeze(np.array([quark_weights_total_up_variations]))
            quark_weights_total_down_variations = np.squeeze(np.array([quark_weights_total_down_variations]))
            gluon_weights_total_up_variations = np.squeeze(np.array([gluon_weights_total_up_variations]))
            gluon_weights_total_down_variations = np.squeeze(np.array([gluon_weights_total_down_variations]))

            bin_edges = quark_weights.to_hist().axes.edges[0]
            save_path = Path(f'{input_file_path}/sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_combined_syst')
            with uproot.recreate(save_path.with_suffix('.root')) as root_file:
                root_file['quark_weights_nominal'] = quark_weights_nominal, bin_edges
                root_file['quark_weights_stat_var'] = np.sqrt(quark_weights.variances()), bin_edges
                root_file['quark_weights_combined_unc_up'] = quark_weights_total_up_variations, bin_edges
                root_file['quark_weights_combined_unc_down'] = quark_weights_total_down_variations, bin_edges
                root_file['gluon_weights_nominal'] = gluon_weights_nominal, bin_edges
                root_file['gluon_weights_stat_var'] = np.sqrt(gluon_weights.variances()), bin_edges
                root_file['gluon_weights_combined_unc_up'] = gluon_weights_total_up_variations, bin_edges
                root_file['gluon_weights_combined_unc_down'] = gluon_weights_total_down_variations, bin_edges

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--variable', dest='variable', type='string', default='qgl', help='Variable to plot. (Possible: qgl, qgl_new, particleNet, deepJet)')
    (opt, args) = parser.parse_args()

    if opt.variable.lower() == 'particlenet':
        opt.variable = 'particleNetAK4_QvsG'
    if opt.variable.lower() == 'deepjet':
        opt.variable = 'btagDeepFlavQG'

    main(opt.config_path, opt.variable)
