import numpy as np
import pandas as pd
import tomllib
import uproot
import hist
import sys

from pathlib import Path
from optparse import OptionParser

def weight_extraction(dijet, zmm):
    nBins_array = [len(h.values()) for h in [dijet['data'], dijet['quark'], dijet['gluon'], dijet['undef'], zmm['data'], zmm['quark'], zmm['gluon'], zmm['undef']]]
    if len(set(nBins_array)) == 1:
        nBins = nBins_array[0]
    else:
        sys.exit('ERROR! The input histograms have incompatible binning.')

    bin_edges = dijet['data'].axes.edges[0]
    gluon_weights = hist.Hist.new.Variable(bin_edges).Weight()
    quark_weights = hist.Hist.new.Variable(bin_edges).Weight()

    for i in range(nBins):
        a = dijet['gluon'].values()[i]
        b = dijet['quark'].values()[i]
        c = dijet['data'].values()[i] - dijet['undef'].values()[i]

        a_var = dijet['gluon'].variances()[i]
        b_var = dijet['quark'].variances()[i]
        c_var = dijet['data'].variances()[i] + dijet['undef'].variances()[i]

        d = zmm['gluon'].values()[i]
        e = zmm['quark'].values()[i]
        f = zmm['data'].values()[i] - zmm['undef'].values()[i]

        d_var = zmm['gluon'].variances()[i]
        e_var = zmm['quark'].variances()[i]
        f_var = zmm['data'].variances()[i] + zmm['undef'].variances()[i]

        det = a*e - b*d

        if det != 0:
            # Solve system of equations using determinants
            x = (c*e - b*f)/det
            y = (a*f - c*d)/det

            # Propagation of uncertainties
            det_squared = np.power(det, 2)

            dxda = np.power(e*(b*f-c*e)/det_squared, 2)
            dxdb = np.power(e*(c*d-a*f)/det_squared, 2)
            dxdc = np.power(e/det, 2)
            dxdd = np.power(b*(c*e-b*f)/det_squared, 2)
            dxde = np.power(b*(a*f-c*d)/det_squared, 2)
            dxdf = np.power(-b/det, 2)
            x_var = a_var*dxda + b_var*dxdb + c_var*dxdc + d_var*dxdd + e_var*dxde + f_var*dxdf

            dyda = np.power(d*(e*c-b*f)/det_squared, 2)
            dydb = np.power(d*(a*f-d*c)/det_squared, 2)
            dydd = np.power(a*(b*f-e*c)/det_squared, 2)
            dyde = np.power(a*(d*c-a*f)/det_squared, 2)
            dydc = np.power(d/det, 2)
            dydf = np.power(a/det, 2)
            y_var = a_var*dyda + b_var*dydb + c_var*dydc + d_var*dydd + e_var*dyde + f_var*dydf
        else:
            x = 0
            y = 0
            x_var = 0
            y_var = 0

        gluon_weights.values()[i] = x
        gluon_weights.variances()[i] = x_var

        quark_weights.values()[i] = y
        quark_weights.variances()[i] = y_var

    return quark_weights, gluon_weights

def read_histograms(dijet_path, zmm_path, base_histogram_name, eta_range, pt_range):
    with uproot.open(dijet_path.with_suffix('.root')) as dijet_file:
        base_histogram_name_dijet = base_histogram_name+f'_dijet_eta{eta_range}_pt{pt_range}'
        dijet_data = dijet_file[base_histogram_name_dijet+'_data'].to_hist()
        dijet_quark = dijet_file[base_histogram_name_dijet+'_quark'].to_hist()
        dijet_gluon = dijet_file[base_histogram_name_dijet+'_gluon'].to_hist()
        dijet_undef = dijet_file[base_histogram_name_dijet+'_undef'].to_hist()
    with uproot.open(zmm_path.with_suffix('.root')) as zmm_file:
        base_histogram_name_zmm = base_histogram_name+f'_zmm_eta{eta_range}_pt{pt_range}'
        zmm_data = zmm_file[base_histogram_name_zmm+'_data'].to_hist()
        zmm_quark = zmm_file[base_histogram_name_zmm+'_quark'].to_hist()
        zmm_gluon = zmm_file[base_histogram_name_zmm+'_gluon'].to_hist()
        zmm_undef = zmm_file[base_histogram_name_zmm+'_undef'].to_hist()

    dijet = {'data':dijet_data, 'quark':dijet_quark, 'gluon':dijet_gluon, 'undef':dijet_undef}
    zmm = {'data':zmm_data, 'quark':zmm_quark, 'gluon':zmm_gluon, 'undef':zmm_undef}
    return dijet, zmm

def main(config_path, variable, working_point, syst_name, save):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n-------------------------------------------')
    print(f'EXTRACTING SCALE FACTORS FOR WORKING POINTS')
    print(f'-------------------------------------------')
    print(f'VARIABLE:    {variable}')
    print(f'SYSTEMATIC:  {syst_name}')
    print(f'WORKING POINT:  {working_point}')

    input_file_path = Path(config['path']['input'])
    print(f'READING HISTOGRAMS FROM: {input_file_path}/wp_val_plots/')

    if save:
        output_path = config['path']['output']
        sf_save_path = Path(f'{output_path}/wp_sf_weights')
        sf_save_path.mkdir(parents=True, exist_ok=True)
        print(f'SAVING SF WEIGHTS TO:    {sf_save_path}/')

    is_nominal_syst = True if syst_name == 'nominal' else False
    base_histogram_name = variable

    wp_bins = config['WP_binning']['bins']
    for wp_bin in wp_bins:
        eta_bin = wp_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bins = wp_bin['pT']
        for i in range(len(pt_bins)-1):
            pt_range = f'{pt_bins[i]}to{pt_bins[i+1]}'

            print(f'BIN:         eta range: {eta_range}, pT range: {pt_range}')
            campaign = config['campaign']
            dijet_path = Path(f'{input_file_path}/wp_val_plots/dijet/{variable}_dijet_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP')
            zmm_path = Path(f'{input_file_path}/wp_val_plots/zmm/{variable}_zmm_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP')

            pt_low = pt_bins[i]
            pt_high = pt_bins[i]
            pt_range = str(pt_low)+'to'+str(pt_high)

            dijet_histograms, zmm_histograms = read_histograms(dijet_path, zmm_path, base_histogram_name, eta_range, pt_range)
            quark_weights, gluon_weights = weight_extraction(dijet_histograms, zmm_histograms)

            dijet_data = dijet_histograms['data']
            dijet_quark = dijet_histograms['quark']
            dijet_gluon = dijet_histograms['gluon']
            dijet_undef = dijet_histograms['undef']
            zmm_data = zmm_histograms['data']
            zmm_quark = zmm_histograms['quark']
            zmm_gluon = zmm_histograms['gluon']
            zmm_undef = zmm_histograms['undef']

            if not is_nominal_syst:
                dijet_path_syst_up = Path(f'{dijet_path}_{syst_name}_syst_up')
                zmm_path_syst_up = Path(f'{zmm_path}_{syst_name}_syst_up')
                dijet_path_syst_down = Path(f'{dijet_path}_{syst_name}_syst_down')
                zmm_path_syst_down = Path(f'{zmm_path}_{syst_name}_syst_down')

                dijet_histograms_syst_up, zmm_histograms_syst_up = read_histograms(dijet_path_syst_up, zmm_path_syst_up, base_histogram_name, eta_range, combine_pt_range)
                dijet_histograms_syst_down, zmm_histograms_syst_down = read_histograms(dijet_path_syst_down, zmm_path_syst_down, base_histogram_name, eta_range, combine_pt_range)

                quark_weights_syst_up, gluon_weights_syst_up = weight_extraction(dijet_histograms_syst_up, zmm_histograms_syst_up)
                quark_weights_syst_down, gluon_weights_syst_down = weight_extraction(dijet_histograms_syst_down, zmm_histograms_syst_down)

                dijet_quark_syst_up = dijet_histograms_syst_up['quark']
                dijet_gluon_syst_up = dijet_histograms_syst_up['gluon']
                dijet_undef_syst_up = dijet_histograms_syst_up['undef']
                zmm_quark_syst_up = zmm_histograms_syst_up['quark']
                zmm_gluon_syst_up = zmm_histograms_syst_up['gluon']
                zmm_undef_syst_up = zmm_histograms_syst_up['undef']

                dijet_quark_syst_down = dijet_histograms_syst_down['quark']
                dijet_gluon_syst_down = dijet_histograms_syst_down['gluon']
                dijet_undef_syst_down = dijet_histograms_syst_down['undef']
                zmm_quark_syst_down = zmm_histograms_syst_down['quark']
                zmm_gluon_syst_down = zmm_histograms_syst_down['gluon']
                zmm_undef_syst_down = zmm_histograms_syst_down['undef']

            dijet_histograms = {'data':dijet_data, 'quark':dijet_quark, 'gluon':dijet_gluon, 'undef':dijet_undef}
            zmm_histograms = {'data':zmm_data, 'quark':zmm_quark, 'gluon':zmm_gluon, 'undef':zmm_undef}
            quark_weights, gluon_weights = weight_extraction(dijet_histograms, zmm_histograms)

            if not is_nominal_syst:
                dijet_syst_up_histograms = {'data':dijet_data, 'quark':dijet_quark_syst_up, 'gluon':dijet_gluon_syst_up, 'undef':dijet_undef_syst_up}
                zmm_syst_up_histograms = {'data':zmm_data, 'quark':zmm_quark_syst_up, 'gluon':zmm_gluon_syst_up, 'undef':zmm_undef_syst_up}
                quark_weights_syst_up, gluon_weights_syst_up = weight_extraction(dijet_syst_up_histograms, zmm_syst_up_histograms)

                dijet_syst_down_histograms = {'data':dijet_data, 'quark':dijet_quark_syst_down, 'gluon':dijet_gluon_syst_down, 'undef':dijet_undef_syst_down}
                zmm_syst_down_histograms = {'data':zmm_data, 'quark':zmm_quark_syst_down, 'gluon':zmm_gluon_syst_down, 'undef':zmm_undef_syst_down}
                quark_weights_syst_down, gluon_weights_syst_down = weight_extraction(dijet_syst_down_histograms, zmm_syst_down_histograms)

            if save:
                weight_save_path = Path(f'{output_path}/wp_sf_weights')

                weight_save_name = weight_save_path.joinpath(f'{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP_{syst_name}_syst')
                with uproot.recreate(weight_save_name.with_suffix('.root')) as root_file:
                    root_file[f'{variable}_quark_weights'] = quark_weights
                    root_file[f'{variable}_gluon_weights'] = gluon_weights
                    if syst_name != 'nominal':
                        root_file[f'{variable}_quark_weights_syst_up'] = quark_weights_syst_up
                        root_file[f'{variable}_gluon_weights_syst_up'] = gluon_weights_syst_up
                        root_file[f'{variable}_quark_weights_syst_down'] = quark_weights_syst_down
                        root_file[f'{variable}_gluon_weights_syst_down'] = gluon_weights_syst_down

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--variable', dest='variable', type='string', default='qgl', help='Variable to plot. (Possible: qgl, particleNet, deepJet)')
    parser.add_option('-s', '--save', dest='save', action='store_true', default=False, help='Option to save plot to .png and .pdf files.')
    parser.add_option('-w', '--wp', dest='working_point', default='medium', help='Choose working point: loose, medium, tight.')
    parser.add_option('--syst', dest='syst_name', type='string', default='nominal', help='Systematic to vary: nominal, gluon, FSR, ISR, PU, JES, JER.')
    (opt, args) = parser.parse_args()

    if opt.working_point.lower() not in ['loose', 'medium', 'tight']:
        sys.exit(f'ERROR! Invalid working point: {opt.working_point.lower()}\nChoose loose, medium, or tight.')

    if opt.syst_name.lower() not in ['nominal', 'gluon', 'fsr', 'isr', 'pu', 'jes', 'jer']:
        sys.exit(f'ERROR! Invalid systematic: {opt.syst_name}\nChoose: nominal, gluon, fsr, isr, pu, jes, jer')

    if opt.variable.lower() == 'particlenet':
        opt.variable = 'particleNetAK4_QvsG'
    if opt.variable.lower() == 'deepjet':
        opt.variable = 'btagDeepFlavQG'

    main(opt.config_path, opt.variable, opt.working_point.lower(), opt.syst_name.lower(), opt.save)
