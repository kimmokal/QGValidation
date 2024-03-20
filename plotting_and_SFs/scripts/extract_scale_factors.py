import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import tomllib
import uproot
import hist
import sys

from pathlib import Path
from matplotlib.ticker import FixedLocator
from optparse import OptionParser

hep.style.use('CMS')
matplotlib.rcParams['font.size'] = 22

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

def main(config_path, variable, extract_quark, extract_gluon, syst_name, display, save):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n------------------------')
    print(f'EXTRACTING SCALE FACTORS')
    print(f'------------------------')
    print(f'VARIABLE:    {variable}')
    print(f'SYSTEMATIC:  {syst_name}')
    print('FLAVOUR:     {}'.format('gluon' if extract_gluon else 'quark'))

    input_file_path = Path(config['path']['input'])
    print(f'READING HISTOGRAMS FROM: {input_file_path}/val_plots/')

    if save:
        output_path = config['path']['output']
        plot_save_path = Path(f'{output_path}/sf_plots')
        plot_save_path.mkdir(parents=True, exist_ok=True)
        print(f'SAVING SF PLOTS TO:      {plot_save_path}/')

        sf_save_path = Path(f'{output_path}/sf_weights')
        sf_save_path.mkdir(parents=True, exist_ok=True)
        print(f'SAVING SF WEIGHTS TO:    {sf_save_path}/')

    is_nominal_syst = True if syst_name == 'nominal' else False

    var_label = {
            'qgl' : 'qgl',
            # 'qgl' : 'qgl_new',
            'deepjet' : 'btagDeepFlavQG',
            'particlenet' : 'particleNetAK4_QvsG'
            }
    base_histogram_name = var_label[variable]

    sf_bins = config['SF_binning']['bins']
    for sf_bin in sf_bins:
        eta_bin = sf_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bins = sf_bin['pT']
        for i in range(len(pt_bins)-1):
            pt_range = f'{pt_bins[i]}to{pt_bins[i+1]}'

            print(f'BIN:         eta range: {eta_range}, pT range: {pt_range}')
            campaign = config['campaign']
            dijet_path = Path(f'{input_file_path}/val_plots/dijet/{var_label[variable]}_dijet_{campaign}_eta{eta_range}_pt{pt_range}')
            zmm_path = Path(f'{input_file_path}/val_plots/zmm/{var_label[variable]}_zmm_{campaign}_eta{eta_range}_pt{pt_range}')

            dijet_histograms, zmm_histograms = read_histograms(dijet_path, zmm_path, base_histogram_name, eta_range, pt_range)
            quark_weights, gluon_weights = weight_extraction(dijet_histograms, zmm_histograms)

            if not is_nominal_syst:
                dijet_path_syst_up = Path(f'{dijet_path}_{syst_name}_syst_up')
                zmm_path_syst_up = Path(f'{zmm_path}_{syst_name}_syst_up')
                dijet_path_syst_down = Path(f'{dijet_path}_{syst_name}_syst_down')
                zmm_path_syst_down = Path(f'{zmm_path}_{syst_name}_syst_down')

                dijet_histograms_syst_up, zmm_histograms_syst_up = read_histograms(dijet_path_syst_up, zmm_path_syst_up, base_histogram_name, eta_range, pt_range)
                dijet_histograms_syst_down, zmm_histograms_syst_down = read_histograms(dijet_path_syst_down, zmm_path_syst_down, base_histogram_name, eta_range, pt_range)

                quark_weights_syst_up, gluon_weights_syst_up = weight_extraction(dijet_histograms_syst_up, zmm_histograms_syst_up)
                quark_weights_syst_down, gluon_weights_syst_down = weight_extraction(dijet_histograms_syst_down, zmm_histograms_syst_down)

            if extract_quark:
                channel_text = 'Quark weights, Pythia'
                plot_weights = quark_weights.values()
                plot_variances = quark_weights.variances()
                if not is_nominal_syst:
                    plot_weights_syst_up = quark_weights_syst_up.values()
                    plot_variances_syst_up = quark_weights_syst_up.variances()
                    plot_weights_syst_down = quark_weights_syst_down.values()
                    plot_variances_syst_down = quark_weights_syst_down.variances()
            elif extract_gluon:
                channel_text = 'Gluon weights, Pythia'
                plot_weights = gluon_weights.values()
                plot_variances = gluon_weights.variances()
                if not is_nominal_syst:
                    plot_weights_syst_up = gluon_weights_syst_up.values()
                    plot_variances_syst_up = gluon_weights_syst_up.variances()
                    plot_weights_syst_down = gluon_weights_syst_down.values()
                    plot_variances_syst_down = gluon_weights_syst_down.variances()

            bin_centers = dijet_histograms['data'].axes.centers[0]
            bin_widths = dijet_histograms['data'].axes.widths[0]

            fig, ax = plt.subplots(figsize=(8,8))
            plt.hlines(1., 0., 1., linestyles=(0, (4, 4)), linewidth=1, colors='black', alpha=0.9)

            weights_nominal = plt.errorbar(bin_centers, plot_weights, yerr=np.sqrt(plot_variances), xerr=bin_widths/2, fmt='o', color='black', lw=1.5, markersize=5, label='Weights (nominal)', zorder=10)

            if is_nominal_syst:
                plt.legend(weights_nominal, ['Weights'], loc='upper right', fontsize=18, handlelength=0.8, handleheight=0.8, handletextpad=0.8)
            else:
                weights_up = plt.errorbar(bin_centers, plot_weights_syst_up, xerr=bin_widths/2, fmt='o', lw=1, markersize=3, color='red', label='Weights (syst. up)')
                weights_down = plt.errorbar(bin_centers, plot_weights_syst_down, xerr=bin_widths/2, fmt='o', lw=1, markersize=3, color='blue', label='Weights (syst. down)')

                uncertainty_bar_up = plt.bar(bin_centers, (plot_weights_syst_up-plot_weights), bottom=plot_weights, width=bin_widths, color='red', alpha=0.4)
                uncertainty_bar_down = plt.bar(bin_centers, (plot_weights_syst_down-plot_weights), bottom=plot_weights, width=bin_widths, color='blue', alpha=0.4)

                plt.legend([weights_nominal, (weights_up, uncertainty_bar_up), (weights_down, uncertainty_bar_down)], ['Weights (nominal)', 'Weights (syst. up)', 'Weights (syst. down)'],loc='upper right', fontsize=18, handlelength=0.8, handleheight=0.8, handletextpad=0.8)

            systematic_label = {
                    'nominal' : 'Nominal',
                    'gluon' : '\nIncoming LHE gluons',
                    'fsr' : ' FSR',
                    'isr' : ' ISR',
                    'pu' : 'PU',
                    'jes' : 'JES',
                    'jer' : 'JER',
                    }

            if not is_nominal_syst:
                systematic_text = f'Systematic: {systematic_label[syst_name]}'
                systematic_text_x = 0.57
                systematic_text_y = 0.70 if syst_name=='gluon' else 0.73
                plt.figtext(systematic_text_x, systematic_text_y, systematic_text, fontsize=18)

            channel_text_x = 0.370
            channel_text_y = 0.33
            plt.figtext(channel_text_x, channel_text_y, channel_text, fontsize=20, fontweight='semibold')

            low_eta = eta_range.split('to')[0]
            high_eta = eta_range.split('to')[1]
            if low_eta == '0_0':
                eta_text = r'|$\mathit{\eta}$| < ' + str(high_eta).replace('_','.')
            else:
                eta_text = str(low_eta).replace('_','.') + r' < |$\mathit{\eta}$| < ' + str(high_eta).replace('_','.')

            low_pt = pt_range.split('to')[0]
            high_pt = pt_range.split('to')[1]
            if pt_range == '30to8000':
                selection_text = r'$\mathit{p_T}$ > 30 GeV' + '\n' + eta_text
            elif pt_range == '80to8000':
                selection_text = r'$\mathit{p_T}$ > 80 GeV' + '\n' + eta_text
            else:
                selection_text = str(low_pt) + r' GeV < $\mathit{p_T}$ < ' + str(high_pt) + ' GeV' + '\n' + eta_text

            selection_text_x = 0.54
            selection_text_y = 0.276
            plt.figtext(selection_text_x, selection_text_y, selection_text, fontsize=20, ha='center', ma='center', va='center')

            plt.xlim([0.,1.])
            plt.ylim([0.0,2.3])
            ax.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.40, 0.10)))
            ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 2.35, 0.05)))
            ax.set_yticklabels(['0.0','','0.2','','0.4','','0.6','','0.8','','1.0','','1.2','','1.4','','1.6','','1.8','','2.0','','2.2',''])
            ax.xaxis.set_major_locator(FixedLocator(np.arange(0.0, 1.10, 0.10)))
            ax.xaxis.set_minor_locator(FixedLocator(np.arange(0.0, 1.05, 0.05)))
            ax.set_xticklabels(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])
            ax.tick_params(axis='both', which='major', pad=7)

            plt.grid(visible=True, which='major')
            plt.grid(visible=True, which='minor', axis='x')

            axis_label = {'qgl' : 'Quark-Gluon Likelihood',
                    'particlenet' : 'ParticleNet discriminant',
                    'deepjet': 'DeepJet discriminant'}

            plt.ylabel('Scale factor')
            plt.xlabel(axis_label[variable], loc='right')
            hep.cms.label('Preliminary', loc=1, lumi='41.5', data=True, fontsize=22)
            fig.tight_layout()

            if save:
                flavour = 'quark' if extract_quark else 'gluon'
                plot_save_name = plot_save_path.joinpath(f'{var_label[variable]}_{flavour}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{syst_name}_syst')

                plt.savefig(plot_save_name.with_suffix('.png'))
                plt.savefig(plot_save_name.with_suffix('.pdf'))

                sf_save_name = sf_save_path.joinpath(f'{var_label[variable]}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{syst_name}_syst')
                with uproot.recreate(sf_save_name.with_suffix('.root')) as root_file:
                    root_file[f'{var_label[variable]}_quark_weights'] = quark_weights
                    root_file[f'{var_label[variable]}_gluon_weights'] = gluon_weights
                    if syst_name != 'nominal':
                        root_file[f'{var_label[variable]}_quark_weights_syst_up'] = quark_weights_syst_up
                        root_file[f'{var_label[variable]}_gluon_weights_syst_up'] = gluon_weights_syst_up
                        root_file[f'{var_label[variable]}_quark_weights_syst_down'] = quark_weights_syst_down
                        root_file[f'{var_label[variable]}_gluon_weights_syst_down'] = gluon_weights_syst_down

            if display:
                plt.show()

            plt.clf()
            plt.close('all')

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--variable', dest='variable', type='string', default='qgl', help='Variable to plot. (Possible: qgl, particleNet, deepJet)')
    parser.add_option('-q', '--quark', dest='extract_quark', action='store_true', default=False, help='Extract quark weights.')
    parser.add_option('-g', '--gluon', dest='extract_gluon', action='store_true', default=False, help='Extract gluon weights.')
    parser.add_option('-d', '--display', dest='display', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-s', '--save', dest='save', action='store_true', default=False, help='Option to save plot to .png and .pdf files.')
    parser.add_option('--syst', dest='syst_name', type='string', default='nominal', help='Systematic to vary: nominal, gluon, FSR, ISR, PU, JES, JER.')
    (opt, args) = parser.parse_args()

    if (opt.extract_gluon == False) and (opt.extract_quark == False):
        sys.exit('ERROR!\nChoice of quark or gluon is required. Use option -q or -g.')
    if (opt.extract_gluon and opt.extract_quark) == True:
        sys.exit('ERROR!\nOnly choose either quark or gluon weights to extract.')

    if opt.syst_name.lower() not in ['nominal', 'gluon', 'fsr', 'isr', 'pu', 'jes', 'jer']:
        sys.exit(f'ERROR! Invalid systematic: {opt.syst_name}\nChoose: nominal, gluon, FSR, ISR, PU, JES, JER')

    if opt.variable.lower() not in ['qgl', 'deepjet', 'particlenet']:
        sys.exit(f'ERROR! Invalid variable: {opt.variable}\nChoose: qgl, deepJet, particleNet')

    main(opt.config_path, opt.variable.lower(), opt.extract_quark, opt.extract_gluon, opt.syst_name.lower(), opt.display, opt.save)
