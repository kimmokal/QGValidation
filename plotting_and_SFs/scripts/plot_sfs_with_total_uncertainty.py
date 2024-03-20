import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import tomllib
import uproot
import hist
import sys

from optparse import OptionParser
from matplotlib.ticker import FixedLocator
from pathlib import Path

hep.style.use('CMS')
matplotlib.rcParams['font.size'] = 22

def main(config_path, variable, plot_quark, plot_gluon, do_fit, display, save):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n-------------------------------------------')
    print(f'PLOTTING SCALE FACTORS WITH TOTAL UNCERTAINTY')
    print(f'---------------------------------------------')
    input_file_path = Path(config['path']['input'])
    print(f'READING SCALE FACTORS FROM: {input_file_path}/sf_weights/')

    if save:
        output_path = config['path']['output']
        plot_save_path = Path(f'{output_path}/sf_plots')
        plot_save_path.mkdir(parents=True, exist_ok=True)
        print(f'SAVING SF PLOTS TO:         {plot_save_path}/')

    print(f'\nVARIABLE:    {variable}')
    print('FLAVOUR:     {}'.format('gluon' if plot_gluon else 'quark'))
    if do_fit:
        print('WITH FIT')

    var_label = {
            'qgl' : 'qgl',
            # 'qgl' : 'qgl_new',
            'particlenet' : 'particleNetAK4_QvsG',
            'deepjet': 'btagDeepFlavQG'
            }

    sf_bins = config['SF_binning']['bins']
    for sf_bin in sf_bins:
        eta_bin = sf_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bins = sf_bin['pT']
        for i in range(len(pt_bins)-1):
            pt_range = f'{pt_bins[i]}to{pt_bins[i+1]}'
            print(f'BIN:         eta range: {eta_range}, pT range: {pt_range}')

            campaign = config['campaign']
            with uproot.open(Path(f'{input_file_path}/sf_weights/{var_label[variable]}_weights_{campaign}_eta{eta_range}_pt{pt_range}_combined_syst').with_suffix('.root')) as f:
                quark_weights_nominal = f['quark_weights_nominal']
                quark_weights_stat_var = f['quark_weights_stat_var']
                quark_weights_combined_unc_up = f['quark_weights_combined_unc_up']
                quark_weights_combined_unc_down = f['quark_weights_combined_unc_down']
                gluon_weights_nominal = f['gluon_weights_nominal']
                gluon_weights_stat_var = f['gluon_weights_stat_var']
                gluon_weights_combined_unc_up = f['gluon_weights_combined_unc_up']
                gluon_weights_combined_unc_down = f['gluon_weights_combined_unc_down']

            if plot_quark:
                channel_text = 'Quark weights, Pythia'
                fit_polynomial_order = {'qgl' : 5,
                        'particlenet' : 5,
                        'deepjet' : 7}

                bin_centers = quark_weights_nominal.axis().centers()
                bin_widths = quark_weights_nominal.axis().widths()
                plot_weights = quark_weights_nominal.values()
                plot_stat_var = quark_weights_stat_var.values()
                plot_weights_var_up = plot_weights + quark_weights_combined_unc_up.values()
                plot_weights_var_down = plot_weights - quark_weights_combined_unc_down.values()
            elif plot_gluon:
                channel_text = 'Gluon weights, Pythia'
                fit_polynomial_order = {'qgl' : 7,
                        'particlenet' : 7,
                        'deepjet' : 7}

                bin_centers = gluon_weights_nominal.axis().centers()
                bin_widths = gluon_weights_nominal.axis().widths()
                plot_weights = gluon_weights_nominal.values()
                plot_stat_var = gluon_weights_stat_var.values()
                plot_weights_var_up = plot_weights + gluon_weights_combined_unc_up.values()
                plot_weights_var_down = plot_weights - gluon_weights_combined_unc_down.values()

            fig, ax = plt.subplots(figsize=(8,8))
            plt.hlines(1., 0., 1., linestyles=(0, (4, 4)), linewidth=1, colors='black', alpha=0.9)

            valid_bins = (plot_weights > 0)
            bin_centers = bin_centers[valid_bins]
            bin_widths = bin_widths[valid_bins]
            plot_weights = plot_weights[valid_bins]
            plot_weights_var_up = plot_weights_var_up[valid_bins]
            plot_weights_var_down = plot_weights_var_down[valid_bins]
            plot_stat_var = plot_stat_var[valid_bins]

            weights_nominal = plt.errorbar(bin_centers, plot_weights, yerr=plot_stat_var, xerr=bin_widths/2, fmt='o', color='black', lw=1.5, markersize=5)

            if do_fit:
                fit_x = np.linspace(0.,1.,200)

                fit_coeffs, coeff_cov_matrix = np.polyfit(bin_centers, plot_weights, fit_polynomial_order[variable], w=1/plot_stat_var, cov='unscaled')
                fit_y = np.polyval(fit_coeffs, fit_x)
                fit_nominal, = plt.plot(fit_x, fit_y, color='black', lw=2)

                # Calculate fit uncertainties
                V = np.vander(fit_x, N=len(fit_coeffs))
                sigma_fit_y = np.sqrt(np.diag(V @ coeff_cov_matrix @ V.T))

                fit_unc_up_coeffs = np.polyfit(fit_x, fit_y+sigma_fit_y, fit_polynomial_order[variable])
                fit_y_unc_up = np.polyval(fit_unc_up_coeffs, fit_x)

                fit_unc_down_coeffs = np.polyfit(fit_x, fit_y-sigma_fit_y, fit_polynomial_order[variable])
                fit_y_unc_down = np.polyval(fit_unc_down_coeffs, fit_x)

                # Fit to total statistical+systematic uncertainty
                fit_stat_syst_unc_up_coeffs = np.polyfit(bin_centers, plot_weights_var_up, fit_polynomial_order[variable])
                fit_y_stat_syst_unc_up = np.polyval(fit_stat_syst_unc_up_coeffs, fit_x)

                fit_stat_syst_unc_down_coeffs = np.polyfit(bin_centers, plot_weights_var_down, fit_polynomial_order[variable])
                fit_y_stat_syst_unc_down = np.polyval(fit_stat_syst_unc_down_coeffs, fit_x)

                # Combine fit+stat.+syst. uncertainty and perform a fit to those
                total_unc_up = fit_y + np.sqrt((fit_y-fit_y_unc_up)**2 + (fit_y-fit_y_stat_syst_unc_up)**2)
                total_unc_down = fit_y - np.sqrt((fit_y-fit_y_unc_down)**2 + (fit_y-fit_y_stat_syst_unc_down)**2)

                fit_total_unc_up_coeffs = np.polyfit(fit_x, total_unc_up, fit_polynomial_order[variable])
                fit_y_total_unc_up = np.polyval(fit_total_unc_up_coeffs, fit_x)

                fit_total_unc_down_coeffs = np.polyfit(fit_x, total_unc_down, fit_polynomial_order[variable])
                fit_y_total_unc_down = np.polyval(fit_total_unc_down_coeffs, fit_x)

                total_unc_fill = plt.fill_between(fit_x, fit_y_total_unc_up, fit_y_total_unc_down, hatch='//////', alpha=.25)

                legend_handles = [weights_nominal, fit_nominal, total_unc_fill]
                legend_labels = ['Weights', 'Fit', 'Total uncertainty\n(stat. + syst. + fit)']
                legend_y = 0.744
            else:
                total_uncertainty = plt.bar(bin_centers, (plot_weights_var_up-plot_weights_var_down), width=bin_widths, bottom=plot_weights_var_down, edgecolor=None, hatch='//////', alpha=0.25)

                legend_handles = [weights_nominal, total_uncertainty]
                legend_labels = ['Weights', 'Total uncertainty\n(stat. + syst.)']
                legend_y = 0.8

            legend_x = 0.475
            plt.legend(legend_handles, legend_labels, loc=(legend_x, legend_y), fontsize=18, handlelength=0.8, handleheight=0.8, handletextpad=0.8)

            channel_text_x = 0.370
            channel_text_y = 0.33
            plt.figtext(channel_text_x, channel_text_y, channel_text, fontsize=20, fontweight='semibold')

            low_eta = eta_range.split('to')[0]
            high_eta = eta_range.split('to')[1]
            low_pt = pt_range.split('to')[0]
            high_pt = pt_range.split('to')[1]
            if low_eta == '0_0':
                eta_text = r'|$\mathit{\eta}$| < ' + str(high_eta).replace('_','.') 
            else:
                eta_text = str(low_eta).replace('_','.')  + r' < |$\mathit{\eta}$| < ' + str(high_eta).replace('_','.') 

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
            hep.cms.label('Preliminary', loc=1, data=True, lumi='41.5', fontsize=22)
            fig.tight_layout()
            if save:
                if plot_quark:
                    flavour = 'quark'
                elif plot_gluon:
                    flavour = 'gluon'

                save_name = Path(f'{plot_save_path}/{var_label[variable]}_{flavour}_weights_{campaign}_eta{eta_range}_pt{pt_range}_combined_syst')

                if do_fit:
                    save_name = Path(f'{save_name}_withFit')

                plt.savefig(save_name.with_suffix('.png'))
                plt.savefig(save_name.with_suffix('.pdf'))

            if display:
                plt.show()

            plt.clf()
            plt.close('all')

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--var', dest='variable', default='qgl', help='Variable to plot (qgl, particlenet, deepjet)')
    parser.add_option('-q', '--quark', dest='plot_quark', action='store_true', default=False, help='Plot quark weights.')
    parser.add_option('-g', '--gluon', dest='plot_gluon', action='store_true', default=False, help='Plot gluon weights.')
    parser.add_option('-f', '--fit', dest='fit', action='store_true', default=False, help='Option to perform a fit to the SFs')
    parser.add_option('-d', '--display', dest='display', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-s', '--save', dest='save', action='store_true', default=False, help='Option to save image and .root files.')
    (opt, args) = parser.parse_args()

    if (opt.plot_gluon == False) and (opt.plot_quark == False):
        sys.exit('ERROR!\nChoice of quark or gluon is required. Use option -q or -g.')
    if (opt.plot_gluon and opt.plot_quark) == True:
        sys.exit('ERROR!\nOnly choose either quark or gluon weights to extract.')

    main(opt.config_path, opt.variable, opt.plot_quark, opt.plot_gluon, opt.fit, opt.display, opt.save)
