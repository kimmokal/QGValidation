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
matplotlib.rcParams['font.size'] = 22

def main(config_path, variable, working_point, display, save, syst_name):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n------------------------------------')
    print(f'PLOTTING WORKING POINT SCALE FACTORS')
    print(f'------------------------------------')
    print(f'VARIABLE:    {variable}')
    print(f'SYSTEMATIC:  {syst_name}')
    print(f'WORKING POINT:  {working_point}')

    input_file_path = Path(config['path']['input'])
    print(f'READING SCALE FACTORS FROM: {input_file_path}/wp_sf_weights/')

    if save:
        output_path = config['path']['output']
        plot_save_path = Path(f'{output_path}/wp_sf_plots')
        plot_save_path.mkdir(parents=True, exist_ok=True)
        print(f'SAVING SCALE FACTOR PLOTS TO: {plot_save_path}/')
    campaign = config['campaign']

    wp_bins = config['WP_binning']['bins']
    for wp_bin in wp_bins:
        eta_bin = wp_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        quark_weights = []
        gluon_weights = []

        pt_bins = wp_bin['pT']
        for i in range(len(pt_bins)-1):
            pt_low = pt_bins[i]
            pt_high = pt_bins[i+1]
            pt_range = f'{pt_bins[i]}to{pt_bins[i+1]}'

            nominal_weight_path = Path(f'{input_file_path}/wp_sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP_nominal_syst')
            with uproot.open(nominal_weight_path.with_suffix('.root')) as f:
                quark_weights.append(f[f'{variable}_quark_weights'].to_hist())
                gluon_weights.append(f[f'{variable}_gluon_weights'].to_hist())

        wp_antiselected_quark_weights = [hist.values()[0] for hist in quark_weights]
        wp_antiselected_quark_weights_variances = [hist.variances()[0] for hist in quark_weights]
        wp_selected_quark_weights = [hist.values()[1] for hist in quark_weights]
        wp_selected_quark_weights_variances = [hist.variances()[1] for hist in quark_weights]

        wp_antiselected_gluon_weights = [hist.values()[0] for hist in gluon_weights]
        wp_antiselected_gluon_weights_variances = [hist.variances()[0] for hist in gluon_weights]
        wp_selected_gluon_weights = [hist.values()[1] for hist in gluon_weights]
        wp_selected_gluon_weights_variances = [hist.variances()[1] for hist in gluon_weights]

        if syst_name != 'nominal':
            quark_weights_syst_up = []
            gluon_weights_syst_up = []
            quark_weights_syst_down = []
            gluon_weights_syst_down = []

            for i in range(len(pt_bins)-1):
                pt_low = pt_bins[i]
                pt_high = pt_bins[i+1]
                pt_range = str(pt_low)+'to'+str(pt_high)
                    
                weight_path = Path(f'{input_file_path}/wp_sf_weights/{variable}_weights_{campaign}_eta{eta_range}_pt{pt_range}_{working_point}WP_{syst_name}_syst')
                with uproot.open(weight_path.with_suffix('.root')) as f:
                    quark_weights_syst_up.append(f[f'{variable}_quark_weights_syst_up'].to_hist())
                    gluon_weights_syst_up.append(f[f'{variable}_gluon_weights_syst_up'].to_hist())
                    quark_weights_syst_down.append(f[f'{variable}_quark_weights_syst_down'].to_hist())
                    gluon_weights_syst_down.append(f[f'{variable}_gluon_weights_syst_down'].to_hist())

            wp_antiselected_quark_weights_syst_up = [hist.values()[0] for hist in quark_weights_syst_up]
            wp_antiselected_quark_weights_syst_down = [hist.values()[0] for hist in quark_weights_syst_down]
            wp_antiselected_gluon_weights_syst_up = [hist.values()[0] for hist in gluon_weights_syst_up]
            wp_antiselected_gluon_weights_syst_down = [hist.values()[0] for hist in gluon_weights_syst_down]
            wp_selected_quark_weights_syst_up = [hist.values()[1] for hist in quark_weights_syst_up]
            wp_selected_quark_weights_syst_down = [hist.values()[1] for hist in quark_weights_syst_down]
            wp_selected_gluon_weights_syst_up = [hist.values()[1] for hist in gluon_weights_syst_up]
            wp_selected_gluon_weights_syst_down = [hist.values()[1] for hist in gluon_weights_syst_down]

            wp_selected_quark_weights_total_variances = [[],[]]
            wp_selected_gluon_weights_total_variances = [[],[]]
            wp_antiselected_quark_weights_total_variances = [[],[]]
            wp_antiselected_gluon_weights_total_variances = [[],[]]

            nBins = len(wp_selected_quark_weights)
            for i in range(nBins):
                ### SELECTED QUARK WEIGHTS
                wp_selected_quark_weight_nominal_value = wp_selected_quark_weights[i]
                wp_selected_quark_weight_statistical_up_value = np.sqrt(wp_selected_quark_weights_variances[i])
                wp_selected_quark_weight_statistical_down_value = np.sqrt(wp_selected_quark_weights_variances[i])

                wp_selected_quark_weight_syst_up_value = wp_selected_quark_weights_syst_up[i]
                wp_selected_quark_weight_syst_down_value = wp_selected_quark_weights_syst_down[i]
                wp_selected_quark_weight_syst_up_difference = wp_selected_quark_weight_syst_up_value - wp_selected_quark_weight_nominal_value
                wp_selected_quark_weight_syst_down_difference = wp_selected_quark_weight_syst_down_value - wp_selected_quark_weight_nominal_value 
                wp_selected_quark_weight_syst_up_variation = np.max([wp_selected_quark_weight_syst_up_difference, wp_selected_quark_weight_syst_down_difference, 0])
                wp_selected_quark_weight_syst_down_variation = np.min([wp_selected_quark_weight_syst_up_difference, wp_selected_quark_weight_syst_down_difference, 0])

                ### SELECTED GLUON WEIGHTS
                wp_selected_gluon_weight_nominal_value = wp_selected_gluon_weights[i]
                wp_selected_gluon_weight_statistical_up_value = np.sqrt(wp_selected_gluon_weights_variances[i])
                wp_selected_gluon_weight_statistical_down_value = np.sqrt(wp_selected_gluon_weights_variances[i])

                wp_selected_gluon_weight_syst_up_value = wp_selected_gluon_weights_syst_up[i]
                wp_selected_gluon_weight_syst_down_value = wp_selected_gluon_weights_syst_down[i]
                wp_selected_gluon_weight_syst_up_difference = wp_selected_gluon_weight_syst_up_value - wp_selected_gluon_weight_nominal_value
                wp_selected_gluon_weight_syst_down_difference = wp_selected_gluon_weight_syst_down_value - wp_selected_gluon_weight_nominal_value 
                wp_selected_gluon_weight_syst_up_variation = np.max([wp_selected_gluon_weight_syst_up_difference, wp_selected_gluon_weight_syst_down_difference, 0])
                wp_selected_gluon_weight_syst_down_variation = np.min([wp_selected_gluon_weight_syst_up_difference, wp_selected_gluon_weight_syst_down_difference, 0])

                ### ANTISELECTED QUARK WEIGHTS
                wp_antiselected_quark_weight_nominal_value = wp_antiselected_quark_weights[i]
                wp_antiselected_quark_weight_statistical_up_value = np.sqrt(wp_antiselected_quark_weights_variances[i])
                wp_antiselected_quark_weight_statistical_down_value = np.sqrt(wp_antiselected_quark_weights_variances[i])

                wp_antiselected_quark_weight_syst_up_value = wp_antiselected_quark_weights_syst_up[i]
                wp_antiselected_quark_weight_syst_down_value = wp_antiselected_quark_weights_syst_down[i]
                wp_antiselected_quark_weight_syst_up_difference = wp_antiselected_quark_weight_syst_up_value - wp_antiselected_quark_weight_nominal_value
                wp_antiselected_quark_weight_syst_down_difference = wp_antiselected_quark_weight_syst_down_value - wp_antiselected_quark_weight_nominal_value 
                wp_antiselected_quark_weight_syst_up_variation = np.max([wp_antiselected_quark_weight_syst_up_difference, wp_antiselected_quark_weight_syst_down_difference, 0])
                wp_antiselected_quark_weight_syst_down_variation = np.min([wp_antiselected_quark_weight_syst_up_difference, wp_antiselected_quark_weight_syst_down_difference, 0])

                ### ANTISELECTED GLUON WEIGHTS
                wp_antiselected_gluon_weight_nominal_value = wp_antiselected_gluon_weights[i]
                wp_antiselected_gluon_weight_statistical_up_value = np.sqrt(wp_antiselected_gluon_weights_variances[i])
                wp_antiselected_gluon_weight_statistical_down_value = np.sqrt(wp_antiselected_gluon_weights_variances[i])

                wp_antiselected_gluon_weight_syst_up_value = wp_antiselected_gluon_weights_syst_up[i]
                wp_antiselected_gluon_weight_syst_down_value = wp_antiselected_gluon_weights_syst_down[i]
                wp_antiselected_gluon_weight_syst_up_difference = wp_antiselected_gluon_weight_syst_up_value - wp_antiselected_gluon_weight_nominal_value
                wp_antiselected_gluon_weight_syst_down_difference = wp_antiselected_gluon_weight_syst_down_value - wp_antiselected_gluon_weight_nominal_value 
                wp_antiselected_gluon_weight_syst_up_variation = np.max([wp_antiselected_gluon_weight_syst_up_difference, wp_antiselected_gluon_weight_syst_down_difference, 0])
                wp_antiselected_gluon_weight_syst_down_variation = np.min([wp_antiselected_gluon_weight_syst_up_difference, wp_antiselected_gluon_weight_syst_down_difference, 0])

                ### SAVE VARIATIONS
                wp_selected_quark_weights_total_variances[0].append(wp_selected_quark_weight_syst_down_variation**2)
                wp_selected_quark_weights_total_variances[1].append(wp_selected_quark_weight_syst_up_variation**2)
                wp_selected_gluon_weights_total_variances[0].append(wp_selected_gluon_weight_syst_down_variation**2)
                wp_selected_gluon_weights_total_variances[1].append(wp_selected_gluon_weight_syst_up_variation**2)
                
                wp_antiselected_quark_weights_total_variances[0].append(wp_antiselected_quark_weight_syst_down_variation**2)
                wp_antiselected_quark_weights_total_variances[1].append(wp_antiselected_quark_weight_syst_up_variation**2)
                wp_antiselected_gluon_weights_total_variances[0].append(wp_antiselected_gluon_weight_syst_down_variation**2)
                wp_antiselected_gluon_weights_total_variances[1].append(wp_antiselected_gluon_weight_syst_up_variation**2)

        fig, ax = plt.subplots(figsize=(8,8))

        x = np.arange(1, len(wp_selected_quark_weights)+1)
        x_below = np.arange(1, len(wp_selected_quark_weights)+1) + 0.15
        x_above = np.arange(1, len(wp_selected_quark_weights)+1) - 0.15

        plt.hlines(1., 0., x[-1]+1, linestyles=(0, (4, 4)), linewidth=1, colors='black', alpha=0.9)

        if syst_name != 'nominal':
            q_above = plt.errorbar(x_above, wp_selected_quark_weights, yerr=np.sqrt(wp_selected_quark_weights_total_variances), capsize=3, fmt='o', color='#0000CC', label='Quark weights above WP')
            g_above = plt.errorbar(x_above, wp_selected_gluon_weights, yerr=np.sqrt(wp_selected_gluon_weights_total_variances), capsize=3, fmt='o', color='#CC0000', label='Gluon weights above WP')
            q_below = plt.errorbar(x_below, wp_antiselected_quark_weights, yerr=np.sqrt(wp_antiselected_quark_weights_total_variances), capsize=3, fmt='s', mfc='none', color='#3232FF', label='Quark weights below WP')
            g_below = plt.errorbar(x_below, wp_antiselected_gluon_weights, yerr=np.sqrt(wp_antiselected_gluon_weights_total_variances), capsize=3, fmt='s', mfc='none', color='#FF3232', label='Gluon weights below WP')
        else:
            q_above = plt.errorbar(x_above, wp_selected_quark_weights, yerr=np.sqrt(wp_selected_quark_weights_variances), capsize=3, fmt='o', color='#0000CC', label='Quark weights above WP')
            g_above = plt.errorbar(x_above, wp_selected_gluon_weights, yerr=np.sqrt(wp_selected_gluon_weights_variances), capsize=3, fmt='o', color='#CC0000', label='Gluon weights above WP')
            q_below = plt.errorbar(x_below, wp_antiselected_quark_weights, yerr=np.sqrt(wp_antiselected_quark_weights_variances), capsize=3, fmt='s', mfc='none', color='#3232FF', label='Quark weights below WP')
            g_below = plt.errorbar(x_below, wp_antiselected_gluon_weights, yerr=np.sqrt(wp_antiselected_gluon_weights_variances), capsize=3, fmt='s', mfc='none', color='#FF3232', label='Gluon weights below WP')

        legend_y = 0.02
        legend_x = 0.0
        legend_fontsize = 17
        plt.legend(loc=(legend_x,legend_y), fontsize=legend_fontsize, ncol=2, handletextpad=0.01, columnspacing=0.29, handles=[q_above, q_below, g_above, g_below])
        
        ylim_low = 0.1
        ylim_high = 2.3
        yticklabels = ['','0.2','','0.4','','0.6','','0.8','','1.0','','1.2','','1.4','','1.6','','1.8','','2.0','','2.2','']
        
        plt.ylim([ylim_low, ylim_high])
        ax.yaxis.set_major_locator(FixedLocator(np.arange(ylim_low, ylim_high+0.10, 0.10)))
        ax.yaxis.set_minor_locator(FixedLocator(np.arange(ylim_low-0.10, ylim_high+0.05, 0.05)))
        ax.set_yticklabels(yticklabels)

        plt.xlim([0., x[-1]+1])
        ax.xaxis.set_major_locator(FixedLocator(np.arange(1., x[-1]+1, 1.)))
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(1., x[-1]+1, 1.)))
        xtick_labels = [str(pt_bins[i])+'–'+str(pt_bins[i+1]) for i in range(len(pt_bins)-1)]
        xtick_labels[-1] = '>{}'.format(pt_bins[-2])
        ax.set_xticklabels(xtick_labels)
        plt.xticks(rotation=45, ha='right')

        plt.grid(visible=True, which='major', alpha=0.7)
   
        plot_label = {
                'qgl' : 'Quark-Gluon Likelihood',
                'qgl_new' : 'Quark-Gluon Likelihood',
                'particleNetAK4_QvsG' : 'ParticleNet',
                'btagDeepFlavQG': 'DeepJet'
                }

        eta_low = eta_range.split('to')[0]
        eta_high = eta_range.split('to')[1]
        if eta_low == '0_0':
            eta_text = r'|$\mathit{\eta}$| < ' + eta_high.replace('_','.')
        else:
            eta_text = eta_low.replace('_','.') + r' < |$\mathit{\eta}$| < ' + eta_high.replace('_','.')
        selection_text = plot_label[variable] + '\n{} WP'.format(working_point.capitalize()) + '\n{}'.format(eta_text)
        
        selection_text_x = 0.355
        selection_text_y = 0.82
        plt.figtext(selection_text_x, selection_text_y, selection_text, fontsize=20, ha='center', ma='center', va='center')

        syst_label = {'nominal' : 'Statistical',
                'total' : 'Total (stat. + syst.)',
                'gluon': 'Gluon Weight',
                'fsr': 'FSR',
                'isr': 'ISR',
                'pu': 'PU',
                'jes': 'JES',
                'jer': 'JER'}

        systematic_text = 'Uncertainties:\n{}'.format(syst_label[syst_name])
        systematic_text_x = 0.76
        systematic_text_y = 0.84
        plt.figtext(systematic_text_x, systematic_text_y, systematic_text, fontsize=20, ha='center', ma='center', va='center')
   
        plt.ylabel('Scale factor')
        plt.xlabel(r'$p_T$ bin (GeV)', loc='center')
        hep.cms.label('Preliminary', loc=0, data=True, lumi='41.5', fontsize=22)
        fig.tight_layout()

        if save:
            plot_save_name = plot_save_path.joinpath(f'{variable}_weights_{campaign}_eta{eta_range}_{working_point}WP_{syst_name}_syst')
            plt.savefig(plot_save_name.with_suffix('.png'), bbox_inches='tight', pad_inches=0.1)
            plt.savefig(plot_save_name.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.1)
            
        if display:
            plt.show()

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--var', dest='variable', default='qgl', help='Variable to plot (qgl, particlenet, deepjet)')
    parser.add_option('-d', '--display', dest='display', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-s', '--save', dest='save', action='store_true', default=False, help='Option to save image and .root files.')
    parser.add_option('-w', '--wp', dest='working_point', default='medium', help='Choose working point: loose, medium, tight.')
    parser.add_option('--syst', dest='syst_name', default='nominal', help='Option to plot systematic uncertainties.')
    (opt, args) = parser.parse_args()

    if opt.working_point.lower() not in ['loose', 'medium', 'tight']:
        sys.exit('ERROR! Invalid working point: {}\nChoose loose, medium, or tight.'.format(opt.working_point.lower()))

    if opt.syst_name.lower() not in ['nominal', 'gluon', 'fsr', 'isr', 'pu', 'jes', 'jer', 'total']:
        sys.exit('ERROR! Invalid systematic variation: {}\nChoose nominal, gluon, fsr, isr, pu, jes, jer, total'.format(opt.working_point.lower()))

    if opt.variable.lower() == 'particlenet':
        opt.variable = 'particleNetAK4_QvsG'
    if opt.variable.lower() == 'deepjet':
        opt.variable = 'btagDeepFlavQG'

    main(opt.config_path, opt.variable, opt.working_point.lower(), opt.display, opt.save, opt.syst_name.lower())
