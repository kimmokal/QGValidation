import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import tomllib
import uproot
import hist
import sys

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from optparse import OptionParser
from pathlib import Path
from roc_util_functions import *

hep.style.use("CMS")
matplotlib.rcParams['font.size'] = 22

def main(config_path, display_plot, show_uncertainty_band, determine_WPs, save_plot):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n-------------------')
    print(f'PLOTTING ROC CURVES')
    print(f'-------------------')

    input_file_path = Path(config['path']['input'])
    print(f'READING FILES FROM: {input_file_path}/hists_for_roc/combined/')

    if save_plot:
        output_path = config['path']['output']
        plot_save_path = Path(f'{output_path}/roc_curves/')
        plot_save_path.mkdir(parents=True, exist_ok=True)
        print(f'WRITING OUTPUT TO:  {plot_save_path}/')

    # Assumes that there is only one bin from which to determine the tagger working points
    wp_bin = config['ROC_curves']['wp_bin'] 
    wp_eta_bin = wp_bin['eta']
    wp_eta_range = f'{wp_eta_bin[0]}to{wp_eta_bin[1]}'.replace('.','_')
    wp_pt_bin = wp_bin['pT']
    wp_pt_range = f'{wp_pt_bin[0]}to{wp_pt_bin[1]}'.replace('.','_')

    roc_bins = config['ROC_curves']['plot_bins']
    for roc_bin in roc_bins:
        eta_bin = roc_bin['eta'] # Assumes that there is only one eta_bin for each roc_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bin = roc_bin['pT'] # Assumes that there is only one pt_bin for each roc_bin
        pt_range = f'{pt_bin[0]}to{pt_bin[1]}'.replace('.','_')

        campaign = config['campaign']
        with uproot.open(f'{input_file_path}/hists_for_roc/combined/qgl_hists_combined_{campaign}_eta{eta_range}_pt{pt_range}.root') as hists:
            qgl_quark_hist = hists['quark_hist'].values()
            qgl_gluon_hist = hists['gluon_hist'].values()
            qgl_quark_hist_up = hists['quark_hist_up'].values()
            qgl_gluon_hist_up = hists['gluon_hist_up'].values()
            qgl_quark_hist_down = hists['quark_hist_down'].values()
            qgl_gluon_hist_down = hists['gluon_hist_down'].values()

        with uproot.open(f'{input_file_path}/hists_for_roc/combined/btagDeepFlavQG_hists_combined_{campaign}_eta{eta_range}_pt{pt_range}.root') as hists:
            deepjet_quark_hist = hists['quark_hist'].values()
            deepjet_gluon_hist = hists['gluon_hist'].values()
            deepjet_quark_hist_up = hists['quark_hist_up'].values()
            deepjet_gluon_hist_up = hists['gluon_hist_up'].values()
            deepjet_quark_hist_down = hists['quark_hist_down'].values()
            deepjet_gluon_hist_down = hists['gluon_hist_down'].values()

        with uproot.open(f'{input_file_path}/hists_for_roc/combined/particleNetAK4_QvsG_hists_combined_{campaign}_eta{eta_range}_pt{pt_range}.root') as hists:
            particlenet_quark_hist = hists['quark_hist'].values()
            particlenet_gluon_hist = hists['gluon_hist'].values()
            particlenet_quark_hist_up = hists['quark_hist_up'].values()
            particlenet_gluon_hist_up = hists['gluon_hist_up'].values()
            particlenet_quark_hist_down = hists['quark_hist_down'].values()
            particlenet_gluon_hist_down = hists['gluon_hist_down'].values()

        if determine_WPs and (eta_range == wp_eta_range) and (pt_range == wp_pt_range):
            print('QGL\n-------------')
            for i in np.arange(0.05, 0.95, 0.01):
                qgl_quark_eff, qgl_gluon_eff = calculate_efficiencies_from_cutoff(qgl_quark_hist, qgl_gluon_hist, np.round(i,2))
                print('cutoff value: {}'.format(np.round(i,2)))
                print('quark eff, qluon eff: {}'.format((qgl_quark_eff, qgl_gluon_eff)))
                print('ratio: {}\n'.format(qgl_quark_eff/qgl_gluon_eff))

            print('DeepJet\n-------------')
            for i in np.arange(0.05, 0.70, 0.01):
                deepjet_quark_eff, deepjet_gluon_eff = calculate_efficiencies_from_cutoff(deepjet_quark_hist, deepjet_gluon_hist, np.round(i,2))
                print('cutoff value: {}'.format(np.round(i,2)))
                print('quark eff, qluon eff: {}'.format((deepjet_quark_eff, deepjet_gluon_eff)))
                print('ratio: {}\n'.format(deepjet_quark_eff/deepjet_gluon_eff))

            print('ParticleNet\n-------------')
            for i in np.arange(0.05, 0.70, 0.01):
                particlenet_quark_eff, particlenet_gluon_eff = calculate_efficiencies_from_cutoff(particlenet_quark_hist, particlenet_gluon_hist, np.round(i,2))
                print('cutoff value: {}'.format(np.round(i,2)))
                print('quark eff, qluon eff: {}'.format((particlenet_quark_eff, particlenet_gluon_eff)))
                print('ratio: {}\n'.format(particlenet_quark_eff/particlenet_gluon_eff))

        fpr_qgl, tpr_qgl = roc_curve_from_hists(qgl_quark_hist, qgl_gluon_hist)
        fpr_deepjet, tpr_deepjet = roc_curve_from_hists(deepjet_quark_hist, deepjet_gluon_hist)
        fpr_particlenet, tpr_particlenet = roc_curve_from_hists(particlenet_quark_hist, particlenet_gluon_hist)

        fpr_qgl_up, tpr_qgl_up = roc_curve_from_hists(qgl_quark_hist_up, qgl_gluon_hist_up)
        fpr_deepjet_up, tpr_deepjet_up = roc_curve_from_hists(deepjet_quark_hist_up, deepjet_gluon_hist_up)
        fpr_particlenet_up, tpr_particlenet_up = roc_curve_from_hists(particlenet_quark_hist_up, particlenet_gluon_hist_up)

        fpr_qgl_down, tpr_qgl_down = roc_curve_from_hists(qgl_quark_hist_down, qgl_gluon_hist_down)
        fpr_deepjet_down, tpr_deepjet_down = roc_curve_from_hists(deepjet_quark_hist_down, deepjet_gluon_hist_down)
        fpr_particlenet_down, tpr_particlenet_down = roc_curve_from_hists(particlenet_quark_hist_down, particlenet_gluon_hist_down)

        qgl_auc = format(-1 * np.trapz(tpr_qgl, fpr_qgl), '.3f')
        deepjet_auc = format(-1 * np.trapz(tpr_deepjet, fpr_deepjet), '.3f')
        particlenet_auc = format(-1 * np.trapz(tpr_particlenet, fpr_particlenet), '.3f')

        fig, ax1 = plt.subplots(figsize=(8,8))

        plt.plot(fpr_qgl, tpr_qgl, linestyle='-', color='darkorange', lw = 2, label='QGL           (AUC: {})'.format(qgl_auc))
        if show_uncertainty_band:
            plt.plot(fpr_qgl_up, tpr_qgl_up, linestyle='-', color='darkorange', lw = 0.5)
            plt.plot(fpr_qgl_down, tpr_qgl_down, linestyle='-', color='darkorange', lw = 0.5)
        xfill_qgl = np.sort(np.concatenate([fpr_qgl_up, fpr_qgl_down]))
        interpolated_qgl_up = np.interp(xfill_qgl, np.sort(fpr_qgl_up), np.sort(tpr_qgl_up))
        interpolated_qgl_down = np.interp(xfill_qgl, np.sort(fpr_qgl_down), np.sort(tpr_qgl_down))
        plt.fill_between(xfill_qgl, interpolated_qgl_up, interpolated_qgl_down, interpolate=True, color='darkorange', alpha=0.2)
        
        plt.plot(fpr_deepjet, tpr_deepjet, linestyle='-', color='red', lw = 2, label='DeepJet     (AUC: {})'.format(deepjet_auc))
        if show_uncertainty_band:
            plt.plot(fpr_deepjet_up, tpr_deepjet_up, linestyle='-', color='red', lw = 0.5)
            plt.plot(fpr_deepjet_down, tpr_deepjet_down, linestyle='-', color='red', lw = 0.5)
        xfill_deepjet = np.sort(np.concatenate([fpr_deepjet_up, fpr_deepjet_down]))
        interpolated_deepjet_up = np.interp(xfill_deepjet, np.sort(fpr_deepjet_up), np.sort(tpr_deepjet_up))
        interpolated_deepjet_down = np.interp(xfill_deepjet, np.sort(fpr_deepjet_down), np.sort(tpr_deepjet_down))
        plt.fill_between(xfill_deepjet, interpolated_deepjet_up, interpolated_deepjet_down, interpolate=True, color='red', alpha=0.2)

        plt.plot(fpr_particlenet, tpr_particlenet, linestyle='-', color='blue', lw = 2, label='ParticleNet (AUC: {})'.format(particlenet_auc))
        if show_uncertainty_band:
            plt.plot(fpr_particlenet_up, tpr_particlenet_up, linestyle='-', color='blue', lw = 0.5)
            plt.plot(fpr_particlenet_down, tpr_particlenet_down, linestyle='-', color='blue', lw = 0.5)
        xfill_particlenet = np.sort(np.concatenate([fpr_particlenet_up, fpr_particlenet_down]))
        interpolated_particlenet_up = np.interp(xfill_particlenet, np.sort(fpr_particlenet_up), np.sort(tpr_particlenet_up))
        interpolated_particlenet_down = np.interp(xfill_particlenet, np.sort(fpr_particlenet_down), np.sort(tpr_particlenet_down))
        plt.fill_between(xfill_particlenet, interpolated_particlenet_up, interpolated_particlenet_down, interpolate=True, color='blue', alpha=0.2)

        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax1.xaxis.set_major_locator(MultipleLocator(0.1))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.02))
        plt.xlabel('Gluon Efficiency')
        plt.ylabel('Quark Efficiency')

        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()

        line_colors = ['blue', 'red', 'darkorange']
        box_handles = [matplotlib.patches.Patch(facecolor=color, label=label, alpha=0.2) for color, label in zip(line_colors, np.flip(ax1_labels))]
        line_handles = [matplotlib.lines.Line2D([0,1], [0,0], lw=2, color=color) for color in line_colors]
        legend_handles = list(zip(box_handles, line_handles))
        labels = np.flip(ax1_labels)

        ax1.legend((legend_handles[0],legend_handles[1],legend_handles[2]), (labels[0],labels[1],labels[2]), loc='lower right', fontsize=22, frameon=False, handleheight=0.6)

        # Workaround hack to make the legend handles look nice
        dummy_ax = ax1.twinx()
        dummy_ax.add_patch(matplotlib.patches.Rectangle((0.3, 0.04), 0.02, 0.2, color='white'))
        dummy_ax.add_patch(matplotlib.patches.Rectangle((0.39, 0.04), 0.01, 0.2, color='white'))
        dummy_ax.set_yticklabels([])
        dummy_ax.yaxis.set_major_locator(MultipleLocator(0.1))
        dummy_ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        dummy_ax.tick_params(axis='both', which='both', direction='in')

        selection_text_x = 0.68
        selection_text_y = 0.395

        eta_low = eta_bin[0]
        eta_high = eta_bin[1]
        if eta_low == '0_0':
            eta_text = r'|$\mathit{\eta}$| < ' + eta_high.replace('_','.')
        else:
            eta_text = eta_low.replace('_','.') + r' < |$\mathit{\eta}$| < ' + eta_high.replace('_','.')

        pt_low = pt_bin[0]
        pt_high = pt_bin[1]
        if str(pt_high) == '8000':
            cut_text = r'$\mathit{p_{\mathrm{T}}}$ > ' + f'{pt_low} GeV' + '\n' + eta_text
        else:
            cut_text = f'{pt_low} GeV' + r' < $\mathit{p_{\mathrm{T}}}$ < ' + f'{pt_high} GeV' + '\n' + eta_text

        plt.figtext(selection_text_x, selection_text_y, cut_text, fontsize=22, ha='center', ma='center', va='center')

        fig.tight_layout()
        hep.cms.label('Preliminary', loc=1, fontsize=22)

        if save_plot:
            save_name = f'{plot_save_path}/QG_taggers_ROC_{campaign}_eta{eta_range}_pt{pt_range}'
            if show_uncertainty_band:
                save_name = f'{save_name}_with_uncertainties'

            plt.savefig(save_name+'.png')
            plt.savefig(save_name+'.pdf')

        if display_plot:
            plt.show()

        plt.clf()
        plt.close('all')

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-d', '--display', dest='display_plot', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-u', '--uncertainty_band', dest='show_uncertainty_band', action='store_true', default=False, help='Option to show uncertainties in the plot.')
    parser.add_option('-w', '--wp', dest='determine_WPs', action='store_true', default=False, help='Option to print out quark and gluon efficiencies for WP determination.')
    parser.add_option('-s', '--save', dest='save_plot', action='store_true', default=False, help='Option to save plot to .png and .pdf files.')
    (opt, args) = parser.parse_args()

    main(opt.config_path, opt.display_plot, opt.show_uncertainty_band, opt.determine_WPs, opt.save_plot)
