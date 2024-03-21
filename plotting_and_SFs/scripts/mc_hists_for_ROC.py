import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import tomllib
import uproot
import hist
import json
import sys
import correctionlib

from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from optparse import OptionParser
from correctionlib import convert

hep.style.use('CMS')
matplotlib.rcParams['font.size'] = 18

def main(config_path, variable, display, save_plot, save_root, inclusive, channel):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n---------------------------------------')
    print(f'PROCESSING MC HISTOGRAMS FOR ROC CURVES')
    print(f'---------------------------------------')
    print(f'PLOTTING VARIABLE:  {variable}')

    input_file_path = Path(config['path']['input_samples'])
    print(f'READING FILES FROM: {input_file_path}/')

    input_sf_path = Path(config['path']['input'])
    print(f'READING SCALE FACTORS FROM: {input_sf_path}/sf_weights/')

    if save_plot or save_root:
        output_path = config['path']['output']
        plot_save_path = Path(f'{output_path}/hists_for_roc/{channel}')
        plot_save_path.mkdir(parents=True, exist_ok=True)
        print(f'WRITING OUTPUT TO:  {plot_save_path}/')

    if channel == 'zmm':
        plot_var = 'Jet_'+variable
        
        mc_file = input_file_path.joinpath('mc_DYJetsToLL')
        with uproot.open(mc_file.with_suffix('.root')) as file:
            var_mc = file['Events'][plot_var].array().to_numpy()
            jet_flav = file['Events']['Jet_PartonFlavour'].array().to_numpy()
            jet_eta_mc = file['Events']['Jet_eta'].array().to_numpy()
            jet_pt_mc = file['Events']['Jet_pt'].array().to_numpy()
            dimuon_pt_mc = file['Events']['Dimuon_pt'].array().to_numpy()
            dimuon_eta_mc = file['Events']['Dimuon_eta'].array().to_numpy()
            mc_weight = file['Events']['weight'].array().to_numpy()
            PU_weight = file['Events']['PU_weight'].array().to_numpy()
            L1prefiring_weight = file['Events']['L1prefiring_weight'].array().to_numpy()

        if variable == 'btagDeepFlavQG':
            var_mc = 1 - var_mc

    elif channel == 'dijet':
        plot_var1 = 'Jet1_'+variable
        plot_var2 = 'Jet2_'+variable
        
        jet1_flav = np.array([])
        jet1_var_mc = np.array([])
        jet1_eta_mc = np.array([])
        jet1_pt_mc = np.array([])

        jet2_flav = np.array([])
        jet2_var_mc = np.array([])
        jet2_eta_mc = np.array([])
        jet2_pt_mc = np.array([])

        mc_weight = np.array([])
        PU_weight = np.array([])
        L1prefiring_weight = np.array([])

        ht_list = ['50to100','100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
        for ht_bin in ht_list:
            mc_file = input_file_path.joinpath(f'mc_QCD_HT{ht_bin}')
            print(f'READING MC FILE:    {mc_file.stem}.root')
            with uproot.open(mc_file.with_suffix('.root')) as file:
                jet1_flav = np.append(jet1_flav, file['Events']['Jet1_PartonFlavour'].array().to_numpy())
                jet1_var_mc = np.append(jet1_var_mc, file['Events'][plot_var1].array().to_numpy())
                jet1_eta_mc = np.append(jet1_eta_mc, file['Events']['Jet1_eta'].array().to_numpy())
                jet1_pt_mc = np.append(jet1_pt_mc, file['Events']['Jet1_pt'].array().to_numpy())
                jet2_flav = np.append(jet2_flav, file['Events']['Jet2_PartonFlavour'].array().to_numpy())
                jet2_var_mc = np.append(jet2_var_mc, file['Events'][plot_var2].array().to_numpy())
                jet2_eta_mc = np.append(jet2_eta_mc, file['Events']['Jet2_eta'].array().to_numpy())
                jet2_pt_mc = np.append(jet2_pt_mc, file['Events']['Jet2_pt'].array().to_numpy())
                mc_weight = np.append(mc_weight, file['Events']['weight'].array().to_numpy())
                PU_weight = np.append(PU_weight, file['Events']['PU_weight'].array().to_numpy())
                L1prefiring_weight = np.append(L1prefiring_weight, file['Events']['L1prefiring_weight'].array().to_numpy())

        if variable == 'btagDeepFlavQG':
            jet1_var_mc = 1 - jet1_var_mc
            jet2_var_mc = 1 - jet2_var_mc

    mc_weight = mc_weight*PU_weight*L1prefiring_weight 

    bins_to_process = config['hist_binning_for_ROC_curves']['bins']
    for bin_to_process in bins_to_process:
        eta_bin = bin_to_process['eta'] # Assumes that there is only one eta_bin for each bin to process
        eta_low = eta_bin[0] 
        eta_high = eta_bin[1]
        str_eta_low = str(eta_low).replace('.','_')
        str_eta_high = str(eta_high).replace('.','_')

        pt_bins = bin_to_process['pT']
        for i in range(len(pt_bins)-1):
            pt_low = pt_bins[i]
            pt_high = pt_bins[i+1]

            if inclusive:
                pt_low = 30
                pt_high = 8000
            print(f'PROCESSING BIN:     {eta_low} < |eta| < {eta_high}, {pt_low} GeV < pT < {pt_high} GeV')

            ### Note that the application of the SFs is hardcoded for this particular SF binning!
            if eta_high > 1.3:
                weight_pt_range = '30to8000'
            else:
                if inclusive:
                    weight_pt_range = '30to8000'
                elif pt_low < 80 and pt_high > 80:
                    weight_pt_range = '30to8000'
                elif pt_high <= 80:
                    weight_pt_range = '30to80'
                else:
                    weight_pt_range = '80to8000'

            weight_variable = {
                    'qgl' : 'qgl',
                    'qgl_new' : 'qgl_new',
                    'particleNetAK4_QvsG' : 'particleNetAK4_QvsG',
                    'btagDeepFlavQG': 'btagDeepFlavQG'
                    }

            eta_range = (str(eta_low) + 'to' + str(eta_high)).replace('.','_')
            pt_range = str(pt_low) + 'to' + str(pt_high)

            campaign = config['campaign']
            weight_path = f'{input_sf_path}/sf_weights/{weight_variable[variable]}_weights_{campaign}_eta{str_eta_low}to{str_eta_high}_pt{weight_pt_range}_combined_syst'
            quark_weights_nominal = convert.from_uproot_THx(weight_path+'.root:quark_weights_nominal', flow='clamp').to_evaluator()
            quark_weights_combined_unc_up = convert.from_uproot_THx(weight_path+'.root:quark_weights_combined_unc_up', flow='clamp').to_evaluator()
            quark_weights_combined_unc_down = convert.from_uproot_THx(weight_path+'.root:quark_weights_combined_unc_down', flow='clamp').to_evaluator()
            gluon_weights_nominal = convert.from_uproot_THx(weight_path+'.root:gluon_weights_nominal', flow='clamp').to_evaluator()
            gluon_weights_combined_unc_up = convert.from_uproot_THx(weight_path+'.root:gluon_weights_combined_unc_up', flow='clamp').to_evaluator()
            gluon_weights_combined_unc_down = convert.from_uproot_THx(weight_path+'.root:gluon_weights_combined_unc_down', flow='clamp').to_evaluator()

            if channel == 'zmm':
                mc_cuts = np.all([(np.abs(jet_eta_mc) >= eta_low), (np.abs(jet_eta_mc) < eta_high), (dimuon_pt_mc > pt_low), (dimuon_pt_mc < pt_high)], axis=0)

                jet_var_mc_loop = var_mc[mc_cuts]
                jet_mc_flav_loop = jet_flav[mc_cuts]
                mc_weight_loop = mc_weight[mc_cuts]
            elif channel == 'dijet':
                tag_eta_cut = 1.3

                tag1_pt_mc = jet1_pt_mc
                tag1_eta_mc = jet1_eta_mc
                probe1_eta_mc = jet2_eta_mc
                probe1_var_mc = jet2_var_mc
                probe1_flav = jet2_flav
                probe1_mc_weight = mc_weight

                tag2_pt_mc = jet2_pt_mc
                tag2_eta_mc = jet2_eta_mc
                probe2_eta_mc = jet1_eta_mc
                probe2_var_mc = jet1_var_mc
                probe2_flav = jet1_flav
                probe2_mc_weight = mc_weight

                probe1_mc_cuts = np.all([(np.abs(probe1_eta_mc) >= eta_low), (np.abs(probe1_eta_mc) < eta_high), (np.abs(tag1_eta_mc) < tag_eta_cut), (tag1_pt_mc > pt_low), (tag1_pt_mc < pt_high)], axis=0)
                probe2_mc_cuts = np.all([(np.abs(probe2_eta_mc) >= eta_low), (np.abs(probe2_eta_mc) < eta_high), (np.abs(tag2_eta_mc) < tag_eta_cut), (tag2_pt_mc > pt_low), (tag2_pt_mc < pt_high)], axis=0)

                probe1_flav = probe1_flav[probe1_mc_cuts]
                probe1_var_mc = probe1_var_mc[probe1_mc_cuts]
                probe1_mc_weight = probe1_mc_weight[probe1_mc_cuts]

                probe2_flav = probe2_flav[probe2_mc_cuts]
                probe2_var_mc = probe2_var_mc[probe2_mc_cuts]
                probe2_mc_weight = probe2_mc_weight[probe2_mc_cuts]

                jet_var_mc_loop = np.append(probe1_var_mc, probe2_var_mc)
                jet_mc_flav_loop = np.append(probe1_flav, probe2_flav)
                mc_weight_loop = np.append(probe1_mc_weight, probe2_mc_weight)

            plot_bins = 100

            plot_range = (0., 1.0000000000001)
            xlims = (0., 1.)
            majortick_spacing = 0.1
            minortick_spacing = 0.02

            is_gluon = np.abs(jet_mc_flav_loop) == 21
            is_quark = np.any([(np.abs(jet_mc_flav_loop) == 1), (np.abs(jet_mc_flav_loop) == 2), (np.abs(jet_mc_flav_loop) == 3), (np.abs(jet_mc_flav_loop) == 4), (np.abs(jet_mc_flav_loop) == 5)], axis=0)

            quark_weights = quark_weights_nominal.evaluate(jet_var_mc_loop[is_quark])
            quark_weights_up = quark_weights_nominal.evaluate(jet_var_mc_loop[is_quark]) + quark_weights_combined_unc_up.evaluate(jet_var_mc_loop[is_quark])
            quark_weights_down = quark_weights_nominal.evaluate(jet_var_mc_loop[is_quark]) - quark_weights_combined_unc_down.evaluate(jet_var_mc_loop[is_quark])
            quark_weights_down = [0. if i < 0 else i for i in quark_weights_down]
            gluon_weights = gluon_weights_nominal.evaluate(jet_var_mc_loop[is_gluon])
            gluon_weights_up = gluon_weights_nominal.evaluate(jet_var_mc_loop[is_gluon]) + gluon_weights_combined_unc_up.evaluate(jet_var_mc_loop[is_gluon])
            gluon_weights_down = gluon_weights_nominal.evaluate(jet_var_mc_loop[is_gluon]) - gluon_weights_combined_unc_down.evaluate(jet_var_mc_loop[is_gluon])
            gluon_weights_down = [0. if i < 0 else i for i in gluon_weights_down]

            hist_quark = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
            hist_gluon = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
            hist_quark_syst_up = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
            hist_gluon_syst_up = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
            hist_quark_syst_down = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
            hist_gluon_syst_down = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()

            # Fill the histograms
            hist_quark.fill(jet_var_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*quark_weights)
            hist_gluon.fill(jet_var_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*gluon_weights)
            hist_total_mc = hist_quark + hist_gluon

            hist_quark_syst_up.fill(jet_var_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*quark_weights_up)
            hist_gluon_syst_up.fill(jet_var_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*gluon_weights_up)
            hist_total_mc_syst_up = hist_gluon_syst_up + hist_quark_syst_up 

            hist_quark_syst_down.fill(jet_var_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*quark_weights_down)
            hist_gluon_syst_down.fill(jet_var_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*gluon_weights_down)
            hist_total_mc_syst_down = hist_gluon_syst_down + hist_quark_syst_down

            fig, (ax1) = plt.subplots(figsize=(8,8))

            stack_hists = [hist_gluon, hist_quark]
            stack_label = ['Gluon', 'Quark']
            stack_colors = ['#cc6666', '#6699cc']
            bar_gluon, bar_quark = hep.histplot(stack_hists, histtype='fill', ax=ax1, stack=True, label=stack_label, color=stack_colors, yerr=0)
            hep.histplot(stack_hists, histtype='step', ax=ax1, stack=True, color=np.repeat('black', 2), linewidth=np.repeat(0.8, 2), yerr=0)

            max_y_val = np.max(hist_total_mc.values())
            ax1.set_ylim([0, max_y_val*1.25])
            ax1.set_xlim(xlims)
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
            ax1.xaxis.set_major_locator(MultipleLocator(majortick_spacing))
            ax1.xaxis.set_minor_locator(MultipleLocator(minortick_spacing))
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.tick_params(axis='both', which='both', direction='in')
        
            ylabel = 'Number of probe jets / bin'
            ax1.set_ylabel(ylabel, fontsize=20, loc='top', labelpad=17)

            ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
            new_handles = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', linewidth=0.8, label=label) for color, label in zip(stack_colors, stack_label)]
            
            legend_x = 0.525
            legend_y = 0.797
            legend1 = ax1.legend(loc=(legend_x, legend_y), handles=new_handles, fontsize=16, ncol=2, frameon=False, labelspacing=0.6, handlelength=0.8, handleheight=0.8, handletextpad=0.4, columnspacing=1.0)

            var_label = {
                    'qgl' : 'Quark-Gluon Likelihood',
                    'qgl_new' : 'Quark-Gluon Likelihood',
                    'particleNetAK4_QvsG' : 'ParticleNet discriminant',
                    'btagDeepFlavQG' : 'DeepJet discriminant'
                    }
            ax1.set_xlabel(var_label[variable], fontsize=20, loc='right')

            plt.sca(ax1)
            channel_text = 'Dijet, Pythia' if channel=='dijet' else 'Z+jets, Pythia'
            channel_text_x = 0.615
            channel_text_y = 0.86
            plt.figtext(channel_text_x, channel_text_y, channel_text, fontsize=20, fontweight='semibold')

            selection_text_x = 0.72
            selection_text_y = 0.72

            if eta_low == 0.0:
                eta_text = r'|$\mathit{\eta}^{\mathrm{probe}}$| < ' + str(eta_high) 
            else:
                eta_text = str(eta_low) + r' < |$\mathit{\eta}^{\mathrm{probe}}$| < ' + str(eta_high) 
            if inclusive:
                selection_text = r'$\mathit{p_T}^\mathrm{tag}}$ > ' + str(pt_low) + ' GeV' + '\n' + eta_text
            else:
                selection_text = str(pt_low) + r' < $\mathit{p_T}^{\mathrm{tag}}$ < ' + str(pt_high) + ' GeV' + '\n' + eta_text
            plt.figtext(selection_text_x, selection_text_y, selection_text, fontsize=16, ha='center', ma='center', va='center')

            hep.cms.label('Preliminary', loc=2, data=False, fontsize=20)
            fig.tight_layout()
            plt.subplots_adjust(hspace=0.1)

            plot_save_name = plot_save_path.joinpath(f'{variable}_{channel}_{campaign}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}')

            if save_plot:
                plt.savefig(plot_save_name.with_suffix('.png'))
                plt.savefig(plot_save_name.with_suffix('.pdf'))

            if save_root:
                with uproot.recreate(plot_save_name.with_suffix('.root')) as root_file:
                    root_file['{variable}_{channel}_eta{eta_range}_pt{pt_low}to{pt_high}_quark'] = hist_quark
                    root_file['{variable}_{channel}_eta{eta_range}_pt{pt_low}to{pt_high}_gluon'] = hist_gluon
                    root_file['{variable}_{channel}_eta{eta_range}_pt{pt_low}to{pt_high}_quark_syst_up'] = hist_quark_syst_up
                    root_file['{variable}_{channel}_eta{eta_range}_pt{pt_low}to{pt_high}_gluon_syst_up'] = hist_gluon_syst_up
                    root_file['{variable}_{channel}_eta{eta_range}_pt{pt_low}to{pt_high}_quark_syst_down'] = hist_quark_syst_down
                    root_file['{variable}_{channel}_eta{eta_range}_pt{pt_low}to{pt_high}_gluon_syst_down'] = hist_gluon_syst_down

            if display:
                plt.show()

            if inclusive:
                break

            plt.clf()
            plt.close('all')

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--variable', dest='variable', type='string', default='qgl', help='Variable to plot. (Possible: qgl, particleNet, deepJet)')
    parser.add_option('-c', '--channel', dest='channel', type='string', default='zmm', help='Choose which channel to process. (Possible: zmm, dijet)')
    parser.add_option('-d', '--display', dest='display', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-i', '--inclusive', dest='inclusive', action='store_true', default=False, help='Option to save plot inclusive in pT.')
    parser.add_option('-r', '--root', dest='save_root', action='store_true', default=False, help='Option to save histogram as a ROOT file.')
    parser.add_option('-s', '--save', dest='save_plot', action='store_true', default=False, help='Option to save plot to .png and .pdf files.')
    (opt, args) = parser.parse_args()

    if opt.channel not in ['zmm', 'dijet']:
        sys.exit('ERROR! Invalid channel: {}\nChoose zmm or dijet.'.format(opt.channel))

    if opt.variable.lower() == 'particlenet':
        opt.variable = 'particleNetAK4_QvsG'
    if opt.variable.lower() == 'deepjet':
        opt.variable = 'btagDeepFlavQG'

    main(opt.config_path, opt.variable, opt.display, opt.save_plot, opt.save_root, opt.inclusive, opt.channel)
