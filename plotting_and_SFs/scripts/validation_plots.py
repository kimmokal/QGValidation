import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import tomllib
import uproot
import hist
import sys

from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from optparse import OptionParser

hep.style.use('CMS')
matplotlib.rcParams['font.size'] = 18

def main(config_path, variable, binning, display, save_plot, save_root, inclusive, syst_name, plot_syst_up, plot_syst_down, channel):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n---------------------------')
    print(f'PROCESSING VALIDATION PLOTS')
    print(f'---------------------------')
    print(f'PLOTTING VARIABLE:  {variable}')
    if syst_name == 'nominal':
        print(f'SYSTEMATIC:         nominal')
    else:
        direction = 'up' if plot_syst_up else 'down'
        print(f'SYSTEMATIC:         {syst_name}')
        print(f'VARIATION:          {direction}')

    input_file_path = Path(config['path']['input_samples'])
    print(f'READING FILES FROM: {input_file_path}/')

    if save_plot or save_root:
        output_path = config['path']['output']
        plot_save_path = Path(f'{output_path}/val_plots/{channel}')
        plot_save_path.mkdir(parents=True, exist_ok=True)
        print(f'WRITING OUTPUT TO:  {plot_save_path}/')

    if channel == 'zmm':
        plot_variable = 'Jet_'+variable

        data_file = input_file_path.joinpath('data_DoubleMuon')
        print(f'READING DATA FILE:  {data_file.stem}.root')
        with uproot.open(data_file.with_suffix('.root')) as file:
            plot_variable_data = file['Events'][plot_variable].array().to_numpy()
            jet_eta_data = file['Events']['Jet_eta'].array().to_numpy()
            jet_pt_data = file['Events']['Jet_pt'].array().to_numpy()
            dimuon_pt_data = file['Events']['Dimuon_pt'].array().to_numpy()
            dimuon_eta_data = file['Events']['Dimuon_eta'].array().to_numpy()

        mc_file = input_file_path.joinpath('mc_DYJetsToLL')

        if syst_name == 'jes' or syst_name == 'jer':
            direction = 'up' if plot_syst_up else 'down'
            mc_file = Path(f'{mc_file}_{syst_name}_{direction}')

        print(f'READING MC FILE:    {mc_file.stem}.root')
        with uproot.open(mc_file.with_suffix('.root')) as file:
            plot_variable_mc = file['Events'][plot_variable].array().to_numpy()
            jet_flavour = file['Events']['Jet_PartonFlavour'].array().to_numpy()
            jet_eta_mc = file['Events']['Jet_eta'].array().to_numpy()
            jet_pt_mc = file['Events']['Jet_pt'].array().to_numpy()
            dimuon_pt_mc = file['Events']['Dimuon_pt'].array().to_numpy()
            dimuon_eta_mc = file['Events']['Dimuon_eta'].array().to_numpy()
            mc_weight = file['Events']['weight'].array().to_numpy()
            PU_weight = file['Events']['PU_weight'].array().to_numpy()
            L1prefiring_weight = file['Events']['L1prefiring_weight'].array().to_numpy()

            if syst_name == 'gluon':
                syst_weight_up = file['Events']['gluon_weight'].array().to_numpy()
                syst_weight_down = np.ones_like(syst_weight_up)
                syst_weight_down[syst_weight_up==1.1] = 0.909
                syst_weight_down[syst_weight_up==1.21] = 0.826
            elif syst_name == 'fsr':
                syst_weight_up = file['Events']['FSR_weight_up'].array().to_numpy()
                syst_weight_down = file['Events']['FSR_weight_down'].array().to_numpy()
            elif syst_name == 'isr':
                syst_weight_up = file['Events']['ISR_weight_up'].array().to_numpy()
                syst_weight_down = file['Events']['ISR_weight_down'].array().to_numpy()
            elif syst_name == 'pu':
                syst_weight_up = file['Events']['PU_weight_up'].array().to_numpy()
                syst_weight_down = file['Events']['PU_weight_down'].array().to_numpy()
                PU_weight = np.ones_like(mc_weight)

        if variable == 'qgl_axis2':
            plot_variable_data = -np.log(plot_variable_data, out=np.zeros_like(plot_variable_data), where=(plot_variable_data!=0))
            plot_variable_mc = -np.log(plot_variable_mc, out=np.zeros_like(plot_variable_mc), where=(plot_variable_mc!=0))

        # Invert DeepJet output so that it matches QGL and ParticleNet
        if variable == 'btagDeepFlavQG':
            plot_variable_data = 1 - plot_variable_data
            plot_variable_mc = 1 - plot_variable_mc

    elif channel == 'dijet':
        plot_variable1 = 'Jet1_'+variable
        plot_variable2 = 'Jet2_'+variable
        
        data_file = input_file_path.joinpath('data_ZeroBias')
        print(f'READING DATA FILE:  {data_file.stem}.root')
        with uproot.open(data_file.with_suffix('.root')) as file:
            jet1_variable_data = file['Events'][plot_variable1].array().to_numpy()
            jet1_eta_data = file['Events']['Jet1_eta'].array().to_numpy()
            jet1_pt_data = file['Events']['Jet1_pt'].array().to_numpy()
            jet2_variable_data = file['Events'][plot_variable2].array().to_numpy()
            jet2_eta_data = file['Events']['Jet2_eta'].array().to_numpy()
            jet2_pt_data = file['Events']['Jet2_pt'].array().to_numpy()

        jet1_flavour = np.array([])
        jet1_variable_mc = np.array([])
        jet1_eta_mc = np.array([])
        jet1_pt_mc = np.array([])

        jet2_flavour = np.array([])
        jet2_variable_mc = np.array([])
        jet2_eta_mc = np.array([])
        jet2_pt_mc = np.array([])

        mc_weight = np.array([])
        PU_weight = np.array([])
        L1prefiring_weight = np.array([])
        syst_weight_up = np.array([])
        syst_weight_down = np.array([])

        ht_list = ['50to100','100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
        for ht_bin in ht_list:
            mc_file = input_file_path.joinpath(f'mc_QCD_HT{ht_bin}')
            if syst_name == 'jes' or syst_name == 'jer':
                direction = 'up' if plot_syst_up else 'down'
                mc_file = Path(f'{mc_file}_{syst_name}_{direction}')
            print(f'READING MC FILE:    {mc_file.stem}.root')

            with uproot.open(mc_file.with_suffix('.root')) as file:
                jet1_flavour = np.append(jet1_flavour, file['Events']['Jet1_PartonFlavour'].array().to_numpy())
                jet1_variable_mc = np.append(jet1_variable_mc, file['Events'][plot_variable1].array().to_numpy())
                jet1_eta_mc = np.append(jet1_eta_mc, file['Events']['Jet1_eta'].array().to_numpy())
                jet1_pt_mc = np.append(jet1_pt_mc, file['Events']['Jet1_pt'].array().to_numpy())
                jet2_flavour = np.append(jet2_flavour, file['Events']['Jet2_PartonFlavour'].array().to_numpy())
                jet2_variable_mc = np.append(jet2_variable_mc, file['Events'][plot_variable2].array().to_numpy())
                jet2_eta_mc = np.append(jet2_eta_mc, file['Events']['Jet2_eta'].array().to_numpy())
                jet2_pt_mc = np.append(jet2_pt_mc, file['Events']['Jet2_pt'].array().to_numpy())
                mc_weight = np.append(mc_weight, file['Events']['weight'].array().to_numpy())
                L1prefiring_weight = np.append(L1prefiring_weight, file['Events']['L1prefiring_weight'].array().to_numpy())

                if syst_name == 'pu':
                    PU_weight = np.append(PU_weight, np.ones_like(file['Events']['weight'].array().to_numpy()))
                    syst_weight_up = np.append(syst_weight_up, file['Events']['PU_weight_up'].array().to_numpy())
                    syst_weight_down = np.append(syst_weight_down, file['Events']['PU_weight_down'].array().to_numpy())
                else:
                    PU_weight = np.append(PU_weight, file['Events']['PU_weight'].array().to_numpy())

                if syst_name == 'gluon':
                    loop_syst_weight_up = file['Events']['gluon_weight'].array().to_numpy()
                    loop_syst_weight_down = np.ones_like(loop_syst_weight_up)
                    loop_syst_weight_down[loop_syst_weight_up==1.1] = 0.909
                    loop_syst_weight_down[loop_syst_weight_up==1.21] = 0.826

                    syst_weight_up = np.append(syst_weight_up, loop_syst_weight_up)
                    syst_weight_down = np.append(syst_weight_down, loop_syst_weight_down)
                elif syst_name == 'fsr':
                    syst_weight_up = np.append(syst_weight_up, file['Events']['FSR_weight_up'].array().to_numpy())
                    syst_weight_down = np.append(syst_weight_down, file['Events']['FSR_weight_down'].array().to_numpy())
                elif syst_name == 'isr':
                    syst_weight_up = np.append(syst_weight_up, file['Events']['ISR_weight_up'].array().to_numpy())
                    syst_weight_down = np.append(syst_weight_down, file['Events']['ISR_weight_down'].array().to_numpy())

        if variable == 'qgl_axis2':
            jet1_variable_data = -np.log(jet1_variable_data, out=np.zeros_like(jet1_variable_data), where=(jet1_variable_data!=0))
            jet1_variable_mc = -np.log(jet1_variable_mc, out=np.zeros_like(jet1_variable_mc), where=(jet1_variable_mc!=0))
            jet2_variable_data = -np.log(jet2_variable_data, out=np.zeros_like(jet2_variable_data), where=(jet2_variable_data!=0))
            jet2_variable_mc = -np.log(jet2_variable_mc, out=np.zeros_like(jet2_variable_mc), where=(jet2_variable_mc!=0))

        if variable == 'btagDeepFlavQG':
            jet1_variable_data = 1 - jet1_variable_data
            jet2_variable_data = 1 - jet2_variable_data
            jet1_variable_mc = 1 - jet1_variable_mc
            jet2_variable_mc = 1 - jet2_variable_mc

    mc_weight = mc_weight*PU_weight*L1prefiring_weight


    if binning == 'validation':
        bins_to_process = config['validation_binning']['bins']
    else:
        bins_to_process = config['SF_binning']['bins']

    for bin_to_process in bins_to_process:
        eta_bin = bin_to_process['eta'] # Assumes that there is only one eta_bin for each bin to process
        eta_low = eta_bin[0] 
        eta_high = eta_bin[1]

        pt_bins = bin_to_process['pT']
        for i in range(len(pt_bins)-1):
            pt_low = pt_bins[i]
            pt_high = pt_bins[i+1]

            if inclusive:
                pt_low = 30
                pt_high = 8000

            print(f'PROCESSING BIN:     {eta_low} < |eta| < {eta_high}, {pt_low} GeV < pT < {pt_high} GeV')

            if channel == 'zmm':
                data_cuts = np.all([(np.abs(jet_eta_data) >= eta_low), (np.abs(jet_eta_data) < eta_high), (dimuon_pt_data > pt_low), (dimuon_pt_data < pt_high)], axis=0)
                mc_cuts = np.all([(np.abs(jet_eta_mc) >= eta_low), (np.abs(jet_eta_mc) < eta_high), (dimuon_pt_mc > pt_low), (dimuon_pt_mc < pt_high)], axis=0)
            
                jet_variable_data_loop = plot_variable_data[data_cuts]
                jet_variable_mc_loop = plot_variable_mc[mc_cuts]
                jet_flavour_loop = jet_flavour[mc_cuts]
                mc_weight_loop = mc_weight[mc_cuts]

                if syst_name not in ['nominal', 'jer', 'jes']:
                    syst_weight_up_loop = syst_weight_up[mc_cuts]
                    syst_weight_down_loop = syst_weight_down[mc_cuts]

            elif channel == 'dijet':
                tag_eta_cut = 1.3

                tag1_pt_data = jet1_pt_data
                tag1_eta_data = jet1_eta_data
                probe1_eta_data = jet2_eta_data
                probe1_variable_data = jet2_variable_data

                tag2_pt_data = jet2_pt_data
                tag2_eta_data = jet2_eta_data
                probe2_eta_data = jet1_eta_data
                probe2_variable_data = jet1_variable_data

                tag1_pt_mc = jet1_pt_mc
                tag1_eta_mc = jet1_eta_mc
                probe1_eta_mc = jet2_eta_mc
                probe1_variable_mc = jet2_variable_mc
                probe1_flavour = jet2_flavour
                probe1_mc_weight = mc_weight

                tag2_pt_mc = jet2_pt_mc
                tag2_eta_mc = jet2_eta_mc
                probe2_eta_mc = jet1_eta_mc
                probe2_variable_mc = jet1_variable_mc
                probe2_flavour = jet1_flavour
                probe2_mc_weight = mc_weight

                if syst_name not in ['nominal', 'jer', 'jes']:
                    probe1_syst_weight_up = syst_weight_up
                    probe1_syst_weight_down = syst_weight_down
                    probe2_syst_weight_up = syst_weight_up
                    probe2_syst_weight_down = syst_weight_down

                probe1_data_cuts = np.all([(np.abs(probe1_eta_data) >= eta_low), (np.abs(probe1_eta_data) < eta_high), (np.abs(tag1_eta_data) < tag_eta_cut), (tag1_pt_data > pt_low), (tag1_pt_data < pt_high)], axis=0)
                probe2_data_cuts = np.all([(np.abs(probe2_eta_data) >= eta_low), (np.abs(probe2_eta_data) < eta_high), (np.abs(tag2_eta_data) < tag_eta_cut), (tag2_pt_data > pt_low), (tag2_pt_data < pt_high)], axis=0)
                probe1_mc_cuts = np.all([(np.abs(probe1_eta_mc) >= eta_low), (np.abs(probe1_eta_mc) < eta_high), (np.abs(tag1_eta_mc) < tag_eta_cut), (tag1_pt_mc > pt_low), (tag1_pt_mc < pt_high)], axis=0)
                probe2_mc_cuts = np.all([(np.abs(probe2_eta_mc) >= eta_low), (np.abs(probe2_eta_mc) < eta_high), (np.abs(tag2_eta_mc) < tag_eta_cut), (tag2_pt_mc > pt_low), (tag2_pt_mc < pt_high)], axis=0)

                probe1_variable_data = probe1_variable_data[probe1_data_cuts]
                probe2_variable_data = probe2_variable_data[probe2_data_cuts]

                probe1_flavour = probe1_flavour[probe1_mc_cuts]
                probe1_variable_mc = probe1_variable_mc[probe1_mc_cuts]
                probe1_mc_weight = probe1_mc_weight[probe1_mc_cuts]

                probe2_flavour = probe2_flavour[probe2_mc_cuts]
                probe2_variable_mc = probe2_variable_mc[probe2_mc_cuts]
                probe2_mc_weight = probe2_mc_weight[probe2_mc_cuts]

                jet_variable_data_loop = np.append(probe1_variable_data, probe2_variable_data)
                jet_variable_mc_loop = np.append(probe1_variable_mc, probe2_variable_mc)
                jet_flavour_loop = np.append(probe1_flavour, probe2_flavour)
                mc_weight_loop = np.append(probe1_mc_weight, probe2_mc_weight)

                if syst_name not in ['nominal', 'jer', 'jes']:
                    probe1_syst_weight_up = probe1_syst_weight_up[probe1_mc_cuts]
                    probe1_syst_weight_down = probe1_syst_weight_down[probe1_mc_cuts]
                    probe2_syst_weight_up = probe2_syst_weight_up[probe2_mc_cuts]
                    probe2_syst_weight_down = probe2_syst_weight_down[probe2_mc_cuts]
                    syst_weight_up_loop = np.append(probe1_syst_weight_up, probe2_syst_weight_down)
                    syst_weight_down_loop = np.append(probe1_syst_weight_down, probe2_syst_weight_down)
        
            if eta_high == 4.7:
                plot_bins = 30
            else:
                plot_bins = 50

            plot_range = (0., 1.0000000000001)
            xlims = (0., 1.)
            majortick_spacing = 0.1
            minortick_spacing = 0.02

            if variable == 'particleNetAK4_QvsG':
                plot_bins = 25
            if variable == 'btagDeepFlavQG':
                plot_bins = 25
            if variable == 'qgl_mult':
                plot_bins = 40
                plot_range = (0., 40.0000000000001)
                xlims = (0., 40.)
                majortick_spacing = 5
                minortick_spacing = 1
            elif variable == 'qgl_axis2':
                plot_bins = 40
                plot_range = (1., 7.)
                xlims = (1., 7.)
                majortick_spacing = 0.5
                minortick_spacing = 0.1
            elif variable == 'qgl_ptD':
                plot_bins = 20
            elif variable == 'pt':
                plot_bins = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100, 110, 120, 150, 200, 250, 300, 400, 550, 700]
                plot_range = (pt_low, pt_high)
                xlims = (pt_low, pt_high)
                majortick_spacing = 100
                minortick_spacing = 25
            elif variable == 'eta':
                plot_bins = 30
                plot_range = (-tag_eta_cut-1.0, tag_eta_cut+1.0)
                xlims = (-tag_eta_cut-1.0, tag_eta_cut+1.0)
                majortick_spacing = 0.2
                minortick_spacing = 0.1

            is_gluon = np.abs(jet_flavour_loop) == 21
            is_quark = np.logical_and((np.abs(jet_flavour_loop) >= 1), (np.abs(jet_flavour_loop) <= 5))
            is_undef = np.invert(np.any([is_quark, is_gluon], axis=0))

            if variable == 'particleNetAK4_QvsG':
                variable_binning = list(np.linspace(0, 1, plot_bins+1))
                variable_binning = variable_binning[2:plot_bins-3]
                variable_binning.insert(0, 0.)
                variable_binning.append(1.0)

                hist_data = hist.Hist.new.Variable(variable_binning).Weight()
                hist_quark = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef = hist.Hist.new.Variable(variable_binning).Weight()

                hist_quark_syst_up = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon_syst_up = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef_syst_up = hist.Hist.new.Variable(variable_binning).Weight()

                hist_quark_syst_down = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon_syst_down = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef_syst_down = hist.Hist.new.Variable(variable_binning).Weight()
            elif variable == 'btagDeepFlavQG':
                variable_binning = list(np.linspace(0, 1, plot_bins+1))
                variable_binning = variable_binning[2:plot_bins-4]
                variable_binning.insert(0, 0.)
                variable_binning.append(1.0)

                hist_data = hist.Hist.new.Variable(variable_binning).Weight()
                hist_quark = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef = hist.Hist.new.Variable(variable_binning).Weight()

                hist_quark_syst_up = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon_syst_up = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef_syst_up = hist.Hist.new.Variable(variable_binning).Weight()

                hist_quark_syst_down = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon_syst_down = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef_syst_down = hist.Hist.new.Variable(variable_binning).Weight()
            else:
                hist_data = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_quark = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_gluon = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_undef = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()

                hist_quark_syst_up = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_gluon_syst_up = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_undef_syst_up = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()

                hist_quark_syst_down = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_gluon_syst_down = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()
                hist_undef_syst_down = hist.Hist.new.Regular(plot_bins, plot_range[0], plot_range[1]).Weight()

            hist_data.fill(jet_variable_data_loop)
            hist_quark.fill(jet_variable_mc_loop[is_quark], weight=mc_weight_loop[is_quark])
            hist_gluon.fill(jet_variable_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon])
            hist_undef.fill(jet_variable_mc_loop[is_undef], weight=mc_weight_loop[is_undef])
            hist_total_mc = hist_gluon + hist_quark + hist_undef

            mc_data_norm = np.sum(hist_data.values())/np.sum(hist_total_mc.values())
            hist_total_mc *= mc_data_norm
            hist_quark *= mc_data_norm
            hist_gluon *= mc_data_norm
            hist_undef *= mc_data_norm

            if syst_name not in ['nominal', 'jer', 'jes']:
                hist_quark_syst_up.fill(jet_variable_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*syst_weight_up_loop[is_quark])
                hist_gluon_syst_up.fill(jet_variable_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*syst_weight_up_loop[is_gluon])
                hist_undef_syst_up.fill(jet_variable_mc_loop[is_undef], weight=mc_weight_loop[is_undef]*syst_weight_up_loop[is_undef])
                hist_total_mc_syst_up = hist_gluon_syst_up + hist_quark_syst_up + hist_undef_syst_up

                hist_quark_syst_down.fill(jet_variable_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*syst_weight_down_loop[is_quark])
                hist_gluon_syst_down.fill(jet_variable_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*syst_weight_down_loop[is_gluon])
                hist_undef_syst_down.fill(jet_variable_mc_loop[is_undef], weight=mc_weight_loop[is_undef]*syst_weight_down_loop[is_undef])
                hist_total_mc_syst_down = hist_gluon_syst_down + hist_quark_syst_down + hist_undef_syst_down

                mc_data_norm_syst_up = np.sum(hist_data.values())/np.sum(hist_total_mc_syst_up.values())
                hist_total_mc_syst_up *= mc_data_norm_syst_up
                hist_quark_syst_up *= mc_data_norm_syst_up
                hist_gluon_syst_up *= mc_data_norm_syst_up
                hist_undef_syst_up *= mc_data_norm_syst_up

                mc_data_norm_syst_down = np.sum(hist_data.values())/np.sum(hist_total_mc_syst_down.values())
                hist_total_mc_syst_down *= mc_data_norm_syst_down
                hist_quark_syst_down *= mc_data_norm_syst_down
                hist_gluon_syst_down *= mc_data_norm_syst_down
                hist_undef_syst_down *= mc_data_norm_syst_down

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8,8), gridspec_kw={'height_ratios':[1,.3]})

            if syst_name in ['nominal', 'jer', 'jes']:
                plot_hist_undef = hist_undef
                plot_hist_gluon = hist_gluon
                plot_hist_quark = hist_quark
                plot_hist_total_mc = hist_total_mc
                hist_ratio_scale = np.divide(np.ones_like(hist_data.values()), hist_total_mc.values(), out=np.zeros_like(hist_data.values()), where=(hist_total_mc.values()!=0.))
            elif plot_syst_up:
                plot_hist_undef = hist_undef_syst_up
                plot_hist_gluon = hist_gluon_syst_up
                plot_hist_quark = hist_quark_syst_up
                plot_hist_total_mc = hist_total_mc_syst_up
                hist_ratio_scale = np.divide(np.ones_like(hist_data.values()), hist_total_mc_syst_up.values(), out=np.zeros_like(hist_data.values()), where=(hist_total_mc_syst_up.values()!=0.))
            elif plot_syst_down:
                plot_hist_undef = hist_undef_syst_down
                plot_hist_gluon = hist_gluon_syst_down
                plot_hist_quark = hist_quark_syst_down
                plot_hist_total_mc = hist_total_mc_syst_down
                hist_ratio_scale = np.divide(np.ones_like(hist_data.values()), hist_total_mc_syst_down.values(), out=np.zeros_like(hist_data.values()), where=(hist_total_mc_syst_down.values()!=0.))

            hist_ratio = hist_data*hist_ratio_scale

            stack_hists = [plot_hist_undef, plot_hist_gluon, plot_hist_quark]
            stack_label = ['Undefined', 'Gluon', 'Quark']
            stack_colors = ['#cccccc', '#cc6666', '#6699cc']
            bar_undef, bar_gluon, bar_quark = hep.histplot(stack_hists, histtype='fill', ax=ax1, stack=True, label=stack_label, color=stack_colors, yerr=0, flow='none')
            hep.histplot(stack_hists, histtype='step', ax=ax1, stack=True, color=np.repeat('black', 3), linewidth=np.repeat(0.8, 3), yerr=0, flow='none')
            bar_data = hep.histplot(hist_data, histtype='errorbar', ax=ax1, xerr=True, yerr=np.sqrt(hist_data.variances()), label='Data', color='black', flow='none')

            bin_centers = hist_total_mc.axes.centers[0]
            bin_widths = hist_total_mc.axes.widths[0]/2

            max_y_val = np.max(np.concatenate((hist_data.values(), hist_total_mc.values())))
            ax1.set_ylim([0, max_y_val*1.25])
            ax1.set_xlim(xlims)
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
            ax1.xaxis.set_major_locator(MultipleLocator(majortick_spacing))
            ax1.xaxis.set_minor_locator(MultipleLocator(minortick_spacing))
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.tick_params(axis='both', which='both', direction='in')
            if variable == 'pt':
                ax1.set_yscale('log')
                ax1.set_ylim([1, 10**7])
                ax1.yaxis.set_minor_locator(AutoMinorLocator())
                ax1.set_xscale('log')
                ax1.tick_params(axis='both', which='both', direction='in')
                ax1.set_yscale('log')
                ax1.set_xticks([])
        
            ylabel = 'Number of probe jets / bin'
            ax1.set_ylabel(ylabel, fontsize=20, loc='top', labelpad=17)

            # Main legend
            ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
            new_handles = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', linewidth=0.8, label=label) for color, label in zip(stack_colors, stack_label)]
            new_handles.append(ax1_handles[3])
            
            legend_x = 0.495
            legend_y = 0.727
            legend1 = ax1.legend(loc=(legend_x, legend_y), handles=new_handles, fontsize=16, ncol=2, frameon=False, labelspacing=0.6, handlelength=0.8, handleheight=0.8, handletextpad=0.4, columnspacing=1.0)

            bar_ratio = hep.histplot(hist_ratio, histtype='errorbar', ax=ax2, xerr=True, color='Black', flow='none')

            ax2.set_xlim(xlims)
            ax2.set_ylim(0.5, 1.5)
            ax2.tick_params(axis='both', which='both', direction='in')
            ax2.tick_params(axis='x', which='both', pad=4)
            ax2.tick_params(axis='y', which='both', pad=6)
            ax2.yaxis.set_major_locator(MultipleLocator(0.2))
            ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax2.xaxis.set_major_locator(MultipleLocator(majortick_spacing))
            ax2.xaxis.set_minor_locator(MultipleLocator(minortick_spacing))
            ax2.axhline(0.8, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
            ax2.axhline(1.0, color='black', linewidth=0.8)
            ax2.axhline(1.2, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
            ax2.set_ylabel(r'$\frac{Data}{Simulation}$', fontsize=24, loc='center', labelpad=18)
        
            var_label = {
                    'qgl' : 'Quark-Gluon Likelihood',
                    'qgl_new' : 'Quark-Gluon Likelihood',
                    'qgl_axis2' : r'-log($\sigma_2$)',
                    'qgl_ptD' : r'$p_T D$',
                    'qgl_mult' : 'Jet constituent multiplicity',
                    'particleNetAK4_QvsG' : 'ParticleNet discriminant',
                    'btagDeepFlavQG' : 'DeepJet discriminant',
                    'pt' : r'$p_T$',
                    'eta' : r'$\eta$'
                    }
            ax2.set_xlabel(var_label[variable], fontsize=20, loc='right')
            
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

            hep.cms.label('Preliminary', loc=1, data=True, lumi='41.5', fontsize=20)
            fig.tight_layout()
            plt.subplots_adjust(hspace=0.1)

            str_eta_low = str(eta_low).replace('.','_')
            str_eta_high = str(eta_high).replace('.','_')
            campaign = config['campaign']
            plot_save_name = plot_save_path.joinpath(f'{variable}_{channel}_{campaign}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}')

            if save_plot:
                if plot_syst_up:
                    plot_save_name = Path(f'{plot_save_name}_{syst_name}_syst_up')
                if plot_syst_down:
                    plot_save_name = Path(f'{plot_save_name}_{syst_name}_syst_down')

                plt.savefig(plot_save_name.with_suffix('.png'))
                plt.savefig(plot_save_name.with_suffix('.pdf'))

            if save_root:
                with uproot.recreate(plot_save_name.with_suffix('.root')) as root_file:
                    root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_data'] = hist_data
                    if syst_name in ['nominal', 'jer', 'jes']:
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_quark'] = hist_quark
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_gluon'] = hist_gluon
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_undef'] = hist_undef
                    elif plot_syst_up:
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_quark'] = hist_quark_syst_up
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_gluon'] = hist_gluon_syst_up
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_undef'] = hist_undef_syst_up
                    elif plot_syst_down:
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_quark'] = hist_quark_syst_down
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_gluon'] = hist_gluon_syst_down
                        root_file[f'{variable}_{channel}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_undef'] = hist_undef_syst_down

            if display:
                plt.show()

            if inclusive:
                break

            plt.clf()
            plt.close('all')

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--config', dest='config_path', type='string', default='', help='Path to config file.')
    parser.add_option('-v', '--variable', dest='variable', type='string', default='qgl', help='Variable to plot. (Possible: qgl, qgl_new, axis2, ptD, mult, particleNet, deepJet, pt, eta)')
    parser.add_option('-c', '--channel', dest='channel', type='string', default='zmm', help='Choose which channel to process. (Possible: zmm, dijet)')
    parser.add_option('-i', '--inclusive', dest='inclusive', action='store_true', default=False, help='Option to save plot inclusive in pT.')
    parser.add_option('-d', '--display', dest='display', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-r', '--root', dest='save_root', action='store_true', default=False, help='Option to save histogram as a ROOT file.')
    parser.add_option('-s', '--save', dest='save_plot', action='store_true', default=False, help='Option to save plot to .png and .pdf files.')
    parser.add_option('--binning', dest='binning', type='string', default='validation', help='Choose binning: "validation" or "scalefactors"')
    parser.add_option('--syst', dest='syst_name', type='string', default='nominal', help='Systematic to vary: nominal, gluon, FSR, ISR, PU, JES, JER.')
    parser.add_option('--syst_up', dest='syst_up', action='store_true', default=False, help='Plot systematic variation up.')
    parser.add_option('--syst_down', dest='syst_down', action='store_true', default=False, help='Plot systematic variation down.')
    (opt, args) = parser.parse_args()

    if opt.channel not in ['zmm', 'dijet']:
        sys.exit(f'ERROR! Invalid channel: {opt.channel}\nChoose: zmm or dijet')

    if opt.binning.lower() not in ['validation', 'scalefactors']:
        sys.exit(f'ERROR! Choose binning type: validation, scalefactors')

    if opt.syst_name.lower() not in ['nominal', 'gluon', 'fsr', 'isr', 'pu', 'jes', 'jer']:
        sys.exit(f'ERROR! Invalid systematic: {opt.syst_name}\nChoose: nominal, gluon, FSR, ISR, PU, JES, JER')

    if opt.syst_name.lower() != 'nominal' and opt.syst_up == False and opt.syst_down == False:
        sys.exit('ERROR! Up or down variation required with option --syst_up or --syst_down.')

    if opt.syst_up and opt.syst_down:
        sys.exit('ERROR! Cannot plot up and down systematic variations simultaneously. Only choose one or none.')
    
    if opt.variable.lower() == 'particlenet':
        opt.variable = 'particleNetAK4_QvsG'
    if opt.variable.lower() == 'deepjet':
        opt.variable = 'btagDeepFlavQG'
    if opt.variable.lower() == 'axis2':
        opt.variable = 'qgl_axis2'
    if opt.variable.lower() == 'mult':
        opt.variable = 'qgl_mult'
    if opt.variable.lower() == 'ptd':
        opt.variable = 'qgl_ptD'

    main(opt.config_path, opt.variable, opt.binning.lower(), opt.display, opt.save_plot, opt.save_root, opt.inclusive, opt.syst_name.lower(), opt.syst_up, opt.syst_down, opt.channel)
