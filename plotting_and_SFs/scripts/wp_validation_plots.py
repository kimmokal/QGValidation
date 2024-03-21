import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib
import uproot
import hist
import json
import sys
import os

from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from optparse import OptionParser

hep.style.use('CMS')
matplotlib.rcParams['font.size'] = 18

def main(config_path, variable, working_point, display, save_plot, save_root, inclusive, syst_name, plot_syst_up, plot_syst_down, channel):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n----------------------------------------------')
    print(f'PROCESSING VALIDATION PLOTS FOR WORKING POINTS')
    print(f'----------------------------------------------')
    print(f'PLOTTING VARIABLE:  {variable}')
    print(f'WORKING POINT:      {working_point}')
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

        if variable == 'btagDeepFlavQG':
            jet1_variable_data = 1 - jet1_variable_data
            jet2_variable_data = 1 - jet2_variable_data
            jet1_variable_mc = 1 - jet1_variable_mc
            jet2_variable_mc = 1 - jet2_variable_mc

    mc_weight = mc_weight*PU_weight*L1prefiring_weight

    working_points = {
            'qgl':{
                'loose':config['qgl_WP']['loose'],
                'medium':config['qgl_WP']['medium'],
                'tight':config['qgl_WP']['tight']},
            'btagDeepFlavQG':{
                'loose':config['deepjet_WP']['loose'],
                'medium':config['deepjet_WP']['medium'],
                'tight':config['deepjet_WP']['tight']},
            'particleNetAK4_QvsG':{
                'loose':config['particlenet_WP']['loose'],
                'medium':config['particlenet_WP']['medium'],
                'tight':config['particlenet_WP']['tight']},
            }
    wp_cutoff = working_points[variable][working_point]

    wp_bins = config['WP_binning']['bins']
    for wp_bin in wp_bins:
        eta_bin = wp_bin['eta'] # Assumes that there is only one eta_bin for each sf_bin
        eta_range = f'{eta_bin[0]}to{eta_bin[1]}'.replace('.','_')

        pt_bins = wp_bin['pT']
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

            plot_range = (0., 1.0000000000001)
            xlims = (0., 1.)
            majortick_spacing = 0.1
            minortick_spacing = 0.02

            is_gluon = np.abs(jet_flavour_loop) == 21
            is_quark = np.logical_and((np.abs(jet_flavour_loop) >= 1), (np.abs(jet_flavour_loop) <= 5))
            is_undef = np.invert(np.any([is_quark, is_gluon], axis=0))

            variable_binning = [0., wp_cutoff, 1.]

            hist_data_temp = hist.Hist.new.Variable(variable_binning).Weight()
            hist_quark_temp = hist.Hist.new.Variable(variable_binning).Weight()
            hist_gluon_temp = hist.Hist.new.Variable(variable_binning).Weight()
            hist_undef_temp = hist.Hist.new.Variable(variable_binning).Weight()

            hist_data_temp.fill(jet_variable_data_loop)
            hist_quark_temp.fill(jet_variable_mc_loop[is_quark], weight=mc_weight_loop[is_quark])
            hist_gluon_temp.fill(jet_variable_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon])
            hist_undef_temp.fill(jet_variable_mc_loop[is_undef], weight=mc_weight_loop[is_undef])
            hist_total_mc_temp = hist_gluon_temp + hist_quark_temp + hist_undef_temp

            mc_data_norm = np.sum(hist_data_temp.values())/np.sum(hist_total_mc_temp.values())
            hist_total_mc_temp *= mc_data_norm
            hist_quark_temp *= mc_data_norm
            hist_gluon_temp *= mc_data_norm
            hist_undef_temp *= mc_data_norm

            # Create new hists with consistent binning where the middle value, i.e. WP cutoff point, is at 0.5
            hist_data = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
            hist_quark = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
            hist_gluon = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
            hist_undef = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()

            if syst_name not in ['nominal', 'jer', 'jes']:
                hist_quark_syst_up_temp = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon_syst_up_temp = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef_syst_up_temp = hist.Hist.new.Variable(variable_binning).Weight()

                hist_quark_syst_down_temp = hist.Hist.new.Variable(variable_binning).Weight()
                hist_gluon_syst_down_temp = hist.Hist.new.Variable(variable_binning).Weight()
                hist_undef_syst_down_temp = hist.Hist.new.Variable(variable_binning).Weight()

                hist_quark_syst_up_temp.fill(jet_variable_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*syst_weight_up_loop[is_quark])
                hist_gluon_syst_up_temp.fill(jet_variable_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*syst_weight_up_loop[is_gluon])
                hist_undef_syst_up_temp.fill(jet_variable_mc_loop[is_undef], weight=mc_weight_loop[is_undef]*syst_weight_up_loop[is_undef])
                hist_total_mc_syst_up_temp = hist_gluon_syst_up_temp + hist_quark_syst_up_temp + hist_undef_syst_up_temp

                hist_quark_syst_down_temp.fill(jet_variable_mc_loop[is_quark], weight=mc_weight_loop[is_quark]*syst_weight_down_loop[is_quark])
                hist_gluon_syst_down_temp.fill(jet_variable_mc_loop[is_gluon], weight=mc_weight_loop[is_gluon]*syst_weight_down_loop[is_gluon])
                hist_undef_syst_down_temp.fill(jet_variable_mc_loop[is_undef], weight=mc_weight_loop[is_undef]*syst_weight_down_loop[is_undef])
                hist_total_mc_syst_down_temp = hist_gluon_syst_down_temp + hist_quark_syst_down_temp + hist_undef_syst_down_temp

                mc_data_norm_syst_up = np.sum(hist_data_temp.values())/np.sum(hist_total_mc_syst_up_temp.values())
                hist_total_mc_syst_up_temp *= mc_data_norm_syst_up
                hist_quark_syst_up_temp *= mc_data_norm_syst_up
                hist_gluon_syst_up_temp *= mc_data_norm_syst_up
                hist_undef_syst_up_temp *= mc_data_norm_syst_up

                mc_data_norm_syst_down = np.sum(hist_data_temp.values())/np.sum(hist_total_mc_syst_down_temp.values())
                hist_total_mc_syst_down_temp *= mc_data_norm_syst_down
                hist_quark_syst_down_temp *= mc_data_norm_syst_down
                hist_gluon_syst_down_temp *= mc_data_norm_syst_down
                hist_undef_syst_down_temp *= mc_data_norm_syst_down

                hist_data_syst_up = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
                hist_quark_syst_up = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
                hist_gluon_syst_up = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
                hist_undef_syst_up = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()

                hist_data_syst_down = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
                hist_quark_syst_down = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
                hist_gluon_syst_down = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()
                hist_undef_syst_down = hist.Hist.new.Variable([0.0, 0.5, 1.0]).Weight()

            for k in [0,1]:
                hist_data.values()[k] = hist_data_temp.values()[k]
                hist_data.variances()[k] = hist_data_temp.variances()[k]
                hist_quark.values()[k] = hist_quark_temp.values()[k]
                hist_quark.variances()[k] = hist_quark_temp.variances()[k]
                hist_gluon.values()[k] = hist_gluon_temp.values()[k]
                hist_gluon.variances()[k] = hist_gluon_temp.variances()[k]
                hist_undef.values()[k] = hist_undef_temp.values()[k]
                hist_undef.variances()[k] = hist_undef_temp.variances()[k]

                if syst_name not in ['nominal', 'jer', 'jes']:
                    hist_quark_syst_up.values()[k] = hist_quark_syst_up_temp.values()[k]
                    hist_quark_syst_up.variances()[k] = hist_quark_syst_up_temp.variances()[k]
                    hist_gluon_syst_up.values()[k] = hist_gluon_syst_up_temp.values()[k]
                    hist_gluon_syst_up.variances()[k] = hist_gluon_syst_up_temp.variances()[k]
                    hist_undef_syst_up.values()[k] = hist_undef_syst_up_temp.values()[k]
                    hist_undef_syst_up.variances()[k] = hist_undef_syst_up_temp.variances()[k]

                    hist_quark_syst_down.values()[k] = hist_quark_syst_down_temp.values()[k]
                    hist_quark_syst_down.variances()[k] = hist_quark_syst_down_temp.variances()[k]
                    hist_gluon_syst_down.values()[k] = hist_gluon_syst_down_temp.values()[k]
                    hist_gluon_syst_down.variances()[k] = hist_gluon_syst_down_temp.variances()[k]
                    hist_undef_syst_down.values()[k] = hist_undef_syst_down_temp.values()[k]
                    hist_undef_syst_down.variances()[k] = hist_undef_syst_down_temp.variances()[k]

            hist_total_mc = hist_quark + hist_gluon + hist_undef
            if syst_name not in ['nominal', 'jer', 'jes']:
                hist_total_mc_syst_up = hist_quark_syst_up + hist_gluon_syst_up + hist_undef_syst_up
                hist_total_mc_syst_down = hist_quark_syst_down + hist_gluon_syst_down + hist_undef_syst_down

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

            max_y_val = np.max(np.concatenate((hist_data.values(), hist_total_mc.values())))
            ax1.set_ylim([0, max_y_val*2.0])
            ax1.set_xlim(xlims)
            ax1.xaxis.set_major_locator(MultipleLocator(0.50))
            ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.tick_params(axis='both', which='both', direction='in')

            major_xtick_loc = np.array([0., 0.25, 0.75, 1.0])
            ax1.set_xticks(major_xtick_loc, [])
            ax1.xaxis.set_minor_locator(MultipleLocator(0.50))

            ylabel = 'Events / bin'
            ax1.set_ylabel(ylabel, fontsize=16, loc='top')

            ax1.vlines(0.5, 0, np.max([hist_total_mc.values()]), color='black', lw=0.75, ls='--')

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

            major_xtick_loc = np.array([0., 0.25, 0.75, 1.0])
            xtick_labels = ['', 'Below WP cutoff', 'Above WP cutoff', '']
            ax2.set_xticks(major_xtick_loc, xtick_labels, fontsize=16)
            ax2.xaxis.set_minor_locator(MultipleLocator(0.50))

            ax2.axhline(1.0, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
            ax2.set_ylabel(r'$\frac{Data}{Simulation}$', fontsize=20, loc='center')
        
            var_label = {
                    'qgl' : 'Quark-Gluon Likelihood',
                    'qgl_new' : 'Quark-Gluon Likelihood',
                    'particleNetAK4_QvsG' : 'ParticleNet discriminant',
                    'btagDeepFlavQG' : 'DeepJet discriminant'
                    }
            ax2.set_xlabel(var_label[variable], fontsize=16, loc='right', labelpad=12)
            
            plt.sca(ax1)

            channel_text = 'Dijet, Pythia' if channel=='dijet' else 'Z+jets, Pythia'
            channel_text_x = 0.615
            channel_text_y = 0.86
            plt.figtext(channel_text_x, channel_text_y, channel_text, fontsize=16, fontweight='semibold')

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

            plot_save_path = Path(f'{os.environ["COFFEAHOME"]}/output/UL17/170124/wp_val_plots/{channel}')
            plot_save_path.mkdir(parents=True, exist_ok=True)

            str_eta_low = str(eta_low).replace('.','_')
            str_eta_high = str(eta_high).replace('.','_')
            campaign = config['campaign']
            plot_save_name = plot_save_path.joinpath(f'{variable}_{channel}_{campaign}_eta{str_eta_low}to{str_eta_high}_pt{pt_low}to{pt_high}_{working_point}WP')

            if plot_syst_up:
                plot_save_name = Path(f'{plot_save_name}_{syst_name}_syst_up')
            if plot_syst_down:
                plot_save_name = Path(f'{plot_save_name}_{syst_name}_syst_down')

            if save_plot:
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
    parser.add_option('-v', '--variable', dest='variable', type='string', default='qgl', help='Variable to plot. (Possible: qgl, particleNet, deepJet)')
    parser.add_option('-c', '--channel', dest='channel', type='string', default='zmm', help='Choose which channel to process. (Possible: zmm, dijet)')
    parser.add_option('-w', '--wp', dest='working_point', default='medium', help='Choose working point: loose, medium, tight.')
    parser.add_option('-i', '--inclusive', dest='inclusive', action='store_true', default=False, help='Option to save plot inclusive in pT.')
    parser.add_option('-d', '--display', dest='display', action='store_true', default=False, help='Option to display plot.')
    parser.add_option('-r', '--root', dest='save_root', action='store_true', default=False, help='Option to save histogram as a ROOT file.')
    parser.add_option('-s', '--save', dest='save_plot', action='store_true', default=False, help='Option to save plot to .png and .pdf files.')
    parser.add_option('--syst', dest='syst_name', type='string', default='nominal', help='Systematic to vary: nominal, gluon, FSR, ISR, PU, JES, JER.')
    parser.add_option('--syst_up', dest='syst_up', action='store_true', default=False, help='Plot systematic variation up.')
    parser.add_option('--syst_down', dest='syst_down', action='store_true', default=False, help='Plot systematic variation down.')
    (opt, args) = parser.parse_args()

    if opt.working_point.lower() not in ['loose', 'medium', 'tight']:
        sys.exit(f'ERROR! Invalid working point: {opt.working_point.lower()}\nChoose loose, medium, or tight.')

    if opt.channel not in ['zmm', 'dijet']:
        sys.exit(f'ERROR! Invalid channel: {opt.channel}\nChoose: zmm or dijet')

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

    main(opt.config_path, opt.variable, opt.working_point.lower(), opt.display, opt.save_plot, opt.save_root, opt.inclusive, opt.syst_name.lower(), opt.syst_up, opt.syst_down, opt.channel)
