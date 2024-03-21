import numpy as np
import tomllib
import uproot
import hist

from optparse import OptionParser
from pathlib import Path
from correctionlib import convert

def load_quark_gluon_hists(path_to_root_file):
    with uproot.open(path_to_root_file) as f:
        for key in f.keys():
            if key.endswith('_quark;1'):
                quark_hist = f[key].to_hist()
            if key.endswith('_gluon;1'):
                gluon_hist = f[key].to_hist()
            if key.endswith('_quark_syst_up;1'):
                quark_hist_up = f[key].to_hist()
            if key.endswith('_gluon_syst_up;1'):
                gluon_hist_up = f[key].to_hist()
            if key.endswith('_quark_syst_down;1'):
                quark_hist_down = f[key].to_hist()
            if key.endswith('_gluon_syst_down;1'):
                gluon_hist_down = f[key].to_hist()
    return quark_hist, gluon_hist, quark_hist_up, gluon_hist_up, quark_hist_down, gluon_hist_down

def main(config_path, variable):
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f'Error occured: {e}')
        print(f'Be sure to provide a valid config file!')
        sys.exit()

    print(f'\n--------------------------------------')
    print(f'COMBINING MC HISTOGRAMS FOR ROC CURVES')
    print(f'--------------------------------------')
    print(f'PROCESSING VARIABLE:  {variable}')

    input_file_path = Path(config['path']['input'])
    print(f'READING FILES FROM: {input_file_path}/hists_for_roc/')

    output_path = config['path']['output']
    save_path = Path(f'{output_path}/hists_for_roc/combined')
    save_path.mkdir(parents=True, exist_ok=True)
    print(f'WRITING OUTPUT TO:  {save_path}/')

    bins_to_combine = config['combine_bins_for_ROC_curves']['bins']
    for bin_to_combine in bins_to_combine:
        eta_bins = bin_to_combine['eta']
        pt_bins = bin_to_combine['pT']

        total_eta_range = f'{eta_bins[0]}to{eta_bins[-1]}'.replace('.', '_')
        total_pt_range = f'{pt_bins[0]}to{pt_bins[-1]}'.replace('.', '_')
        print(f'COMBINING BINS:     eta: {eta_bins}, pT: {pt_bins} to eta{total_eta_range}_pt{total_pt_range}')

        for j in range(len(eta_bins)-1):
            eta_low = eta_bins[j]
            eta_high = eta_bins[j+1]
            eta_range = f'{eta_low}to{eta_high}'.replace('.', '_')

            for i in range(len(pt_bins)-1):
                pt_low = pt_bins[i]
                pt_high = pt_bins[i+1]
                pt_range = str(pt_low) + 'to' + str(pt_high)

                campaign = config['campaign']
                hist_dijet_path = f'{input_file_path}/hists_for_roc/dijet/{variable}_dijet_{campaign}_eta{eta_range}_pt{pt_range}.root'
                hist_zmm_path = f'{input_file_path}/hists_for_roc/zmm/{variable}_zmm_{campaign}_eta{eta_range}_pt{pt_range}.root'

                hist_quark_dijet, hist_gluon_dijet, hist_quark_dijet_up, hist_gluon_dijet_up, hist_quark_dijet_down, hist_gluon_dijet_down = load_quark_gluon_hists(hist_dijet_path)
                hist_quark_zmm, hist_gluon_zmm, hist_quark_zmm_up, hist_gluon_zmm_up, hist_quark_zmm_down, hist_gluon_zmm_down = load_quark_gluon_hists(hist_zmm_path)

                # Combine channels
                hist_quark = hist_quark_dijet + hist_quark_zmm
                hist_gluon = hist_gluon_dijet + hist_gluon_zmm
                hist_quark_up = hist_quark_dijet_up + hist_quark_zmm_up
                hist_gluon_up = hist_gluon_dijet_up + hist_gluon_zmm_up
                hist_quark_down = hist_quark_dijet_down + hist_quark_zmm_down
                hist_gluon_down = hist_gluon_dijet_down + hist_gluon_zmm_down

                # Combine histograms
                if 'hist_quark_total' not in locals():
                    hist_quark_total = hist_quark
                    hist_gluon_total = hist_gluon
                    hist_quark_total_up = hist_quark_up
                    hist_gluon_total_up = hist_gluon_up
                    hist_quark_total_down = hist_quark_down
                    hist_gluon_total_down = hist_gluon_down
                else:
                    hist_quark_total += hist_quark
                    hist_gluon_total += hist_gluon
                    hist_quark_total_up += hist_quark_up
                    hist_gluon_total_up += hist_gluon_up
                    hist_quark_total_down += hist_quark_down
                    hist_gluon_total_down += hist_gluon_down

        del hist_quark_total
        del hist_gluon_total
        del hist_quark_total_up
        del hist_gluon_total_up
        del hist_quark_total_down
        del hist_gluon_total_down

        output_path = config['path']['output']
        save_path = Path(f'{output_path}/hists_for_roc/combined')

        root_save_name = f'{variable}_hists_combined_{campaign}_eta{total_eta_range}_pt{total_pt_range}'
        with uproot.recreate(f'{save_path}/{root_save_name}.root') as root_file:
            root_file['quark_hist'] = hist_quark_total
            root_file['gluon_hist'] = hist_gluon_total
            root_file['quark_hist_up'] = hist_quark_total_up
            root_file['gluon_hist_up'] = hist_gluon_total_up
            root_file['quark_hist_down'] = hist_quark_total_down
            root_file['gluon_hist_down'] = hist_gluon_total_down

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
