import numpy as np
import uproot

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

def roc_curve_from_hists(quark_hist, gluon_hist):
    true_pos_rate = []
    false_pos_rate = []
    
    quark_sum = np.sum(quark_hist)
    gluon_sum = np.sum(gluon_hist)

    for i in range(len(quark_hist)):
        true_positives = np.sum(quark_hist[i:])
        false_positives = np.sum(gluon_hist[i:])

        true_pos_rate.append(true_positives/quark_sum)
        false_pos_rate.append(false_positives/gluon_sum)

    true_pos_rate.append(0.)
    false_pos_rate.append(0.)
    
    return false_pos_rate, true_pos_rate

def calculate_efficiencies_from_cutoff(quark_hist, gluon_hist, cutoff_value):
    binning = np.linspace(0.,1.,len(quark_hist)+1)
    cutoff_bin = np.searchsorted(binning, cutoff_value)
    
    quark_sum = np.sum(quark_hist)
    gluon_sum = np.sum(gluon_hist)

    true_positives = np.sum(quark_hist[cutoff_bin:])
    false_positives = np.sum(gluon_hist[cutoff_bin:])

    quark_eff = np.round(true_positives/quark_sum, 3)
    gluon_eff = np.round(false_positives/gluon_sum, 3)

    return quark_eff, gluon_eff

def calculate_quark_eff(quark_hist, gluon_hist, gluon_eff_value):
    quark_sum = np.sum(quark_hist)
    gluon_sum = np.sum(gluon_hist)

    for cutoff_bin in range(len(quark_hist)):
        true_positives = np.sum(quark_hist[cutoff_bin:])
        false_positives = np.sum(gluon_hist[cutoff_bin:])

        gluon_eff = np.round(false_positives/gluon_sum, 2)
        if gluon_eff == round(gluon_eff_value, 2):
            quark_eff = np.round(true_positives/quark_sum, 2)
            return quark_eff, cutoff_bin

def calculate_gluon_eff(quark_hist, gluon_hist, quark_eff_value):
    quark_sum = np.sum(quark_hist)
    gluon_sum = np.sum(gluon_hist)

    for i in range(len(quark_hist)):
        true_positives = np.sum(quark_hist[i:])
        false_positives = np.sum(gluon_hist[i:])

        quark_eff = np.round(true_positives/quark_sum, 3)
        if round(quark_eff, 3) == round(quark_eff_value, 3):
            gluon_eff = np.round(false_positives/gluon_sum, 3)
            return gluon_eff

