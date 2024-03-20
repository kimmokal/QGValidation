import sys
import os
import correctionlib
import uproot
import numpy as np

from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor, CorrectedJetsFactory, JECStack
from correctionlib.convert import from_uproot_THx

data_names = ['DoubleMuon','ZeroBias']
mc_names = ['DY','QCD_HT50to100','QCD_HT100to200','QCD_HT200to300','QCD_HT300to500','QCD_HT500to700','QCD_HT700to1000',
            'QCD_HT1000to1500','QCD_HT1500to2000','QCD_HT2000toInf','QCD_Pt_15to30','QCD_Pt_30to50','QCD_Pt_50to80','QCD_Pt_80to120',
            'QCD_Pt_120to170','QCD_Pt_170to300','QCD_Pt_300to470','QCD_Pt_470to600','QCD_Pt_600to800','QCD_Pt_800to1000',
            'QCD_Pt_1000to1400','QCD_Pt_1400to1800','QCD_Pt_1800to2400','QCD_Pt_2400to3200','QCD_Pt_3200toInf', 'QCD_Pt-15to7000']

def lumi_json(campaign):
    if campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit('ERROR: Only UL16_preVFP, UL16_postVFP, UL17, UL18 campaigns supported!')

    lumi_json_path = os.path.join(os.environ['COFFEAHOME'],'data','json')
    lumi_json_file = {
            'UL16_preVFP': 'Cert_UL2016_preVFP_GoldenJSON.txt',
            'UL16_postVFP': 'Cert_UL2016_postVFP_GoldenJSON.txt',
            'UL17': 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
            'UL18': 'Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'
            }

    return f'{lumi_json_path}/{lumi_json_file[campaign]}'

def hlt_trigger(campaign, channel):
    if campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit('ERROR: Only UL16_preVFP, UL16_postVFP, UL17, UL18 campaigns supported!')
    if channel not in ['zmm', 'dijet']:
        sys.exit('ERROR: Only zmm and dijet channels supported!')

    zmm_trigger = {
            'UL16_preVFP' : 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
            'UL16_postVFP' : 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
            'UL17' : 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
            'UL18' : 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'
            }

    trigger = 'ZeroBias' if (channel == 'dijet') else zmm_trigger[campaign]
    return trigger

def jerc_objects(campaign):
    if campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit('ERROR: Only UL16_preVFP, UL16_postVFP, UL17, UL18 campaigns supported!')

    jer_mc_SF = {
            'UL16_preVFP': 'Summer20UL16APV_JRV3_MC_SF_AK4PFchs',
            'UL16_postVFP': 'Summer20UL16_JRV3_MC_SF_AK4PFchs',
            'UL17': 'Summer19UL17_JRV2_MC_SF_AK4PFchs',
            'UL18': 'Summer19UL18_JRV2_MC_SF_AK4PFchs'
            }

    jer_mc_PtResolution = {
            'UL16_preVFP': 'Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs',
            'UL16_postVFP': 'Summer20UL16_JRV3_MC_PtResolution_AK4PFchs',
            'UL17': 'Summer19UL17_JRV2_MC_PtResolution_AK4PFchs',
            'UL18': 'Summer19UL18_JRV2_MC_PtResolution_AK4PFchs'
            }

    jec_mc_L1FastJet = {
            'UL16_preVFP': 'Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs',
            'UL16_postVFP': 'Summer19UL16_V7_MC_L1FastJet_AK4PFchs',
            'UL17': 'Summer19UL17_V5_MC_L1FastJet_AK4PFchs',
            'UL18': 'Summer19UL18_V5_MC_L1FastJet_AK4PFchs'
            }

    jec_mc_L2Relative = {
            'UL16_preVFP': 'Summer19UL16APV_V7_MC_L2Relative_AK4PFchs',
            'UL16_postVFP': 'Summer19UL16_V7_MC_L2Relative_AK4PFchs',
            'UL17': 'Summer19UL17_V5_MC_L2Relative_AK4PFchs',
            'UL18': 'Summer19UL18_V5_MC_L2Relative_AK4PFchs'
            }

    jec_mc_UncSources = {
            'UL16_preVFP': 'RegroupedV2_Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs',
            'UL16_postVFP': 'RegroupedV2_Summer19UL16_V7_MC_UncertaintySources_AK4PFchs',
            'UL17': 'RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs',
            'UL18': 'RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs'
            }

    coffea_base_path = os.environ['COFFEAHOME']
    jerc_extractor = extractor()
    jerc_extractor.add_weight_sets([
        f'* * {coffea_base_path}/utils/JERC/{jer_mc_SF[campaign]}.txt',
        f'* * {coffea_base_path}/utils/JERC/{jer_mc_PtResolution[campaign]}.txt',
        f'* * {coffea_base_path}/utils/JERC/{jec_mc_L1FastJet[campaign]}.txt',
        f'* * {coffea_base_path}/utils/JERC/{jec_mc_L2Relative[campaign]}.txt',
        f'* * {coffea_base_path}/utils/JERC/{jec_mc_UncSources[campaign]}.junc.txt',
        ])
    jerc_extractor.finalize()
    jerc_evaluator = jerc_extractor.make_evaluator()

    jec_stack_names = [key for key in jerc_evaluator.keys()]
    jec_inputs = {name: jerc_evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"

    return CorrectedJetsFactory(name_map, jec_stack)


def qgl_pdf_and_binning(campaign):
    if campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit('ERROR: Only UL16_preVFP, UL16_postVFP, UL17, UL18 campaigns supported!')

    pdf = {
            'UL16_preVFP': 'PDF_QGL_JMEnano_UL16_Ak4CHS',
            'UL16_postVFP': 'PDF_QGL_JMEnano_UL16_Ak4CHS',
            'UL17': 'pdfQG_AK4chs_13TeV_UL17_ghosts',
            'UL18': 'PDF_QGL_JMEnano_UL18_Ak4CHS',
            }

    coffea_base_path = os.environ['COFFEAHOME']
    pdf_path = f'{coffea_base_path}/utils/QGL/{pdf[campaign]}'

    try:
        qgl_evaluator = correctionlib.CorrectionSet.from_file(f'{pdf_path}.corr.json')
    except:
        sys.exit(f'ERROR: {pdf_path}.corr.json not found. Remember to convert the .root to file to .json!')

    with uproot.open(f'{pdf_path}.root') as qgl_file:
        qgl_rho_bins = qgl_file['rhoBins'].members['fElements']
        qgl_eta_bins = qgl_file['etaBins'].members['fElements']
        qgl_pt_bins = qgl_file['ptBins'].members['fElements']

    qgl_rho_bin_dict = {}
    qgl_eta_bin_dict = {}
    qgl_pt_bin_dict = {}

    for i in range(len(qgl_rho_bins)):
        qgl_rho_bin_dict[i] = qgl_rho_bins[i]
    for i in range(len(qgl_eta_bins)):
        qgl_eta_bin_dict[i] = qgl_eta_bins[i]
    for i in range(len(qgl_pt_bins)):
        qgl_pt_bin_dict[i] = qgl_pt_bins[i]

    return qgl_evaluator, qgl_rho_bin_dict, qgl_eta_bin_dict, qgl_pt_bin_dict


def pileup_weights(campaign, channel):
    if campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit('ERROR: Only UL16_preVFP, UL16_postVFP, UL17, UL18 campaigns supported!')
    if channel not in ['zmm', 'dijet']:
        sys.exit('ERROR: Only zmm and dijet channels supported!')

    dataset = 'goldenJSON' if channel=='zmm' else 'HLT_ZeroBias'

    coffea_base_path = os.environ['COFFEAHOME']
    pileup_weights = from_uproot_THx(f'{coffea_base_path}/utils/pileup/PU_weights_{dataset}_69200ub_pythia8_{campaign}.root:weight', flow='clamp')
    pileup_weights_up = from_uproot_THx(f'{coffea_base_path}/utils/pileup/PU_weights_{dataset}_72400ub_pythia8_{campaign}.root:weight', flow='clamp')
    pileup_weights_down = from_uproot_THx(f'{coffea_base_path}/utils/pileup/PU_weights_{dataset}_66000ub_pythia8_{campaign}.root:weight', flow='clamp')

    return pileup_weights, pileup_weights_up, pileup_weights_down


def jet_veto_maps(campaign):
    if campaign != 'UL18':
        sys.exit('ERROR: Currently jet veto maps are included only for UL18.')
    coffea_base_path = os.environ['COFFEAHOME']
    return correctionlib.CorrectionSet.from_file(f'{coffea_base_path}/utils/jet_veto_maps/jetvetomaps_UL18.json.gz')['Summer19UL18_V1']


def object_selection(obj, name, campaign, selection=None):
    '''Apply standard selection to ojects.
    Inputs: 
    obj: jagged_array to apply the selection
    selection: bool mask for selection different than the standard
    Returns: obj array with selection applied '''

    if campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit('ERROR: Only UL16_preVFP, UL16_postVFP, UL17, UL18 campaigns supported!')

    if 'VFP' in campaign:
        campaign = 'UL16'

    obj_list = ['jet', 'muon', 'electron']
    
    if name not in obj_list and selection == None:
        sys.exit('ERROR: For non standard objects, provide a selection mask to be applied')

    if selection != None:
        return obj[selection]
    else:
        if campaign == 'UL16':
            if name == 'jet':
                selection = (
                    (np.abs(obj['eta']) < 4.7) &
                    (obj['pt'] > 20) &
                    (obj['jetId']==6) & # tight
                    (obj['mass'] > -1)
                )
            elif name == 'muon':
                selection = (
                    (np.abs(obj['eta']) < 2.3) &
                    (obj['pt'] > 20)  &
                    (obj['dxy'] < 0.2) &
                    (obj['dz'] < 0.5) &
                    (obj['pfRelIso04_all'] < 0.15) &
                    (obj['tightId']) &
                    (obj['mass'] > -1)
                )
            elif name == 'electron':
                selection = (
                    (np.abs(obj['eta']) < 2.4) & (obj['pt'] > 10)  &
                    (
                        ((np.abs(obj['eta']) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj['eta']) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutBased'] >= 1) &
                    (obj['mass'] > -1)        
                )
        elif campaign == 'UL17':
            if name == 'jet':
                selection = (
                    (np.abs(obj['eta']) < 4.7) &
                    (obj['pt'] > 20)  & 
                    (obj['jetId']==6) & # tight
                    (obj['mass'] > -1) 
                )
            elif name == 'muon':
                selection = (
                    (np.abs(obj['eta']) < 2.3) &
                    (obj['pt'] > 20) &
                    (obj['pfRelIso04_all'] < 0.15) &
                    (obj['tightId']) &
                    (obj['mass'] > -1)  
                )
            elif name == 'electron':
                selection = (
                    (np.abs(obj['eta']) < 2.4) & (obj['pt'] > 10)  &
                    (
                        ((np.abs(obj['eta']) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj['eta']) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutBased'] >= 1) &
                    (obj['mass'] > -1)        
                )
        elif campaign == 'UL18':
            if name == 'jet':
                selection = (
                    (np.abs(obj['eta']) < 4.7) &
                    (obj['pt'] > 20)  & 
                    (obj['jetId']==6) & # tight
                    (obj['mass'] > -1) 
                )
            elif name == 'muon':
                selection = (
                    (np.abs(obj['eta']) < 2.3) &
                    (obj['pt'] > 20) &
                    (obj['pfRelIso04_all'] < 0.15) &
                    (obj['tightId']) &
                    (obj['mass'] > -1)  
                )
            elif name == 'electron':
                selection = (
                    (np.abs(obj['eta']) < 2.4) & (obj['pt'] > 10)  &
                    (
                        ((np.abs(obj['eta']) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj['eta']) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutBased'] >= 1) &
                    (obj['mass'] > -1)        
                )

        return obj[selection]
    
if __name__ == '__main__':
    jerc_objects('UL18')
