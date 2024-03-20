import uproot
import coffea
import correctionlib
import sys
import os
import json
import gzip
import cachetools
import copy

import numpy as np
import awkward as ak
import analysis_helpers as utils

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor, CorrectedJetsFactory, JECStack
from correctionlib.convert import from_uproot_THx

def find_qgl_bin(bins_dict, value):
    if (value < bins_dict[0]) or (value > bins_dict[len(bins_dict)-1]):
        return -1
    bin_num = 0
    while value > bins_dict[bin_num+1]:
        bin_num = bin_num+1
    return bin_num

def compute_jet_qgl(jet, qgl_evaluator):
    jet_rho_bin = jet['rho_bin']
    jet_eta_bin = jet['eta_bin']
    jet_pt_bin = jet['pt_bin']

    if (jet_rho_bin < 0 or jet_eta_bin < 0 or jet_pt_bin < 0):
        return -1.

    jet_mult = jet['qgl_mult']
    jet_ptD = jet['qgl_ptD']
    jet_axis2 = jet['qgl_axis2']
    jet_axis2 = -np.log(jet_axis2) if jet_axis2 > 0 else 0. 
    
    quark_likelihood = 1.
    gluon_likelihood = 1.

    var_names = ['mult', 'ptD', 'axis2']
    for var in var_names:
        quark_string = f'{var}/{var}_quark_eta{jet_eta_bin}_pt{jet_pt_bin}_rho{jet_rho_bin}'
        gluon_string = f'{var}/{var}_gluon_eta{jet_eta_bin}_pt{jet_pt_bin}_rho{jet_rho_bin}'

        if var == 'mult':
            input_var = float(jet_mult)
        if var == 'ptD':
            input_var = jet_ptD
        if var == 'axis2':
            input_var = jet_axis2

        var_quark_likelihood = qgl_evaluator[quark_string].evaluate(input_var)
        var_gluon_likelihood = qgl_evaluator[gluon_string].evaluate(input_var)

        if (var_quark_likelihood < 0) or (var_gluon_likelihood < 0):
            return -1.

        quark_likelihood *= var_quark_likelihood
        gluon_likelihood *= var_gluon_likelihood

    return round(quark_likelihood/(quark_likelihood+gluon_likelihood), 5)

class Channel():
    def __init__(self, name):
        self.name = name
        self.sel=None
        self.jets=None
        self.muons=None

    def apply_sel(self):                
        self.jets = self.jets[self.sel]
        if self.muons is not None:
            self.muons = self.muons[self.sel]
        
class DijetProcessor(processor.ProcessorABC):
    def __init__(self, campaign, use_jet_veto_maps, puppi, jes_up, jes_down, jer_up, jer_down):        
        self.output = {
            'cutflow' : processor.defaultdict_accumulator(int), 
            'weight': processor.column_accumulator(np.array([])),
            'PU_weight': processor.column_accumulator(np.array([])),
            'PU_weight_up': processor.column_accumulator(np.array([])),
            'PU_weight_down': processor.column_accumulator(np.array([])),
            'FSR_weight_up': processor.column_accumulator(np.array([])),
            'FSR_weight_down': processor.column_accumulator(np.array([])),
            'ISR_weight_up': processor.column_accumulator(np.array([])),
            'ISR_weight_down': processor.column_accumulator(np.array([])),
            'L1prefiring_weight': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_up': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_down': processor.column_accumulator(np.array([])),
            'gluon_weight': processor.column_accumulator(np.array([])),
            'Jet1_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet1_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet1_pt': processor.column_accumulator(np.array([])),
            'Jet1_eta': processor.column_accumulator(np.array([])),
            'Jet1_phi': processor.column_accumulator(np.array([])),
            'Jet1_mass': processor.column_accumulator(np.array([])),
            'Jet1_qgl': processor.column_accumulator(np.array([])),
            'Jet1_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet1_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet1_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavQG': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavG': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavB': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavCvB': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavCvL': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_B': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_CvsB': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_CvsL': processor.column_accumulator(np.array([])),
            'Jet2_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet2_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet2_pt': processor.column_accumulator(np.array([])),
            'Jet2_eta': processor.column_accumulator(np.array([])),
            'Jet2_phi': processor.column_accumulator(np.array([])),
            'Jet2_mass': processor.column_accumulator(np.array([])),
            'Jet2_qgl': processor.column_accumulator(np.array([])),
            'Jet2_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet2_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet2_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavQG': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavG': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavB': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavCvB': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavCvL': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_B': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_CvsB': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_CvsL': processor.column_accumulator(np.array([])),
            'alpha' : processor.column_accumulator(np.array([])),
            'fixedGridRhoFastjetAll': processor.column_accumulator(np.array([])),
            'LHE_HT': processor.column_accumulator(np.array([])),
            'LHE_HTIncoming': processor.column_accumulator(np.array([])),
            'Generator_binvar': processor.column_accumulator(np.array([])),
            'PV_npvs': processor.column_accumulator(np.array([])),
            'PV_npvsGood': processor.column_accumulator(np.array([])),
            'Pileup_nTrueInt': processor.column_accumulator(np.array([])),
            }

        self.campaign = campaign
        self.puppi = puppi
        self.jes_up = jes_up
        self.jes_down = jes_down
        self.jer_up = jer_up
        self.jer_down = jer_down

        self.jet_corrector = utils.jerc_objects(campaign)
        self.qgl_evaluator, self.qgl_rho_bin_dict, self.qgl_eta_bin_dict, self.qgl_pt_bin_dict = utils.qgl_pdf_and_binning(campaign)

        if use_jet_veto_maps:
            self.jet_veto_maps = utils.jet_veto_maps(campaign)
        else:
            self.jet_veto_maps = None

        pileup_weights, pileup_weights_up, pileup_weights_down = utils.pileup_weights(campaign, 'zmm')
        self.pileup_weight_evaluator = pileup_weights.to_evaluator()
        self.pileup_weight_up_evaluator = pileup_weights_up.to_evaluator()
        self.pileup_weight_down_evaluator = pileup_weights_down.to_evaluator()
    
    def process(self, events):
        output = self.output
        dataset = events.metadata['dataset']

        if dataset in utils.data_names:
            filetype = 'data'
        elif dataset in utils.mc_names:
            filetype = 'mc'

        if self.puppi:
            jets = events.JetPuppi
        else:
            jets = events.Jet
        jets = events.Jet

        if filetype == 'mc':            
            lumi_mask = np.ones(len(jets), dtype=np.bool_) # dummy lumi mask for MC
        else:
            run = events['run']
            lumiblock = events['luminosityBlock']
            lumi_json_file = utils.lumi_json(self.campaign)
            lumi_mask = LumiMask(lumi_json_file)(run, lumiblock) 

        nEvents = len(events)
        event_rho = events.fixedGridRhoFastjetAll
        PV_npvs = events.PV.npvs
        PV_npvsGood = events.PV.npvsGood
        Pileup_nTrueInt = events.Pileup.nTrueInt if 'Pileup' in events.fields else ak.ones_like(PV_npvs)
        if 'Generator' in events.fields:
            Generator_binvar = events.Generator.binvar if 'binvar' in events.Generator.fields else ak.ones_like(PV_npvs)
        if 'LHE' in events.fields:
            LHE_HT = events.LHE.HT if 'HT' in events.LHE.fields else ak.ones_like(PV_npvs)
            LHE_HTIncoming = events.LHE.HTIncoming if 'HTIncoming' in events.LHE.fields else ak.ones_like(PV_npvs)
            LHEPart_pdgId = events.LHEPart.pdgId if 'pdgId' in events.LHEPart.fields else ak.ones_like(PV_npvs)
            LHEPart_status = events.LHEPart.status if 'status' in events.LHEPart.fields else ak.ones_like(PV_npvs)
        PSWeight = events.PSWeight if 'PSWeight' in events.fields else ak.ones_like(PV_npvs)

        if filetype == 'data':            
            Generator_binvar = ak.ones_like(PV_npvs)
            LHE_HT = ak.ones_like(PV_npvs)
            LHE_HTIncoming = ak.ones_like(PV_npvs)
            LHEPart_pdgId = ak.ones_like(PV_npvs)
            LHEPart_status = ak.ones_like(PV_npvs)

        # Cutflow to keep track of number of events and jets
        output['cutflow']['all events'] += nEvents
        output['cutflow']['all jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #

        trigger = utils.hlt_trigger(self.campaign, 'dijet')
        trigger_mask = (events.HLT[trigger] == 1)

        # # # # # # # #
        # MET FILTERS #
        # # # # # # # #

        MET_mask = (
                (events.Flag.goodVertices == 1) & (events.Flag.globalSuperTightHalo2016Filter == 1) & (events.Flag.HBHENoiseFilter==1) &
                (events.Flag.HBHENoiseIsoFilter==1) & (events.Flag.EcalDeadCellTriggerPrimitiveFilter==1) & (events.Flag.BadPFMuonFilter==1) &
                (events.Flag.BadPFMuonDzFilter==1) & (events.Flag.eeBadScFilter==1) & (events.Flag.ecalBadCalibFilter==1)
                )

        # # # # # # # # # # #
        # OBJECT SELECTIONS #
        # # # # # # # # # # #

        jets = utils.object_selection(jets, 'jet', self.campaign)
        output['cutflow']['cleaned jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #

        jet_mask = (ak.num(jets) >= 2)

        event_mask = jet_mask & lumi_mask & trigger_mask & MET_mask

        # TO-DO: Change implementation of cuts to use coffea.PackedSelection, available in newer Coffea version

        if 'GenVtx' in events.fields:
            vtx_mask = np.abs(events.GenVtx.z - events.PV.z) < 0.2
            event_mask = event_mask & vtx_mask

        event_rho = event_rho[event_mask]
        PV_npvs = PV_npvs[event_mask]
        PV_npvsGood = PV_npvsGood[event_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[event_mask]
        Generator_binvar = Generator_binvar[event_mask]
        LHE_HT = LHE_HT[event_mask]
        LHE_HTIncoming = LHE_HTIncoming[event_mask]
        LHEPart_pdgId = LHEPart_pdgId[event_mask]
        LHEPart_status = LHEPart_status[event_mask]
        PSWeight = PSWeight[event_mask]

        dijet = Channel('dijet')
        dijet.jets = jets
        dijet.sel = event_mask
        dijet.apply_sel()

        # Apply JECs to MC jets
        if filetype == 'mc':
            # Impose the requirement delta_r < 0.2 between reco jets and gen jets for the JER "scaling method"
            genjet_delta_r = dijet.jets.delta_r(dijet.jets.matched_gen)
            matched_gen_pt = ak.where(genjet_delta_r > 0.2, 0., dijet.jets.matched_gen['pt'])

            # Add the missing variables needed for energy corrections
            dijet.jets['pt_raw'] = (1 - dijet.jets['rawFactor'])*dijet.jets['pt']
            dijet.jets['mass_raw'] = (1 - dijet.jets['rawFactor'])*dijet.jets['mass']
            dijet.jets['pt_gen'] = ak.values_astype(ak.fill_none(matched_gen_pt, 0.), np.float32)
            dijet.jets['rho'] = ak.broadcast_arrays(event_rho, dijet.jets['pt'])[0]

            jerc_cache = cachetools.Cache(np.inf)
            dijet.jets = self.jet_corrector.build(dijet.jets, jerc_cache)

            # Apply JES and JER variations before event selection
            if self.jes_up:
                dijet.jets = dijet.jets.JES_Total.up
            elif self.jes_down:
                dijet.jets = dijet.jets.JES_Total.down

            if self.jer_up:
                dijet.jets = dijet.jets.JER.up
            elif self.jer_down:
                dijet.jets = dijet.jets.JER.down

            # Sort again by jet pT
            sorted_pt_arg = ak.argsort(dijet.jets['pt'], ascending=False)
            dijet.jets = dijet.jets[sorted_pt_arg]

            # GenJet matching
            leading_jet_genJetIdx = dijet.jets[:,0].genJetIdx
            subleading_jet_genJetIdx = dijet.jets[:,1].genJetIdx
            genjet_match_mask = ak.all([(leading_jet_genJetIdx >= 0), (subleading_jet_genJetIdx >= 0)], axis=0)
        else:
            dijet.jets['rho'] = ak.broadcast_arrays(event_rho, dijet.jets['pt'])[0]
            genjet_match_mask = np.ones(len(dijet.jets), dtype=np.bool_)

        deltaphi_mask = np.abs(dijet.jets[:,0].delta_phi(dijet.jets[:,1])) > 2.7

        subsubleading_mask = np.asarray(dijet.jets.mass[:,0] > -1) # Start with always true
        leading_jet_pt = dijet.jets['pt'][ak.num(dijet.jets) >= 3, 0]
        subleading_jet_pt = dijet.jets['pt'][ak.num(dijet.jets) >= 3, 1]
        subsubleading_jet_pt = dijet.jets['pt'][ak.num(dijet.jets) >= 3, 2]
        dijet_pt_average = (leading_jet_pt + subleading_jet_pt)*0.5
        subsubleading_mask[ak.num(dijet.jets) >= 3] = np.asarray(subsubleading_jet_pt < dijet_pt_average)

        alpha = np.zeros(len(dijet.jets))
        alpha[ak.num(dijet.jets) >= 3] = subsubleading_jet_pt / dijet_pt_average

        leading_jet_pt = dijet.jets['pt'][:,0]
        subleading_jet_pt = dijet.jets['pt'][:,1]
        pt_balance_mask = (leading_jet_pt-subleading_jet_pt)/(leading_jet_pt+subleading_jet_pt) < 0.7

        if self.jet_veto_maps:
            # NB! The jet veto maps are inverted in order to use them as masks,
            # since by default a jet veto map outputs a zero if the jet passes the veto
            leading_jet_eta = dijet.jets['eta'][:,0]
            subleading_jet_eta = dijet.jets['eta'][:,1]
            leading_jet_phi = dijet.jets['phi'][:,0]
            subleading_jet_phi = dijet.jets['phi'][:,1]

            # Necessary transformation for phi so that the edge values don't go out of bounds for the jet veto maps.
            # This doesn't affect the results.
            leading_jet_phi = (leading_jet_phi + np.pi) % (2 * np.pi) - np.pi
            subleading_jet_phi = (subleading_jet_phi + np.pi) % (2 * np.pi) - np.pi

            # Drop the event if either of the two leading jets is in a jet veto map area
            leading_jet_veto_map_mask = self.jet_veto_maps.evaluate("jetvetomap", leading_jet_eta, leading_jet_phi) == 0
            subleading_jet_veto_map_mask = self.jet_veto_maps.evaluate("jetvetomap", subleading_jet_eta, subleading_jet_phi) == 0
            jet_veto_map_mask = leading_jet_veto_map_mask & subleading_jet_veto_map_mask 
        else:
            jet_veto_map_mask = np.ones(len(dijet.jets), dtype=np.bool_)

        # Apply dijet selection
        dijet_mask = deltaphi_mask & subsubleading_mask & pt_balance_mask & genjet_match_mask & jet_veto_map_mask
        dijet.sel = dijet_mask
        dijet.apply_sel()
        output['cutflow']['dijet selection'] += len(dijet.jets)

        alpha = alpha[dijet_mask]
        PV_npvs = PV_npvs[dijet_mask]
        PV_npvsGood = PV_npvsGood[dijet_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[dijet_mask]
        Generator_binvar = Generator_binvar[dijet_mask]
        LHE_HT = LHE_HT[dijet_mask]
        LHE_HTIncoming = LHE_HTIncoming[dijet_mask]
        LHEPart_pdgId = LHEPart_pdgId[dijet_mask]
        LHEPart_status = LHEPart_status[dijet_mask]
        PSWeight = PSWeight[dijet_mask]

        # Calculate the QGL value for each jet
        find_rho_bin = lambda x : find_qgl_bin(self.qgl_rho_bin_dict, x)
        find_eta_bin = lambda x : find_qgl_bin(self.qgl_eta_bin_dict, x)
        find_pt_bin = lambda x : find_qgl_bin(self.qgl_pt_bin_dict, x)

        # Only pick the variables relevant for computing QGL
        qgl_variables = ['pt', 'eta', 'rho', 'qgl_axis2', 'qgl_mult', 'qgl_ptD']

        leading_jets = ak.zip({var: dijet.jets[:,0][var] for var in qgl_variables})
        subleading_jets = ak.zip({var: dijet.jets[:,1][var] for var in qgl_variables})

        leading_jets['rho_bin'] = np.fromiter(map(find_rho_bin, leading_jets['rho']), dtype=int)
        leading_jets['eta_bin'] = np.fromiter(map(find_eta_bin, np.abs(leading_jets['eta'])), dtype=int)
        leading_jets['pt_bin'] = np.fromiter(map(find_pt_bin, leading_jets['pt']), dtype=int)

        subleading_jets['rho_bin'] = np.fromiter(map(find_rho_bin, subleading_jets['rho']), dtype=int)
        subleading_jets['eta_bin'] = np.fromiter(map(find_eta_bin, np.abs(subleading_jets['eta'])), dtype=int)
        subleading_jets['pt_bin'] = np.fromiter(map(find_pt_bin, subleading_jets['pt']), dtype=int)

        jet1_qgl = np.fromiter(map(lambda jet: compute_jet_qgl(jet, self.qgl_evaluator), leading_jets), dtype=float)
        jet2_qgl = np.fromiter(map(lambda jet: compute_jet_qgl(jet, self.qgl_evaluator), subleading_jets), dtype=float)

        # Add weights and save output
        n_dijet = len(dijet.jets)
        if n_dijet > 0:
            if filetype == 'mc':
                with open(os.path.join(os.environ['COFFEAHOME'],'utils','utils.json'),'r') as utils_json_file:
                    utils_json = json.load(utils_json_file)

                xsecs = utils_json[self.campaign]['xsec']
                nGenEvents = utils_json[self.campaign]['nGenEvents']
                lumi = utils_json[self.campaign]['lumi']

                sample_name = next((x for x in xsecs.keys() if dataset in x), None)
                if sample_name is None:
                    sys.exit('ERROR: xsec for sample not found in the JSON file.')
                if sample_name not in nGenEvents.keys():
                    sys.exit('ERROR: nGenEvents for sample not found in the JSON file.')
                sample_xsec = xsecs[sample_name]*1000 # Convert from pb to fb
                sample_nGenEvents = nGenEvents[sample_name]
                sample_lumi = lumi['ZeroBias']
                mc_scale = sample_lumi*sample_xsec / sample_nGenEvents

                weights = events['genWeight'][event_mask]
                weights = weights[dijet_mask].to_numpy()*mc_scale

                PU_weight = self.pileup_weight_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_up = self.pileup_weight_up_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_down = self.pileup_weight_down_evaluator.evaluate(Pileup_nTrueInt)

                nPSWeight = [len(x) for x in events.PSWeight]
                if len(np.unique(nPSWeight))==1 and np.unique(nPSWeight[0])==44:
                    FSR_weight_down = PSWeight[:,2].to_numpy()
                    FSR_weight_up = PSWeight[:,3].to_numpy()
                    ISR_weight_down = PSWeight[:,24].to_numpy()
                    ISR_weight_up = PSWeight[:,25].to_numpy()
                else:
                    sys.exit('ERROR! PSWeights not assigned correctly, as there aren\'t 44 weights.')

                L1prefiring_weight = events.L1PreFiringWeight.Nom[event_mask][dijet_mask].to_numpy()
                L1prefiring_weight_up = events.L1PreFiringWeight.Up[event_mask][dijet_mask].to_numpy()
                L1prefiring_weight_down = events.L1PreFiringWeight.Dn[event_mask][dijet_mask].to_numpy()

                LHEPart_isGluon = np.abs(LHEPart_pdgId == 21)
                LHEPart_isIncoming = LHEPart_status == -1
                gluon_weights = np.power(1.10, ak.num(LHEPart_pdgId[LHEPart_isGluon & LHEPart_isIncoming])).to_numpy()

                jet1_PartonFlavour = dijet.jets['partonFlavour'][:,0].to_numpy()
                jet1_HadronFlavour = dijet.jets['hadronFlavour'][:,0].to_numpy()
                jet2_PartonFlavour = dijet.jets['partonFlavour'][:,1].to_numpy()
                jet2_HadronFlavour = dijet.jets['hadronFlavour'][:,1].to_numpy()
            else:
                weights = np.ones(len(dijet.jets))
                PU_weight = np.ones(len(dijet.jets))
                PU_weight_up = np.ones(len(dijet.jets))
                PU_weight_down = np.ones(len(dijet.jets))
                gluon_weights = np.ones(len(dijet.jets))
                FSR_weight_up = np.ones(len(dijet.jets))
                FSR_weight_down = np.ones(len(dijet.jets))
                ISR_weight_up = np.ones(len(dijet.jets))
                ISR_weight_down = np.ones(len(dijet.jets))
                L1prefiring_weight = np.ones(len(dijet.jets))
                L1prefiring_weight_up = np.ones(len(dijet.jets))
                L1prefiring_weight_down = np.ones(len(dijet.jets))
                jet1_PartonFlavour = np.ones(len(dijet.jets))
                jet1_HadronFlavour = np.ones(len(dijet.jets))
                jet2_PartonFlavour = np.ones(len(dijet.jets))
                jet2_HadronFlavour = np.ones(len(dijet.jets))

            output['weight'] += processor.column_accumulator(np.reshape(weights, (n_dijet,)))
            output['PU_weight'] += processor.column_accumulator(np.reshape(PU_weight, (n_dijet,)))
            output['PU_weight_up'] += processor.column_accumulator(np.reshape(PU_weight_up, (n_dijet,)))
            output['PU_weight_down'] += processor.column_accumulator(np.reshape(PU_weight_down, (n_dijet,)))
            output['FSR_weight_up'] += processor.column_accumulator(np.reshape(FSR_weight_up, (n_dijet,)))
            output['FSR_weight_down'] += processor.column_accumulator(np.reshape(FSR_weight_down, (n_dijet,)))
            output['ISR_weight_up'] += processor.column_accumulator(np.reshape(ISR_weight_up, (n_dijet,)))
            output['ISR_weight_down'] += processor.column_accumulator(np.reshape(ISR_weight_down, (n_dijet,)))
            output['L1prefiring_weight'] += processor.column_accumulator(np.reshape(L1prefiring_weight, (n_dijet,)))
            output['L1prefiring_weight_up'] += processor.column_accumulator(np.reshape(L1prefiring_weight_up, (n_dijet,)))
            output['L1prefiring_weight_down'] += processor.column_accumulator(np.reshape(L1prefiring_weight_down, (n_dijet,)))
            output['gluon_weight'] += processor.column_accumulator(np.reshape(gluon_weights, (n_dijet,)))
            output['Jet1_PartonFlavour'] += processor.column_accumulator(np.reshape(jet1_PartonFlavour, (n_dijet,)))
            output['Jet1_HadronFlavour'] += processor.column_accumulator(np.reshape(jet1_HadronFlavour, (n_dijet,)))
            output['Jet1_pt'] += processor.column_accumulator(np.reshape(dijet.jets['pt'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_eta'] += processor.column_accumulator(np.reshape(dijet.jets['eta'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_phi'] += processor.column_accumulator(np.reshape(dijet.jets['phi'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_mass'] += processor.column_accumulator(np.reshape(dijet.jets['mass'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl'] += processor.column_accumulator(np.reshape(jet1_qgl, (n_dijet,)))
            output['Jet1_qgl_axis2'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_axis2'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_ptD'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_ptD'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_mult'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_mult'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavUDS'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavQG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavQG'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavG'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavB'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavCvB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvB'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavCvL'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvL'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_QvsG'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_B'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_B'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_CvsB'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsB'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_CvsL'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsL'][:,0].to_numpy(), (n_dijet,)))
            output['Jet2_PartonFlavour'] += processor.column_accumulator(np.reshape(jet2_PartonFlavour, (n_dijet,)))
            output['Jet2_HadronFlavour'] += processor.column_accumulator(np.reshape(jet2_HadronFlavour, (n_dijet,)))
            output['Jet2_pt'] += processor.column_accumulator(np.reshape(dijet.jets['pt'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_eta'] += processor.column_accumulator(np.reshape(dijet.jets['eta'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_phi'] += processor.column_accumulator(np.reshape(dijet.jets['phi'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_mass'] += processor.column_accumulator(np.reshape(dijet.jets['mass'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl'] += processor.column_accumulator(np.reshape(jet2_qgl, (n_dijet,)))
            output['Jet2_qgl_axis2'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_axis2'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_ptD'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_ptD'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_mult'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_mult'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavUDS'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavQG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavQG'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavG'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavB'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavCvB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvB'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavCvL'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvL'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_QvsG'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_B'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_B'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_CvsB'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsB'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_CvsL'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsL'][:,1].to_numpy(), (n_dijet,)))
            output['alpha'] += processor.column_accumulator(np.reshape(alpha, (n_dijet,)))
            output['fixedGridRhoFastjetAll'] += processor.column_accumulator(np.reshape(dijet.jets['rho'][:,0].to_numpy(), (n_dijet,)))
            output['LHE_HT'] += processor.column_accumulator(np.reshape(LHE_HT.to_numpy(), (n_dijet,)))
            output['LHE_HTIncoming'] += processor.column_accumulator(np.reshape(LHE_HTIncoming.to_numpy(), (n_dijet,)))
            output['Generator_binvar'] += processor.column_accumulator(np.reshape(Generator_binvar.to_numpy(), (n_dijet,)))
            output['PV_npvs'] += processor.column_accumulator(np.reshape(PV_npvs.to_numpy(), (n_dijet,)))
            output['PV_npvsGood'] += processor.column_accumulator(np.reshape(PV_npvsGood.to_numpy(), (n_dijet,)))
            output['Pileup_nTrueInt'] += processor.column_accumulator(np.reshape(Pileup_nTrueInt.to_numpy(), (n_dijet,)))
        return output

    def postprocess(self, accumulator):
        return accumulator

# Processor for z+jets analysis
class ZmmProcessor(processor.ProcessorABC):
    def __init__(self, campaign, use_jet_veto_maps, puppi, jes_up, jes_down, jer_up, jer_down):        
        self.output = {
            'cutflow' : processor.defaultdict_accumulator(int), 
            'weight': processor.column_accumulator(np.array([])),
            'PU_weight': processor.column_accumulator(np.array([])),
            'PU_weight_up': processor.column_accumulator(np.array([])),
            'PU_weight_down': processor.column_accumulator(np.array([])),
            'FSR_weight_up': processor.column_accumulator(np.array([])),
            'FSR_weight_down': processor.column_accumulator(np.array([])),
            'ISR_weight_up': processor.column_accumulator(np.array([])),
            'ISR_weight_down': processor.column_accumulator(np.array([])),
            'L1prefiring_weight': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_up': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_down': processor.column_accumulator(np.array([])),
            'gluon_weight': processor.column_accumulator(np.array([])),
            'Jet_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet_pt': processor.column_accumulator(np.array([])),
            'Jet_eta': processor.column_accumulator(np.array([])),
            'Jet_phi': processor.column_accumulator(np.array([])),
            'Jet_mass': processor.column_accumulator(np.array([])),
            'Jet_qgl': processor.column_accumulator(np.array([])),
            'Jet_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavQG': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavG': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavB': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavCvB': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavCvL': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_B': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_CvsB': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_CvsL': processor.column_accumulator(np.array([])),
            'Dimuon_mass': processor.column_accumulator(np.array([])),
            'Dimuon_pt': processor.column_accumulator(np.array([])),
            'Dimuon_eta': processor.column_accumulator(np.array([])),
            'alpha': processor.column_accumulator(np.array([])),
            'fixedGridRhoFastjetAll': processor.column_accumulator(np.array([])),
            'PV_npvs': processor.column_accumulator(np.array([])),
            'PV_npvsGood': processor.column_accumulator(np.array([])),
            'Pileup_nTrueInt': processor.column_accumulator(np.array([])),
            }

        self.campaign = campaign
        self.puppi = puppi
        self.jes_up = jes_up
        self.jes_down = jes_down
        self.jer_up = jer_up
        self.jer_down = jer_down

        self.jet_corrector = utils.jerc_objects(campaign)
        self.qgl_evaluator, self.qgl_rho_bin_dict, self.qgl_eta_bin_dict, self.qgl_pt_bin_dict = utils.qgl_pdf_and_binning(campaign)

        if use_jet_veto_maps:
            self.jet_veto_maps = utils.jet_veto_maps(campaign)
        else:
            self.jet_veto_maps = None

        pileup_weights, pileup_weights_up, pileup_weights_down = utils.pileup_weights(campaign, 'zmm')
        self.pileup_weight_evaluator = pileup_weights.to_evaluator()
        self.pileup_weight_up_evaluator = pileup_weights_up.to_evaluator()
        self.pileup_weight_down_evaluator = pileup_weights_down.to_evaluator()

    def process(self, events):
        output = self.output
        dataset = events.metadata['dataset']

        if dataset in utils.data_names:
            filetype = 'data'
        elif dataset in utils.mc_names:
            filetype = 'mc'

        if self.puppi:
            jets = events.JetPuppi
        else:
            jets = events.Jet

        muons = events.Muon
        electrons = events.Electron

        if filetype == 'mc':
            lumi_mask = np.ones(len(muons), dtype=np.bool_) #dummy lumi mask for MC
        else:
            run = events['run']
            lumiblock = events['luminosityBlock']
            lumi_json_file = utils.lumi_json(self.campaign)
            lumi_mask = LumiMask(lumi_json_file)(run, lumiblock) 

        nEvents = len(events)
        event_rho = events.fixedGridRhoFastjetAll
        PV_npvs = events.PV.npvs
        PV_npvsGood = events.PV.npvsGood

        Pileup_nTrueInt = events.Pileup.nTrueInt if 'Pileup' in events.fields else ak.ones_like(PV_npvs)
        if 'Generator' in events.fields:
            Generator_binvar = events.Generator.binvar if 'binvar' in events.Generator.fields else ak.ones_like(PV_npvs)
        if 'LHE' in events.fields:
            LHEPart_pdgId = events.LHEPart.pdgId if 'pdgId' in events.LHEPart.fields else ak.ones_like(PV_npvs)
            LHEPart_status = events.LHEPart.status if 'status' in events.LHEPart.fields else ak.ones_like(PV_npvs)
        PSWeight = events.PSWeight if 'PSWeight' in events.fields else ak.ones_like(PV_npvs)

        if filetype == 'data':            
            Generator_binvar = ak.ones_like(PV_npvs)
            LHEPart_pdgId = ak.ones_like(PV_npvs)
            LHEPart_status = ak.ones_like(PV_npvs)

        # This keeps track of how many events there are, as well as how many of each object exist in this events
        output['cutflow']['all events'] += nEvents
        output['cutflow']['all jets'] += ak.num(jets).to_numpy().sum()
        output['cutflow']['all muons'] += ak.num(muons).to_numpy().sum()
        output['cutflow']['all electrons'] += ak.num(electrons).to_numpy().sum()

        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #
        
        trigger = utils.hlt_trigger(self.campaign, 'zmm')
        trigger_mask = (events.HLT[trigger] == 1)

        # # # # # # # #
        # MET FILTERS #
        # # # # # # # #

        MET_mask = (
                (events.Flag.goodVertices == 1) & (events.Flag.globalSuperTightHalo2016Filter == 1) & (events.Flag.HBHENoiseFilter==1) &
                (events.Flag.HBHENoiseIsoFilter==1) & (events.Flag.EcalDeadCellTriggerPrimitiveFilter==1) & (events.Flag.BadPFMuonFilter==1) &
                (events.Flag.BadPFMuonDzFilter==1) & (events.Flag.eeBadScFilter==1) & (events.Flag.ecalBadCalibFilter==1)
                )

        # # # # # # # # # # #
        # OBJECT SELECTIONS #
        # # # # # # # # # # #
        
        muons = utils.object_selection(muons, 'muon', self.campaign)
        electrons = utils.object_selection(electrons, 'electron', self.campaign)
        jets = utils.object_selection(jets, 'jet', self.campaign)

        # Now we want to make sure no jets are within 0.4 delta-R of any muon.
        cross_jmu = ak.cartesian([jets, muons], nested=True)
        check_jmu = ak.all((cross_jmu.slot0.delta_r(cross_jmu.slot1) > 0.4), axis=-1)
        jets = jets[(check_jmu)&(jets.mass>-1)]

        output['cutflow']['cleaned jets'] += ak.num(jets).to_numpy().sum()
        output['cutflow']['cleaned muons'] += ak.num(muons).to_numpy().sum()
        output['cutflow']['cleaned electrons'] += ak.num(electrons).to_numpy().sum()

        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #

        jet_mask = (ak.num(jets) >= 1) 
        electron_mask = (ak.num(electrons) == 0)
        muon_mask = (ak.num(muons) == 2)

        PV_z = events.PV.z
        if 'GenVtx_z' in events.fields:
            GenVtx_z = events.GenVtx_z
            vtx_mask = np.abs(GenVtx_z - PV_z) < 0.2
        else:
            vtx_mask = np.ones_like(len(PV_z), dtype=bool)

        event_mask = jet_mask & electron_mask & muon_mask & lumi_mask & trigger_mask & MET_mask & vtx_mask

        event_rho = event_rho[event_mask]
        PV_npvs = PV_npvs[event_mask]
        PV_npvsGood = PV_npvsGood[event_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[event_mask]
        LHEPart_pdgId = LHEPart_pdgId[event_mask]
        LHEPart_status = LHEPart_status[event_mask]
        PSWeight = PSWeight[event_mask]

        z_jets = Channel('zmm')
        z_jets.muons = muons
        z_jets.jets = jets
        z_jets.sel = event_mask
        z_jets.apply_sel()

        # Apply JECs to MC jets
        if filetype == 'mc':
            # Impose the requirement delta_r < 0.2 between reco jets and gen jets for the JER "scaling method"
            genjet_delta_r = z_jets.jets.delta_r(z_jets.jets.matched_gen)
            matched_gen_pt = ak.where(genjet_delta_r > 0.2, 0., z_jets.jets.matched_gen['pt'])

            # Add the missing variables needed for energy corrections
            z_jets.jets['pt_raw'] = (1 - z_jets.jets['rawFactor'])*z_jets.jets['pt']
            z_jets.jets['mass_raw'] = (1 - z_jets.jets['rawFactor'])*z_jets.jets['mass']
            z_jets.jets['pt_gen'] = ak.values_astype(ak.fill_none(matched_gen_pt, 0.), np.float32)
            z_jets.jets['rho'] = ak.broadcast_arrays(event_rho, z_jets.jets['pt'])[0]

            jerc_cache = cachetools.Cache(np.inf)
            z_jets.jets = self.jet_corrector.build(z_jets.jets, jerc_cache)

            # Apply JES and JER variations before event selection
            if self.jes_up:
                z_jets.jets = z_jets.jets.JES_Total.up
            elif self.jes_down:
                z_jets.jets = z_jets.jets.JES_Total.down

            if self.jer_up:
                z_jets.jets = z_jets.jets.JER.up
            elif self.jer_down:
                z_jets.jets = z_jets.jets.JER.down

            # Sort again by jet pT
            sorted_pt_arg = ak.argsort(z_jets.jets['pt'], ascending=False)
            z_jets.jets = z_jets.jets[sorted_pt_arg]
        else:
            z_jets.jets['rho'] = ak.broadcast_arrays(event_rho, z_jets.jets['pt'])[0]

        leading_jet_pt = z_jets.jets['pt'][:,0]
        z_charge = z_jets.muons[:,0].charge+z_jets.muons[:,1].charge
        charge_mask = z_charge == 0

        z_cand = ak.combinations(z_jets.muons, 2)
        z_cand = z_cand.slot0 + z_cand.slot1

        cross_jz = ak.cartesian([z_jets.jets, z_cand])
        z_jet_deltaphi = cross_jz.slot0.delta_phi(cross_jz.slot1)
        deltaphi_mask = np.abs(z_jet_deltaphi[:,0]) > 2.7

        mass_mask = (z_cand.mass > 71.2) & (z_cand.mass < 111.2)

        subleading_mask = np.asarray(z_jets.jets.mass[:,0] > -1) # Start with always true
        subleading_jet_pt = z_jets.jets['pt'][ak.num(z_jets.jets) >= 2, 1]
        z_pt = z_cand.pt[ak.num(z_jets.jets) >= 2, 0]
        subleading_mask[ak.num(z_jets.jets) >= 2] = np.asarray((subleading_jet_pt / z_pt) < 1.0)

        alpha = np.zeros(len(z_jets.jets))
        alpha[ak.num(z_jets.jets) >= 2] = subleading_jet_pt / z_pt

        z_pt_mask = z_cand.pt > 15

        if filetype == 'mc':
            genjet_match_mask = z_jets.jets[:,0].genJetIdx >= 0
        else:
            genjet_match_mask = np.ones(len(z_jets.jets), dtype=np.bool_)

        if self.jet_veto_maps:
            # NB! The jet veto maps are inverted in order to use them as masks,
            # since by default a jet veto map outputs a zero if the jet passes the veto
            leading_jet_eta = z_jets.jets['eta'][:,0]
            leading_jet_phi = z_jets.jets['phi'][:,0]

            # Necessary transformation for phi so that the edge values don't go out of bounds for the jet veto maps.
            # This doesn't affect the results.
            leading_jet_phi = (leading_jet_phi + np.pi) % (2 * np.pi) - np.pi

            # Drop the event if the leading jet is in a jet veto map area
            jet_veto_map_mask = self.jet_veto_maps.evaluate("jetvetomap", leading_jet_eta, leading_jet_phi) == 0
        else:
            jet_veto_map_mask = np.ones(len(z_jets.jets), dtype=np.bool_)

        # Apply Z+Jets selection
        dilepton_mask = ak.flatten(mass_mask) & charge_mask & deltaphi_mask & subleading_mask & ak.flatten(z_pt_mask) & genjet_match_mask & jet_veto_map_mask
        z_jets.sel = dilepton_mask
        z_jets.apply_sel()
        output['cutflow']['dilepton selection'] += len(z_jets.jets)

        alpha = alpha[dilepton_mask]
        PV_npvs = PV_npvs[dilepton_mask]
        PV_npvsGood = PV_npvsGood[dilepton_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[dilepton_mask]
        LHEPart_pdgId = LHEPart_pdgId[dilepton_mask]
        LHEPart_status = LHEPart_status[dilepton_mask]
        PSWeight = PSWeight[dilepton_mask]

        # Calculate the QGL value for each leading jet
        find_rho_bin = lambda x : find_qgl_bin(self.qgl_rho_bin_dict, x)
        find_eta_bin = lambda x : find_qgl_bin(self.qgl_eta_bin_dict, x)
        find_pt_bin = lambda x : find_qgl_bin(self.qgl_pt_bin_dict, x)

        # Only pick the variables relevant for computing QGL
        qgl_variables = ['pt', 'eta', 'rho', 'qgl_axis2', 'qgl_mult', 'qgl_ptD']

        leading_jets = ak.zip({var: z_jets.jets[:,0][var] for var in qgl_variables})
        leading_jets['rho_bin'] = np.fromiter(map(find_rho_bin, leading_jets['rho']), dtype=int)
        leading_jets['eta_bin'] = np.fromiter(map(find_eta_bin, np.abs(leading_jets['eta'])), dtype=int)
        leading_jets['pt_bin'] = np.fromiter(map(find_pt_bin, leading_jets['pt']), dtype=int)

        z_jets.jets['qgl'] = np.fromiter(map(lambda jet: compute_jet_qgl(jet, self.qgl_evaluator), leading_jets), dtype=float)

        n_z_jets = len(z_jets.jets)
        if n_z_jets > 0:
            if filetype == 'mc':
                with open(os.path.join(os.environ['COFFEAHOME'],'utils','utils.json'),'r') as utils_json_file:
                    utils_json = json.load(utils_json_file)

                xsecs = utils_json[self.campaign]['xsec']
                nGenEvents = utils_json[self.campaign]['nGenEvents']
                lumi = utils_json[self.campaign]['lumi']

                sample_name = next((x for x in xsecs.keys() if dataset in x), None)
                if sample_name is None:
                    sys.exit('ERROR: xsec for sample not found in the JSON file.')
                if sample_name not in nGenEvents.keys():
                    sys.exit('ERROR: nGenEvents for sample not found in the JSON file.')
                sample_xsec = xsecs[sample_name]*1000 # Convert from pb to fb
                sample_nGenEvents = nGenEvents[sample_name]
                sample_lumi = lumi['DoubleMuon']
                mc_scale = sample_lumi*sample_xsec / sample_nGenEvents

                weights = events['genWeight'][event_mask]
                weights = weights[dilepton_mask].to_numpy()*mc_scale

                PU_weight = self.pileup_weight_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_up = self.pileup_weight_up_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_down = self.pileup_weight_down_evaluator.evaluate(Pileup_nTrueInt)

                nPSWeight = [len(x) for x in events.PSWeight]
                if len(np.unique(nPSWeight))==1 and np.unique(nPSWeight[0])==44:
                    FSR_weight_down = PSWeight[:,2].to_numpy()
                    FSR_weight_up = PSWeight[:,3].to_numpy()
                    ISR_weight_down = PSWeight[:,24].to_numpy()
                    ISR_weight_up = PSWeight[:,25].to_numpy()
                else:
                    sys.exit('ERROR! PSWeights not assigned correctly, as there aren\'t 44 weights.')

                L1prefiring_weight = events.L1PreFiringWeight.Nom[event_mask][dilepton_mask].to_numpy()
                L1prefiring_weight_up = events.L1PreFiringWeight.Up[event_mask][dilepton_mask].to_numpy()
                L1prefiring_weight_down = events.L1PreFiringWeight.Dn[event_mask][dilepton_mask].to_numpy()

                LHEPart_isGluon = np.abs(LHEPart_pdgId == 21)
                LHEPart_isIncoming = LHEPart_status == -1
                gluon_weights = np.power(1.10, ak.num(LHEPart_pdgId[LHEPart_isGluon & LHEPart_isIncoming])).to_numpy()

                jet_PartonFlavour = z_jets.jets['partonFlavour'][:,0].to_numpy()
                jet_HadronFlavour = z_jets.jets['hadronFlavour'][:,0].to_numpy()
            else:
                weights = np.ones(len(z_jets.jets))
                PU_weight = np.ones(len(z_jets.jets))
                PU_weight_up = np.ones(len(z_jets.jets))
                PU_weight_down = np.ones(len(z_jets.jets))
                gluon_weights = np.ones(len(z_jets.jets))
                FSR_weight_up = np.ones(len(z_jets.jets))
                FSR_weight_down = np.ones(len(z_jets.jets))
                ISR_weight_up = np.ones(len(z_jets.jets))
                ISR_weight_down = np.ones(len(z_jets.jets))
                L1prefiring_weight = np.ones(len(z_jets.jets))
                L1prefiring_weight_up = np.ones(len(z_jets.jets))
                L1prefiring_weight_down = np.ones(len(z_jets.jets))
                jet_PartonFlavour = np.ones(len(z_jets.jets))
                jet_HadronFlavour = np.ones(len(z_jets.jets))

            output['weight'] += processor.column_accumulator(np.reshape(weights, (n_z_jets,)))
            output['PU_weight'] += processor.column_accumulator(np.reshape(PU_weight, (n_z_jets,)))
            output['PU_weight_up'] += processor.column_accumulator(np.reshape(PU_weight_up, (n_z_jets,)))
            output['PU_weight_down'] += processor.column_accumulator(np.reshape(PU_weight_down, (n_z_jets,)))
            output['FSR_weight_up'] += processor.column_accumulator(np.reshape(FSR_weight_up, (n_z_jets,)))
            output['FSR_weight_down'] += processor.column_accumulator(np.reshape(FSR_weight_down, (n_z_jets,)))
            output['ISR_weight_up'] += processor.column_accumulator(np.reshape(ISR_weight_up, (n_z_jets,)))
            output['ISR_weight_down'] += processor.column_accumulator(np.reshape(ISR_weight_down, (n_z_jets,)))
            output['L1prefiring_weight'] += processor.column_accumulator(np.reshape(L1prefiring_weight, (n_z_jets,)))
            output['L1prefiring_weight_up'] += processor.column_accumulator(np.reshape(L1prefiring_weight_up, (n_z_jets,)))
            output['L1prefiring_weight_down'] += processor.column_accumulator(np.reshape(L1prefiring_weight_down, (n_z_jets,)))
            output['gluon_weight'] += processor.column_accumulator(np.reshape(gluon_weights, (n_z_jets,)))
            output['Jet_PartonFlavour'] += processor.column_accumulator(np.reshape(jet_PartonFlavour, (n_z_jets,)))
            output['Jet_HadronFlavour'] += processor.column_accumulator(np.reshape(jet_HadronFlavour, (n_z_jets,)))
            output['Jet_pt'] += processor.column_accumulator(np.reshape(z_jets.jets['pt'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_eta'] += processor.column_accumulator(np.reshape(z_jets.jets['eta'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_phi'] += processor.column_accumulator(np.reshape(z_jets.jets['phi'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_mass'] += processor.column_accumulator(np.reshape(z_jets.jets['mass'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl'] += processor.column_accumulator(np.reshape(z_jets.jets['qgl'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_axis2'] += processor.column_accumulator(np.reshape(z_jets.jets['qgl_axis2'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_ptD'] += processor.column_accumulator(np.reshape(z_jets.jets['qgl_ptD'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_mult'] += processor.column_accumulator(np.reshape(z_jets.jets['qgl_mult'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(z_jets.jets['btagDeepFlavUDS'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavQG'] += processor.column_accumulator(np.reshape(z_jets.jets['btagDeepFlavQG'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavG'] += processor.column_accumulator(np.reshape(z_jets.jets['btagDeepFlavG'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavB'] += processor.column_accumulator(np.reshape(z_jets.jets['btagDeepFlavB'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavCvB'] += processor.column_accumulator(np.reshape(z_jets.jets['btagDeepFlavCvB'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavCvL'] += processor.column_accumulator(np.reshape(z_jets.jets['btagDeepFlavCvL'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_QvsG'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_B'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_B'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_CvsB'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_CvsB'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_CvsL'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_CvsL'][:,0].to_numpy(), (n_z_jets,)))
            output['Dimuon_mass'] += processor.column_accumulator(np.reshape(z_cand[(dilepton_mask)].mass.to_numpy(), (n_z_jets,)))
            output['Dimuon_pt'] += processor.column_accumulator(np.reshape(z_cand[(dilepton_mask)].pt.to_numpy(), (n_z_jets,)))
            output['Dimuon_eta'] += processor.column_accumulator(np.reshape(z_cand[(dilepton_mask)].eta.to_numpy(), (n_z_jets,)))
            output['alpha'] += processor.column_accumulator(np.reshape(alpha, (n_z_jets,)))
            output['fixedGridRhoFastjetAll'] += processor.column_accumulator(np.reshape(z_jets.jets['rho'][:,0].to_numpy(), (n_z_jets,)))
            output['PV_npvs'] += processor.column_accumulator(np.reshape(PV_npvs.to_numpy(), (n_z_jets,)))
            output['PV_npvsGood'] += processor.column_accumulator(np.reshape(PV_npvsGood.to_numpy(), (n_z_jets,)))
            output['Pileup_nTrueInt'] += processor.column_accumulator(np.reshape(Pileup_nTrueInt.to_numpy(), (n_z_jets,)))
        return output

    def postprocess(self, accumulator):
        return accumulator
