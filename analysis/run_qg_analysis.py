import uproot
import coffea
import sys
import os
import json
import numpy as np
import analysis_utils as utils
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from qg_processors import DijetProcessor, ZmmProcessor
from coffea import processor
from optparse import OptionParser

def main():
    parser = OptionParser(usage='%prog [opt]')
    parser.add_option('--workers', dest='workers', type='int', default=1, help='Number of workers to use. Default: %default')
    parser.add_option('--chunk', dest='chunk', type='int', default=4096, help='Chunk size. Default %default')
    parser.add_option('--maxchunk', dest='maxchunk', type='int', default=None, help='Maximum number of chunks. Default: %default')
    parser.add_option('--channel', dest='channel', type='string', default='zmm', help='Specify which channel to run [zmm/dijet]. Default: %default')
    parser.add_option('--files', dest='files', type='string', default='', help='Specify .txt file containing paths to JMENano files.')
    parser.add_option('--run', dest='run_samples', type='string', default='data', help='Specify which samples to run [data/mc/all]. Default: %default')
    parser.add_option('--campaign', dest='campaign', type='string', default='UL17', help='Specify the campaign [UL16_preVFP, UL16_postVFP, UL17, UL18]. Default: %default')
    parser.add_option('--out_dir', dest='out_dir', type='string', default='UL17', help='Output subdirectory to which the data is stored. Default: %default')
    parser.add_option('--out_id', dest='out_id', type='string', default='', help='ID for output file.')
    parser.add_option('--puppi', dest='puppi', action='store_true', default=False, help='Option for processing PUPPI jets. Default: %default')
    parser.add_option('--jetvetomaps', dest='jetvetomaps', action='store_true', default=False, help='Option to apply jet veto maps. Default: %default')
    parser.add_option('--jes_up', dest='jes_up', action='store_true', default=False, help='Option for processing JES up variations. Default: %default')
    parser.add_option('--jes_down', dest='jes_down', action='store_true', default=False, help='Option for processing JES down variations. Default: %default')
    parser.add_option('--jer_up', dest='jer_up', action='store_true', default=False, help='Option for processing JES up variations. Default: %default')
    parser.add_option('--jer_down', dest='jer_down', action='store_true', default=False, help='Option for processing JES down variations. Default: %default')
    (opt, args) = parser.parse_args()
    
    try:
        os.environ['COFFEAHOME']
    except:
        sys.exit('ERROR: Enviroment variables not set. Run activate_setup.sh first!')

    if opt.campaign not in ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']:
        sys.exit(f'ERROR: campaign {opt.campaign} not supported. Please choose from: UL16_preVFP, UL16_postVFP, UL17, UL18')

    if opt.jes_up and opt.jes_down:
        sys.exit('ERROR: Cannot do JES up and down variations simultaneously!')
    if opt.jes_up:
        opt.out_id += '_jes_up'
    elif opt.jes_down:
        opt.out_id += '_jes_down'
    if opt.jer_up and opt.jer_down:
        sys.exit('ERROR: Cannot do JER up and down variations simultaneously!')
    if opt.jer_up:
        opt.out_id += '_jer_up'
    elif opt.jer_down:
        opt.out_id += '_jer_down'

    if opt.run_samples == 'data':
        fout_name = f'data_{opt.out_id}.root'
    elif opt.run_samples == 'mc':
        fout_name = f'mc_{opt.out_id}.root'
    else:
        sys.exit('ERROR: You must specify what kind of dataset you want to run [--run]')

    print(f'Reading files from {opt.files}')
    files = []
    with open(os.path.join(opt.files),'r') as fread:
        files = fread.readlines()
        files = [x.strip() for x in files] 
    
    data_names = utils.data_names
    mc_names = utils.mc_names
    sample_names = data_names + mc_names
    fileset={}
    for f in files:
        for name in sample_names:
            if name in f:
                if name in fileset:
                    fileset[name].append(f)
                else:
                    fileset[name] = [f]

    for sample in fileset:
        print(f'{sample}:\nFiles: {fileset[sample]}')

    if opt.channel == 'zmm':
        qg_processor = ZmmProcessor(
                campaign = opt.campaign,
                use_jet_veto_maps = opt.jetvetomaps,
                puppi = opt.puppi,
                jes_up = opt.jes_up,
                jes_down = opt.jes_down,
                jer_up = opt.jer_up,
                jer_down = opt.jer_down
                )
    elif opt.channel == 'dijet':
        qg_processor = DijetProcessor(
                campaign = opt.campaign,
                use_jet_veto_maps = opt.jetvetomaps,
                puppi = opt.puppi,
                jes_up = opt.jes_up,
                jes_down = opt.jes_down,
                jer_up = opt.jer_up,
                jer_down = opt.jer_down
                )
    else:
        sys.exit('ERROR: Invalid channel!')

    JMENanoSchema = NanoAODSchema
    JMENanoSchema.mixins["JetPuppi"] = "Jet"
    JMENanoSchema.all_cross_references["JetPuppi_genJetIdx"] = "GenJet"

    job_executor = processor.FuturesExecutor(workers=opt.workers, mergepool=opt.workers, merging=True)
    run = processor.Runner(
                        executor = job_executor,
                        maxchunks = opt.maxchunk,
                        chunksize = opt.chunk,
                        schema=NanoAODSchema
                        )
    output = run(fileset, treename='Events', processor_instance=qg_processor)

    if not os.path.exists(os.path.join(os.environ['COFFEAHOME'],'output',opt.out_dir)):
        os.makedirs(os.path.join(os.environ['COFFEAHOME'],'output',opt.out_dir))

    cutflow_dict = {cutflow : output['cutflow'][cutflow] for cutflow in output['cutflow']}
    for cutflow, value in cutflow_dict.items():
        print(cutflow, value)

    with uproot.recreate(os.path.join(os.environ['COFFEAHOME'],'output',opt.out_dir,fout_name)) as fout:
        fout['Cutflow'] = {cutflow : [output['cutflow'][cutflow]] for cutflow in output['cutflow']}
        del output['cutflow']

        branch_dict = {}
        for var in output:
            branch_dict[var] = np.float32
            output[var] = output[var].value
        fout.mktree('Events', branch_dict)
        fout['Events'].extend(output)
    
if __name__ == '__main__':
    main()
