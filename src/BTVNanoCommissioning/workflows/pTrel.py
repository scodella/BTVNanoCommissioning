import collections, awkward as ak, numpy as np
import os
import uproot
import correctionlib
from coffea import processor
from coffea.lookup_tools import extractor
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector
from BTVNanoCommissioning.helpers.BTA_helper import get_hadron_mass

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    common_shifts,
    weight_manager,
)

# user helper function
from BTVNanoCommissioning.helpers.func import (
    flatten,
    update,
    uproot_writeable,
    dump_lumi,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    ele_cuttightid,
    btag_wp_dict,
)

def load_Campaign(self):

    ## Method
    self._method = "pTrel" if "pTrel" in self.tag else "System8"

    ## HLT
    self.triggerInfos = collections.OrderedDict()
    self.triggerInfos['BTagMu_AK4DiJet20_Mu5']  = { 'jetPtRange' : [  20.,   50. ], 'ptAwayJet' : 20., 'ptTriggerEmulation' :  30., 'jetTrigger' :  'PFJet40' }
    self.triggerInfos['BTagMu_AK4DiJet40_Mu5']  = { 'jetPtRange' : [  50.,  100. ], 'ptAwayJet' : 30., 'ptTriggerEmulation' :  50., 'jetTrigger' :  'PFJet40' }
    self.triggerInfos['BTagMu_AK4DiJet70_Mu5']  = { 'jetPtRange' : [ 100.,  140. ], 'ptAwayJet' : 30., 'ptTriggerEmulation' :  80., 'jetTrigger' :  'PFJet60' }
    self.triggerInfos['BTagMu_AK4DiJet110_Mu5'] = { 'jetPtRange' : [ 140.,  200. ], 'ptAwayJet' : 30., 'ptTriggerEmulation' : 140., 'jetTrigger' :  'PFJet80' }
    self.triggerInfos['BTagMu_AK4DiJet170_Mu5'] = { 'jetPtRange' : [ 200.,  300. ], 'ptAwayJet' : 30., 'ptTriggerEmulation' : 200., 'jetTrigger' : 'PFJet140' }
    self.triggerInfos['BTagMu_AK4Jet300_Mu5']   = { 'jetPtRange' : [ 300., 1400. ], 'ptAwayJet' : 30., 'ptTriggerEmulation' :   0., 'jetTrigger' : 'PFJet260' }
   
    if self._campaign=="Summer24":
        totalLuminosity = 106787.472039533859253
        self.triggerInfos['BTagMu_AK4DiJet20_Mu5']['relativeEffectiveLuminosity']     = 33.365124159121163/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet20_Mu5']['jetRelativeEffectiveLuminosity']  = 0.216365419814463/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet40_Mu5']['relativeEffectiveLuminosity']     = 208.532025994507313/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet40_Mu5']['jetRelativeEffectiveLuminosity']  = 0.216365419814463/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet70_Mu5']['relativeEffectiveLuminosity']     = 1026.619204896035790/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet70_Mu5']['jetRelativeEffectiveLuminosity']  = 1.622740649/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet110_Mu5']['relativeEffectiveLuminosity']    = 4106.476819584/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet110_Mu5']['jetRelativeEffectiveLuminosity'] = 6.254903502/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet170_Mu5']['relativeEffectiveLuminosity']    = 17794.732884865/totalLuminosity
        self.triggerInfos['BTagMu_AK4DiJet170_Mu5']['jetRelativeEffectiveLuminosity'] = 71.178931539/totalLuminosity
        self.triggerInfos['BTagMu_AK4Jet300_Mu5']['relativeEffectiveLuminosity']      = 106768.397309188/totalLuminosity
        self.triggerInfos['BTagMu_AK4Jet300_Mu5']['jetRelativeEffectiveLuminosity']   = 834.128103978/totalLuminosity

    ## jet pt bins
    self.jetPtBins = collections.OrderedDict()
    if "Kinematics" in self.tag and "jetpt" not in self.tag:
        for trigger in self.triggerInfos:
            self.jetPtBins[trigger] = { 'jetPtRange' : self.triggerInfos[trigger]['jetPtRange'], 'trigger' : trigger }
    else:
        jetPtEdges = [ 20, 30, 50, 70, 100, 140, 200, 300, 600, 1000, 1400 ]
        for jetPtEdge in range(len(jetPtEdges)-1):
            ptbin = 'Pt'+str(jetPtEdges[jetPtEdge])+'to'+str(jetPtEdges[jetPtEdge+1])
            self.jetPtBins[ptbin] = {}
            self.jetPtBins[ptbin]['jetPtRange'] = [ float(jetPtEdges[jetPtEdge]), float(jetPtEdges[jetPtEdge+1]) ]
            for trigger in self.triggerInfos:
                if jetPtEdges[jetPtEdge]>=self.triggerInfos[trigger]['jetPtRange'][0]:
                    if jetPtEdges[jetPtEdge+1]<=self.triggerInfos[trigger]['jetPtRange'][1]:
                        self.jetPtBins[ptbin]['trigger'] = trigger

    ## Away jet
    self.btagAwayJetDiscriminant = "Bprob"
    self.btagAwayJetCut = 1.221 if "AwayJetDown" in self.tag else 5.134 if "AwayJetUp" in self.tag else 2.555
    self.btagVetoCut = 1.221

    ## bCorrector
    if len(list(btag_wp_dict[self._year+"_"+self._campaign].keys()))==1:
        if self._campaign=="Summer24":
            self.bCorrectorDiscriminant = "btagDeepFlavB" 
            self.bCorrectorCut = 0.6708 

    ## Muon selection
    self.muonKinBins = collections.OrderedDict()
    if self._method=="pTrel":
        self.muonKinBins["Bin1"] = { "range" : [  20.,   50. ], "pt" : 5., "dr" : 0.20 }
        self.muonKinBins["Bin2"] = { "range" : [  50.,  100. ], "pt" : 5., "dr" : 0.15 }
        self.muonKinBins["Bin3"] = { "range" : [ 100., 1400. ], "pt" : 5., "dr" : 0.12 }
        if "Mu" in self.tag:
            if "MuDRUp" in self.tag:
                self.muonKinBins["Bin1"]["dr"] = 0.15
                self.muonKinBins["Bin2"]["dr"] = 0.12
                self.muonKinBins["Bin3"]["dr"] = 0.09
            else:
                for mubin in self.muonKinBins:
                    if "MuPtUp"   in self.tag: self.muonKinBins[mubin]["pt"] = 8.
                    if "MuPtDown" in self.tag: self.muonKinBins[mubin]["pt"] = 6.
                    if "MuDRDown" in self.tag: self.muonKinBins[mubin]["dr"] = 999.
    elif self._method=="System8":
        self.muonKinBins["Bin1"] = { "range" : [  20., 1400. ], "pt" : 5., "dr" : 0.40 }
        if "MuPtUp"   in self.tag: self.muonKinBins["Bin1"]["pt"] = 8.
        if "MuPtDown" in self.tag: self.muonKinBins["Bin1"]["pt"] = 6.
        if "MuDRUp"   in self.tag: self.muonKinBins["Bin1"]["dr"] = 0.3
        if "MuDRDown" in self.tag: self.muonKinBins["Bin1"]["dr"] = 999.

    for ptbin in self.jetPtBins:
        for mubin in self.muonKinBins:
            if self.jetPtBins[ptbin]['jetPtRange'][0]>=self.muonKinBins[mubin]['range'][0]:         
                if self.jetPtBins[ptbin]['jetPtRange'][1]<=self.muonKinBins[mubin]['range'][1]:
                    self.jetPtBins[ptbin]["muPtCut"] = self.muonKinBins[mubin]["pt"]
                    self.jetPtBins[ptbin]["muDRCut"] = self.muonKinBins[mubin]["dr"]

    ## pthat safety cut
    if "Light" in self.tag:
        self.ptHatSafetyCuts = {  '15to80' : 200., '80to120' : 250., '120to170' : 340., '170to300' : 520., '300to10000' : 999999. }
    else:
        self.ptHatSafetyCuts = {  '15to20' :  60.,   '20to30' :  85.,   '30to50' : 120.,   '50to80' : 160.,  
                                 '80to120' : 220., '120to170' : 320., '170to300' : 440., '300to470' : 620., 
                                '470to600' : 720., '600to800' : 920., '800to10000' : 999999. }
        if int(self._year)>=2024:
            self.ptHatSafetyCuts['170to300'] =  540.
            self.ptHatSafetyCuts['300to470'] =  740.
            self.ptHatSafetyCuts['470to600'] =  870.
            self.ptHatSafetyCuts['600to800'] = 1000.

    ## prescale run range
    if self._year=="2022": self.ps_run_num = "355374_362760"
    elif self._year=="2023": self.ps_run_num = "366727_370790"
    elif self._year=="2024": self.ps_run_num = "378985_386951"

def pthat_safety_cut(ptHatSafetyCuts, pthat): # This kind of sucks!!!
    evt_zeros = ak.zeros_like(pthat)
    ptHatSafetyCut = evt_zeros
    for pthatbin in ptHatSafetyCuts:
        minpthat, maxpthat, jetptcut = float(pthatbin.split('to')[0]), float(pthatbin.split('to')[1]), ptHatSafetyCuts[pthatbin]
        ptHatSafetyCut = ak.where( (pthat>=minpthat) & (pthat<maxpthat), evt_zeros+jetptcut, ptHatSafetyCut )
    return ptHatSafetyCut                    

def get_psweight(jetPtBins, ps_run_num, jetPtBin, run, luminosityBlock, triggerInfos=False):
    psweight = ak.zeros_like(jetPtBin)
    for iptbin, ptbin in enumerate(jetPtBins):
        ptBin = ak.full_like(jetPtBin, iptbin+1)
        if triggerInfos: ptbin_trigger = triggerInfos[jetPtBins[ptbin]["trigger"]]["jetTrigger"]
        else: ptbin_trigger = jetPtBins[ptbin]['trigger']
        pseval = correctionlib.CorrectionSet.from_file( f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{ptbin_trigger}_run{ps_run_num}.json" )
        psweight = ak.values_astype( psweight + (jetPtBin==ptBin)*pseval["prescaleWeight"].evaluate(run, f"HLT_{ptbin_trigger}", ak.values_astype(luminosityBlock, np.float32)), float,)
    return psweight

def get_kinematic_weight(jetPt, jetEta, method, campaign, sample, level):
    sampleDir, sampleData = sample.split("/")[0], sample.split("/")[1]
    ext = extractor()
    ext.add_weight_sets([f"* * src/BTVNanoCommissioning/data/KinematicWeights/{campaign}/{sampleDir}/{method}_{sampleData}_{level}.root"])
    ext.finalize()
    return ext.make_evaluator()["kinematicWeights"](jetPt, jetEta)

def get_bfragmentation_weight(xB, genJetPt, campaign, shift=False):
    ext = extractor()
    ext.add_weight_sets([f"* * src/BTVNanoCommissioning/data/BFragmentation/bfragweights_vs_pt.root"])
    ext.finalize()
    passJet = ak.all(xB<1)*ak.all(genJetPt>=30)
    failJet = ak.full_like(xB, 1) - passJet
    bfragweight     = ak.values_astype( passJet*(ext.make_evaluator()["fragCP5BL"](xB, genJetPt))     + failJet, float,)
    bfragweightUp   = ak.values_astype( passJet*(ext.make_evaluator()["fragCP5BLup"](xB, genJetPt))   + failJet, float,)
    bfragweightDown = ak.values_astype( passJet*(ext.make_evaluator()["fragCP5BLdown"](xB, genJetPt)) + failJet, float,)
    if shift: 
        return bfragweight, bfragweightUp, bfragweightDown
    else:
        return ak.full_like(xB, 1.), bfragweightUp/bfragweight, bfragweightDown/bfragweight

def get_decay_weight(bHadronId, campaign):
    ext = extractor()      
    ext.add_weight_sets([f"* * src/BTVNanoCommissioning/data/BFragmentation/bdecayweights.root"])
    ext.finalize()
    failJet = ak.all(bHadronId!=511)*ak.all(bHadronId!=521)*ak.all(bHadronId!=531)*ak.all(bHadronId!=5122)
    passJet = ak.full_like(bHadronId, 1) - failJet
    #bdecayweightUp   = ak.values_astype( (passJet)*(ext.make_evaluator()["semilepbrup"](bHadronId))   + failJet, float,)
    #bdecayweightDown = ak.values_astype( (passJet)*(ext.make_evaluator()["semilepbrdown"](bHadronId)) + failJet, float,)
    return ak.full_like(bHadronId, 1.), ak.full_like(bHadronId, 1.1), ak.full_like(bHadronId, 0.9) #decayweightUp, bdecayweightDown

class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        self.tag = selectionModifier
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)
        for sfm in list(self.SF_map.keys()):
            if sfm!="campaign" and sfm!="PU":
                del self.SF_map[sfm]
        #if 'JME' in self.SF_map:
        #  del self.SF_map['JME'] #'campaign', 'PU', 'JME', 'jetveto', 'MUO_cfg', 'EGM_cfg', 'MUO', 'EGM'
        load_Campaign(self) 

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events)
        shifts = common_shifts(self, events)
        
        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    ## Processed events per-chunk, made selections, filled histogram, stored root files
    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        ######################
        #  Create histogram  # : Get the histogram dict from `histogrammer`
        ######################
        _hist_event_dict = (
            {"": None}
            if self.noHist
            else histogrammer(events, self.tag, self._year, self._campaign)
        )

        output = {
            "sumw": processor.defaultdict_accumulator(float),
            **_hist_event_dict,
        }
        if shift_name is None:
            if isRealData:
                output["sumw"] = len(events)
            else:
                output["sumw"] = ak.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ##### (mu)jet selection
 
        if "workingPoints" in self.tag:

            jet_sel = ak.fill_none( (events.Jet.pt>=30.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)), False, axis=-1 )
            event_jet = events.Jet[ jet_sel ]
            
            req_sel = (ak.num(event_jet.pt)>0 & ak.values_astype(events.PV.npvs>0, np.int32))

        elif "Light" in self.tag:

            req_trg = HLT_helper(events, [ self.triggerInfos[trigger]["jetTrigger"] for trigger in self.triggerInfos ] )

            trkj = events.JetPFCands[ (events.JetPFCands.pf.trkQuality!=0) & (events.JetPFCands.pt>1.0) ]
            trkj = trkj[ (trkj.pf.trkHighPurity==1) & (trkj.pf.trkAlgo!=9) & (trkj.pf.trkAlgo!=10) & (trkj.pf.numberOfHits>=11) &  (trkj.pf.trkPt>5.) 
                    & (trkj.pf.numberOfPixelHits>=2) & (trkj.pf.trkChi2<10) & (trkj.pf.lostOuterHits<=2) & (trkj.pf.dz<1.0) & (abs(trkj.pf.trkEta)<2.4) ]

            tagged_jet = events.Jet[ ak.fill_none( (events.Jet.pt>=20.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)) & (events.Jet[self.btagAwayJetDiscriminant]>=self.btagVetoCut), False, axis=-1 ) ]
            light_jet_sel = ak.fill_none( (events.Jet.pt>=20.) & (events.Jet.pt<1000.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)) & (ak.count_nonzero(events.Jet.metric_table(tagged_jet)>=0.05,axis=2)==0), False, axis=-1 )
            light_jet = events.Jet[light_jet_sel]

            req_sel = (ak.num(light_jet)>=1) & (ak.num(trkj)>=1) & req_trg

        else:   

            req_trg = HLT_helper(events, [ trigger for trigger in self.triggerInfos ] )

            softMuonStrategy = "BTAeasy"
            if softMuonStrategy=="BTA": # crush on data
                mutrkj = events.JetPFCands[ (events.JetPFCands.pf.trkQuality!=0) & (events.JetPFCands.pt>2.0) & (abs(events.JetPFCands.pf.pdgId)==13) & (events.JetPFCands.pf.numberOfHits>=11) & (events.JetPFCands.pf.numberOfPixelHits>=2) & (events.JetPFCands.pf.trkChi2<10) ]
                mucand = events.Muon[ (events.Muon.pt>5.) & (events.Muon.looseId>0.5) & (abs(events.Muon.eta)<2.4) & (events.Muon.jetIdx>=0) & (events.Muon.isGlobal) & (events.Muon.nStations>=2) ]
                pair = ak.cartesian( [mutrkj, mucand], axis=1, nested=True )  # dim: (event, pfcand, mu)
                matched = ( (pair["0"].pf.pdgId==pair["1"].pdgId) & (pair["0"].pf.delta_r(pair["1"])<0.01) & ((pair["0"].pt-pair["1"].pt)/pair["0"].pt<0.1) )
                event_softmu = ak.firsts( pair["1"][matched], axis=2 )  # dim same with mutrkj: (event, matched-pfcand), None if not matching with a muon
            elif softMuonStrategy=="BTAeasy": 
                mutrkj = events.JetPFCands[ (events.JetPFCands.pf.trkQuality!=0) & (events.JetPFCands.pt>2.0) & (abs(events.JetPFCands.pf.pdgId)==13) & (events.JetPFCands.pf.numberOfHits>=11) & (events.JetPFCands.pf.numberOfPixelHits>=2) & (events.JetPFCands.pf.trkChi2<10) ]
                mutrkj["eta"] = mutrkj.pf.eta
                mutrkj["phi"] = mutrkj.pf.phi
                event_softmu = events.Muon[ (events.Muon.pt>5.) & (events.Muon.looseId>0.5) & (abs(events.Muon.eta)<2.4) & (events.Muon.jetIdx>=0) & (events.Muon.isGlobal) & (events.Muon.nStations>=2) & (ak.count_nonzero(events.Muon.metric_table(mutrkj)<0.01,axis=2)>=1) ]
            elif softMuonStrategy=="pogMediumID":
                event_softmu = events.Muon[ (events.Muon.pt>5.) & (events.Muon.mediumId>0.5) & (abs(events.Muon.eta)<2.4) & (events.Muon.jetIdx>=0) ]

            mujet_sel = ak.fill_none( (ak.all(events.Jet.metric_table(event_softmu)<0.5,axis=2)) & (events.Jet.pt>=20.) & (events.Jet.pt<1000.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)), False, axis=-1 )
            event_mujet = events.Jet[ mujet_sel ]

            if self._method=="pTrel":

                awayjet_sel = ak.fill_none( (ak.all(events.Jet.metric_table(event_mujet)>1.5,axis=2)) & (events.Jet.pt>=20.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)) & (events.Jet[self.btagAwayJetDiscriminant]>=self.btagAwayJetCut), False, axis=-1 )
                event_awayjet = events.Jet[ awayjet_sel ]
                emuljet_sel = ak.fill_none( (ak.all(events.Jet.metric_table(event_mujet)>0.05,axis=2)) & (events.Jet.pt>=20.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)), False, axis=-1 )
                event_emuljet = events.Jet[ emuljet_sel ]

                req_sel = (ak.num(event_softmu.pt)==1) & (ak.num(event_mujet.pt)==1) & (ak.num(event_awayjet.pt)==1) & req_trg

            elif self._method=="System8":

                awayjet_sel = ak.fill_none( (ak.all(events.Jet.metric_table(event_mujet)>0.05,axis=2)) & (events.Jet.pt>=20.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)), False, axis=-1 )
                event_awayjet = events.Jet[ awayjet_sel ]

                req_sel = (ak.num(event_softmu.pt)==1) & (ak.num(event_mujet.pt)==1) & (ak.num(event_awayjet.pt)>=1) & req_trg

        ## Apply all selections
        event_level = ( req_lumi & req_sel )

        ##<==== finish selection
        event_level = ak.fill_none(event_level, False)

        # Skip empty events -
        if len(events[event_level]) == 0:
            return {dataset: output}

        ##===>  Ntuplization  : store custom information
        ####################
        # Selected objects # : Pruned objects with reduced event_level
        ####################
        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]

        if "workingPoints" in self.tag:

            pruned_ev["SelJet"] = event_jet[event_level]

        else:

            if "Light" in self.tag:

                light_jets = light_jet[event_level]
                away_jet_cand = events.Jet[ ak.fill_none( (events.Jet.pt>=20.) & (abs(events.Jet.eta)<2.5) & (jet_id(events, self._campaign)), False, axis=-1 ) ]
                away_jet = away_jet_cand[event_level]
                trkj_evt = trkj[event_level]
                trkj_evt["eta"] = trkj_evt.pf.trkEta
                trkj_evt["phi"] = trkj_evt.pf.trkPhi

                jet_zeros = ak.zeros_like(light_jets.pt, dtype=int)
                light_jets["jetPtBin"] = jet_zeros
                light_jets["nTrkInc"] = jet_zeros
                light_jets["jetWeight"] = ak.ones_like(light_jets.pt)
                light_jets["relativeEffectiveLuminosity"] = ak.zeros_like(light_jets.pt)
                for iptbin, ptbin in enumerate(self.jetPtBins):
                    triggerCut = ak.values_astype( HLT_helper(pruned_ev, [ self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["jetTrigger"] ] ), int,)
                    minPtCut, maxPtCut = float(self.jetPtBins[ptbin]["jetPtRange"][0]), float(self.jetPtBins[ptbin]["jetPtRange"][1])
                    awayPtCut = float(self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["ptAwayJet"])
                    emulPtCut = float(self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["ptTriggerEmulation"])
                    muPtCut, muDRCut = self.jetPtBins[ptbin]["muPtCut"], self.jetPtBins[ptbin]["muDRCut"]
                    light_jets["jetPtBin"] = ak.where( (triggerCut) & (light_jets.pt>=minPtCut) & (light_jets.pt<maxPtCut) & (ak.count_nonzero(light_jets.metric_table(away_jet[away_jet.pt>=awayPtCut])>=1.5,axis=2)>=1) & (ak.count_nonzero(light_jets.metric_table(away_jet[away_jet.pt>=emulPtCut])>=0.05,axis=2)>=1) & (ak.count_nonzero(light_jets.metric_table(trkj_evt[trkj_evt.pf.trkPt>=muPtCut])<=muDRCut,axis=2)>=1), jet_zeros+1+iptbin, light_jets["jetPtBin"] )
                    light_jets["nTrkInc"] = ak.where( (light_jets.pt>=minPtCut) & (light_jets.pt<maxPtCut), ak.count_nonzero(light_jets.metric_table(trkj_evt[trkj_evt.pf.trkPt>=muPtCut])<=muDRCut,axis=2), light_jets["nTrkInc"])
                    light_jets["relativeEffectiveLuminosity"] = ak.where( (light_jets.pt>=minPtCut) & (light_jets.pt<maxPtCut), jet_zeros+self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["jetRelativeEffectiveLuminosity"], light_jets["relativeEffectiveLuminosity"])

                if not isRealData:
                    light_jets["jetPtBin"] = ak.where( light_jets.pt<pthat_safety_cut(self.ptHatSafetyCuts, pruned_ev.Generator.binvar), light_jets["jetPtBin"], jet_zeros )

                if "Templates" in self.tag:

                    pair = ak.cartesian( [light_jets, trkj_evt], axis=1, nested=True )  # dim: (event, jet, pfcand)
                    matched = (pair["0"].delta_r(pair["1"].pf)<0.5) #& (pair["0"].pt_orig == pair["1"].jet.pt)
                    trkj_jetbased = pair["1"][matched]  # dim: (event, jet, pfcand)

                    vec = ak.zip( {"pt": trkj_jetbased.pf.trkPt, "eta": trkj_jetbased.pf.trkEta, "phi": trkj_jetbased.pf.trkPhi, "mass": trkj_jetbased.pf.mass,}, 
                             behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector", )
                    trkj_jetbased["ptrel"] = (vec.subtract(light_jets)).cross(light_jets).p/light_jets.p
                    
                    trk_zeros = ak.zeros_like(trkj_jetbased.ptrel, dtype=int)
                    trkj_jetbased["jetPtBin"] = light_jets["jetPtBin"]
                    trkj_jetbased["nTrkInc"] = light_jets["nTrkInc"]
                    for iptbin, ptbin in enumerate(self.jetPtBins):
                        minPtCut, maxPtCut = float(self.jetPtBins[ptbin]["jetPtRange"][0]), float(self.jetPtBins[ptbin]["jetPtRange"][1])
                        muPtCut, muDRCut = self.jetPtBins[ptbin]["muPtCut"], self.jetPtBins[ptbin]["muDRCut"]
                        trkj_jetbased["jetPtBin"] = ak.where( (light_jets.pt>=minPtCut) & (light_jets.pt<maxPtCut) & ((trkj_jetbased.pf.trkPt<muPtCut) | (vec.delta_r(light_jets)>muDRCut)), trk_zeros, trkj_jetbased["jetPtBin"] )

            else:

                pruned_ev["SelMuo"] = event_softmu[event_level][:, 0]
                pruned_ev["SelJet"] = event_mujet[event_level][:, 0]
                pruned_ev["AwayJet"] = event_awayjet[event_level][:, 0]
                if "pTrel" in self.tag:
                    pruned_ev["EmulJet"] = event_emuljet[event_level][:, 0]

                mujet_zeros = ak.zeros_like(pruned_ev["SelJet"].pt)
                pruned_ev["jetPtBin"] = ak.zeros_like(pruned_ev["SelJet"].pt)
                pruned_ev["relativeEffectiveLuminosity"] = mujet_zeros
                for iptbin, ptbin in enumerate(self.jetPtBins):
                    triggerCut = ak.values_astype( HLT_helper(pruned_ev, [ self.jetPtBins[ptbin]["trigger"] ] ), int,)
                    minPtCut, maxPtCut = float(self.jetPtBins[ptbin]["jetPtRange"][0]), float(self.jetPtBins[ptbin]["jetPtRange"][1])
                    awayPtCut = float(self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["ptAwayJet"])
                    muPtCut, muDRCut = self.jetPtBins[ptbin]["muPtCut"], self.jetPtBins[ptbin]["muDRCut"]

                    if self._method=="pTrel":
                        emulPtCut = float(self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["ptTriggerEmulation"])
                        pruned_ev["jetPtBin"] = ak.values_astype( pruned_ev["jetPtBin"] + (iptbin+1)*triggerCut*((pruned_ev.SelJet.pt>=minPtCut) & (pruned_ev.SelJet.pt<maxPtCut) & (pruned_ev.AwayJet.pt>=awayPtCut) & (pruned_ev.EmulJet.pt>=emulPtCut) & (pruned_ev.SelMuo.pt>=muPtCut) & (pruned_ev.SelMuo.delta_r(pruned_ev.SelJet)<=muDRCut)), int,)
                    elif self._method=="System8":
                        pruned_ev["jetPtBin"] = ak.values_astype( pruned_ev["jetPtBin"] + (iptbin+1)*triggerCut*((pruned_ev.SelJet.pt>=minPtCut) & (pruned_ev.SelJet.pt<maxPtCut) & (pruned_ev.AwayJet.pt>=awayPtCut) & (pruned_ev.SelMuo.pt>=muPtCut) & (pruned_ev.SelMuo.delta_r(pruned_ev.SelJet)<=muDRCut)), int,)

                    pruned_ev["relativeEffectiveLuminosity"] = ak.where( (pruned_ev.SelJet.pt>=minPtCut) & (pruned_ev.SelJet.pt<maxPtCut), mujet_zeros+self.triggerInfos[self.jetPtBins[ptbin]["trigger"]]["relativeEffectiveLuminosity"], pruned_ev["relativeEffectiveLuminosity"])

                if not isRealData:
                    pruned_ev["jetPtBin"] = ak.values_astype( pruned_ev["jetPtBin"]*(pruned_ev["SelJet"].pt<pthat_safety_cut(self.ptHatSafetyCuts, pruned_ev.Generator.binvar)), int,)

                if "Kinematics" in self.tag:
                    pruned_ev["PV"] = events.PV[event_level]
                    pruned_ev["muJetDR"] = pruned_ev.SelMuo.delta_r(pruned_ev.SelJet)

                elif "Templates" in self.tag:   
                    pruned_ev["ptrel"] = pruned_ev.SelMuo.cross(pruned_ev.SelJet).p/pruned_ev.SelJet.p
                    if isRealData:
                        pruned_ev["jetFlavour"] = ak.zeros_like(pruned_ev["SelJet"].pt, dtype=int)
                    else:
                        pruned_ev["jetFlavour"] = pruned_ev["SelJet"].hadronFlavour
                    wp_dict_campaign = btag_wp_dict[self._year+"_"+self._campaign]
                    for tagger in wp_dict_campaign:
                        pruned_ev[tagger] = ak.zeros_like(pruned_ev["SelJet"].pt)
                        for wp in wp_dict_campaign[tagger]["b"]:
                            if wp!="No":
                                pruned_ev[tagger] = ak.values_astype( pruned_ev[tagger] + (pruned_ev.SelJet["btag"+tagger+"B"]>wp_dict_campaign[tagger]["b"][wp]), int,)
                    if self._method=="pTrel" and len(list(wp_dict_campaign.keys()))==1:
                        bpruned_ev["bCorrector"] = ak.values_astype( (pruned_ev.SelJet[self.bCorrectorDiscriminant]>=self.bCorrectorCut), int,)
                    if self._method=="System8":
                        pruned_ev["taggedAwayJet"] = ak.values_astype( (pruned_ev.AwayJet[self.btagAwayJetDiscriminant]>=self.btagAwayJetCut), int,)

        ## <========= end: store custom objects
        ####################
        #     Output       #
        ####################
        # Configure SFs
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        if "Light" in self.tag: 
            light_jets["jetWeight"] = ak.values_astype( light_jets["jetWeight"]*(light_jets["jetPtBin"]>=1), float,)
            sample = "QCD_sf/"
        else:
            weights.add("jetbinsel", (pruned_ev["jetPtBin"]>=1))
            sample = "QCD_smu_sf/"
        if isRealData: sample += "data"
        else: sample += "MC"
        # Prescales
        if "data" in sample: 
            if "NoPS" not in self.tag: 
                if "QCD_smu_sf" in sample:
                    weights.add("psweight", get_psweight(self.jetPtBins, self.ps_run_num, pruned_ev["jetPtBin"], pruned_ev.run, pruned_ev.luminosityBlock))
                elif "QCD_sf" in sample:
                    light_jets["jetWeight"] = ak.values_astype( light_jets["jetWeight"]*get_psweight(self.jetPtBins, self.ps_run_num, light_jets["jetPtBin"], pruned_ev.run, pruned_ev.luminosityBlock, self.triggerInfos), float,)
        elif sample=="QCD_sf/MC":
            if "NoPS" in self.tag: light_jets["jetWeight"] = ak.values_astype( light_jets["jetWeight"]*light_jets["relativeEffectiveLuminosity"], float,)
        # b-jet templates uncertainties
        elif sample=="QCD_smu_sf/MC":
            if "NoPS" in self.tag: weights.add("efflumi", pruned_ev["relativeEffectiveLuminosity"])
            # b-jet templates uncertainties
            if "Templates" in self.tag:
                is_heavy_hadron = lambda p, pid: (abs(p.pdgId) // 100 == pid) | ( abs(p.pdgId) // 1000 == pid )
                sel_bhadrons = is_heavy_hadron(pruned_ev.GenPart, 5) & (pruned_ev.GenPart.hasFlags("isLastCopy")) & (ak.all(pruned_ev.GenPart.metric_table(pruned_ev.SelJet)<0.5,axis=2))
                bhadrons = pruned_ev.GenPart[sel_bhadrons]
                BHadron = ak.zip( { "pT": bhadrons.pt,
                                    "eta": bhadrons.eta,
                                    "phi": bhadrons.phi,
                                    "pdgID": bhadrons.pdgId,
                                    "mass": get_hadron_mass(bhadrons.pdgId),
                                    "hasBdaughter": ak.values_astype(
                                       ak.any(is_heavy_hadron(bhadrons.children, 5), axis=-1), int
                                     ),  # B hadrons with B-daughters not removed
                                   } )
                lastBHadron = BHadron[ BHadron.hasBdaughter==0 ]
                # Gluon splitting
                gluonSplitting     = ak.values_astype( 1.0*(ak.num(lastBHadron)<2) + 1.0*(ak.num(lastBHadron)>=2), float,)
                gluonSplittingUp   = ak.values_astype( 1.0*(ak.num(lastBHadron)<2) + 1.5*(ak.num(lastBHadron)>=2), float,)
                gluonSplittingDown = ak.values_astype( 1.0*(ak.num(lastBHadron)<2) + 0.5*(ak.num(lastBHadron)>=2), float,)
                weights.add("gluonSplitting", gluonSplitting, gluonSplittingUp, gluonSplittingDown)
                # b-hadron fragmentation
                genJetPt = ak.values_astype( ak.sum(pruned_ev.GenJet.pt*ak.all(pruned_ev.GenJet.metric_table(pruned_ev.SelJet)<0.5,axis=2),axis=-1), float,)
                xB = ak.values_astype( (ak.num(BHadron)>0)*ak.sum(BHadron.pT*(BHadron.mass==ak.max(BHadron.mass, axis=-1))/genJetPt,axis=-1), float,)
                #xB = ak.values_astype( (ak.num(BHadron)>0)*BHadron.pT*(BHadron.mass==ak.max(BHadron.mass, axis=-1))/genJetPt, float,)
                bfragweight, bfragweightUp, bfragweightDown = get_bfragmentation_weight(xB, genJetPt, self._year+"_"+self._campaign) 
                weights.add("bfragmentation", bfragweight, bfragweightUp, bfragweightDown)
                bHadronId = ak.values_astype( -1*(ak.num(lastBHadron)!=1) + (ak.num(lastBHadron)==1)*ak.sum(lastBHadron.pdgID, axis=-1), float,)
                bdecayweight, bdecayweightUp, bdecayweightDown = get_decay_weight(bHadronId, self._year+"_"+self._campaign)
                weights.add("bdecay", bdecayweight, bdecayweightUp, bdecayweightDown)
        # Kinematic corrections
        if "-" in self.tag and sample!="QCD_smu_sf/data":
            for level in [ "-".join( self.tag.split("-")[1:x] ) for x in range(2,len(self.tag.split("-"))+1) ]:
                if sample=="QCD_smu_sf/MC":
                    weights.add(level.split("-")[-1], get_kinematic_weight(pruned_ev.SelJet.pt, pruned_ev.SelJet.eta, self._method, self._year+"_"+self._campaign, sample, level))
                else:
                    light_jets["jetWeight"] = ak.values_astype( light_jets["jetWeight"]*get_kinematic_weight(light_jets.pt, light_jets.eta, self._method, self._year+"_"+self._campaign, sample, level), float,)

        if "Light" in self.tag:
            pruned_ev["SelJet"] = light_jets
            if "Templates" in self.tag:
                trkj_jetbased["jetWeight"] = light_jets["jetWeight"]
                pruned_ev["TrkInc"] = trkj_jetbased

        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]
        if not isRealData:
            #pruned_ev["weight"] = weights.weight()
            for ind_wei in weights.weightStatistics.keys():
                pruned_ev[f"{ind_wei}_weight"] = weights.partial_weight(
                    include=[ind_wei]
                )
        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays
        if self.isArray:
            array_writer(self, pruned_ev, events, systematics[0], dataset, isRealData)

        return {dataset: output}

    ## post process, return the accumulator, compressed
    def postprocess(self, accumulator):
        return accumulator
