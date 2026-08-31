import hist as Hist
from BTVNanoCommissioning.utils.selection import btag_wp_dict

def get_histograms(axes, **kwargs):

    hists = {}
   
    workflow = kwargs.get("workflow", "")
    year = kwargs.get("year", "")
    campaign = kwargs.get("campaign", "")

    if "workingPoints" in workflow:
        for tagger in btag_wp_dict[year+"_"+campaign]:
            hists["btagdisc_"+tagger] = Hist.Hist(axes["syst"], axes["flav"], axes["btagdisc"], Hist.storage.Weight())

    elif "pTrel" in workflow or "System8" in workflow:
        
        if "Kinematics" in workflow:
            hists["jetpt"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["jetpt"],  Hist.storage.Weight())
            hists["jeteta"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["jeteta"], Hist.storage.Weight())

            if "Light" not in workflow:
                if "Optimization" not in workflow:
                    hists["nPV"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["npv"], Hist.storage.Weight())
                    hists["DR"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["mujetdrkin"], Hist.storage.Weight())
                    hists["muopt"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["muopt"], Hist.storage.Weight())

                else:
                    hists["muonDR"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["ptrelcut"], axes["mujetdrkin"], Hist.storage.Weight())
                    hists["ptrel"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["mujetdrcut"], axes["ptrel"], Hist.storage.Weight())
                    hists["awayJetBTagDiscriminant"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["awayjetdrcut"], axes["awayjetbtagdisc"], Hist.storage.Weight())
                    hists["awayJetDR"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["awayjetbtagcut"], axes["awayjetdr"], Hist.storage.Weight())

        elif "Templates" in workflow:

            if "Light" in workflow:
                hists["ptrel"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["mujetdr"], axes["ptrel"], Hist.storage.Weight())

            else:
                if "pTrel" in workflow:
                    for tagger in btag_wp_dict[year+"_"+campaign]:
                        hists["ptrel_"+tagger] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["mujetdr"], axes["ptrel"], axes["btagwp"], Hist.storage.Weight())
                    if len(list(btag_wp_dict[year+"_"+campaign].keys()))==1:
                        hists["ptrel_bCorrector"] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["mujetdr"], axes["ptrel"], axes["btagwp"], Hist.storage.Weight())

                elif "System8" in workflow:
                    for tagger in btag_wp_dict[year+"_"+campaign]:
                        hists["ptrel_"+tagger] = Hist.Hist(axes["syst"], axes["ptbin"], axes["flav"], axes["ptrel"], axes["btagwp"], axes["tagawj"], Hist.storage.Weight())

    return hists


def qg_writer(
    events,
    output,
    weights,
    systematics: list,
    isSyst: bool,
    SF_map: dict,
):
    for syst in systematics:
        if not isSyst and syst != "nominal":
            break
        weight = (
            weights.weight()
            if syst == "nominal" or syst not in list(weights.variations)
            else weights.weight(modifier=syst)
        )
        # weight = weight * weights.partial_weight(include=["psweight"])

        for histname, hist in output.items():
            if "Var" not in histname or "Obj" not in histname:
                continue
            hobj = histname.split("_Var")[0].replace("Obj", "")
            var = histname.split("_Var")[1].split("_")[0]
            is_pteta = histname.endswith("_pteta")
            if hobj not in events.fields:
                continue
            if var not in events[hobj].fields:
                continue

            obj_axes = {
                "syst": syst,
                var: ak.flatten(events[hobj][var], axis=None),
                # "weight": weight,
            }
            if is_pteta:
                obj_axes["pt"] = ak.flatten(events[hobj]["pt"], axis=None)
                obj_axes["eta"] = ak.flatten(np.abs(events[hobj]["eta"]), axis=None)

            if hobj != "Tag":
                if "partonFlavour" not in events[hobj].fields:
                    obj_axes["flav"] = ak.zeros_like(
                        ak.flatten(events[hobj].pt, axis=None), dtype=int
                    )
                else:
                    obj_axes["flav"] = ak.flatten(
                        _flavor_label(events[hobj].partonFlavour), axis=None
                    )

            w = ak.flatten(ak.broadcast_arrays(weight, events[hobj][var])[0], axis=None)
            obj_axes["weight"] = w

            output[histname].fill(**obj_axes)

    return output
