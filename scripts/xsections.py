import os
import argparse
import json
from BTVNanoCommissioning.utils.sample import predefined_sample

parser = argparse.ArgumentParser(description="Compute cross sections")

parser.add_argument(
    "-wf",
    "--from_workflow",
    help="Use the predefined workflows",
    required=True,
    default=None,
)

parser.add_argument(
    "--DAS_campaign",
    help="campaign info, specifying dataset name in DAS",
    default=None,
    required=True,
    type=str,
)

parser.add_argument(
    "--process",
    help="List to process to analyze, leave None to select all",
    default=None,
    type=str,
)

parser.add_argument(
    "--cmspath",
    help="Path to cmsrel for computing the cross section",
    default="CMSSW_15_0_4"
)

parser.add_argument(
    "--energy",
    help="CME of the processes",
    default="13p6"
)

parser.add_argument(
    "--nfiles",
    help="Number of files to use for computing the cross section",
    default=10,
    type=int,
)

parser.add_argument(
    "--fetch",
    help="Get input files from fecth instead than from DAS",
    default=False,
    action="store_true"
)

parser.add_argument(
    "--campaign",
    help="Campaign",
    default=None
)

def runCommand(cmd):
    result = os.popen(cmd).read().split("\n")
    result.remove("")
    return result

def xSection(files, cmspath, process):

    script = "+scriptxsection"+process+".sh"
    os.system("rm -f "+script)
    with open(script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n")
        f.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
        f.write("export CMSSWDIR="+cmspath+"/src\n")
        f.write("cd $CMSSWDIR\n")
        f.write("eval `scramv1 ru -sh`\n")
        f.write("ulimit -c 0\n")
        f.write("cd "+os.getenv("PWD")+"\n")
        f.write("cmsRun $CMSSWDIR/ana.py inputFiles=\""+files+"\" maxEvents=-1\n")

    xsec, xsec_unc, eq_lumi, frac_neg_w = "-1.", "-1.", "-1.", "-1."

    if os.path.isfile(script):

        os.system("chmod a+x "+script)
        xsec_output = runCommand("./"+script+" 2>&1")
        os.system("rm "+script)

        for line in xsec_output:
            if "After filter: final cross section" in line: 
                xsec = line.split(" ")[6]
                xsec_unc = line.split(" ")[8]
            elif "After filter: final equivalent lumi" in line: 
                eq_lumi = line.split(" ")[10]
            elif "After filter: final fraction of events with negative weights" in line:
                frac_neg_w = line.split(" ")[10]

    return xsec, xsec_unc, eq_lumi, frac_neg_w

    #After filter: final cross section = 4.162e+08 +- 3.416e+05 pb
    #After filter: final fraction of events with negative weights = 0.000e+00 +- 0.000e+00
    #After filter: final equivalent lumi for 1M events (1/fb) = 2.403e-06 +- 3.108e-09

if __name__ == "__main__":

    args = parser.parse_args()

    if args.fetch and args.campaign==None:
        print("xsections: a campaign must be specified when chosing fetch to get the input files")
        exit()

    if args.from_workflow not in predefined_sample:
        print("xsections: workflow", args.from_workflow, "not valid")
        exit()

    processList = predefined_sample[args.from_workflow]["MC"] if args.process==None else args.process.split(",")

    if len(processList)==0:
        print("xsections: no process found for workflow", args.from_workflow)  
        exit()

    os.system("cmsrel "+args.cmspath+"; cd "+args.cmspath+"/src; curl https://raw.githubusercontent.com/cms-sw/genproductions/master/Utilities/calculateXSectionAndFilterEfficiency/genXsec_cfg.py -o ana.py")

    xsec_dictionary_name = args.from_workflow if args.process==None else args.from_workflow+"__"+args.process
    xsec_dictionary = open(xsec_dictionary_name+".py", "x")

    for process in processList:

        nanodatasetList = runCommand("dasgoclient -query=\"instance=prod/global dataset=/"+process+"/"+args.DAS_campaign+"/NANOAODSIM\"")
        if len(nanodatasetList)==0:
            print("xsections: nanoaod dataset not found for process", process)
            continue
        elif len(nanodatasetList)>1:
            print("xsections: more than one nanoaod dataset available for process", process, ":", nanodatasetList)
            continue
        else: nanodataset = nanodatasetList[0]

        minidatasetList = runCommand("dasgoclient -query=\"instance=prod/global parent dataset="+nanodataset+"\"")

        if args.fetch:
            fetch_command_list = [ "echo \""+minidatasetList[0]+"\"&>tmp", "python scripts/fetch.py -c "+args.campaign+" -i tmp -o tmp.json" ]
            os.system(" ; ".join(fetch_command_list))
            with open("metadata/"+args.campaign+"/tmp.json", "r") as metadata:
                fileDic = json.loads(metadata.read())
                fileList = fileDic[process]
            os.system("rm metadata/"+args.campaign+"/tmp.json")
        else:
            fileList = runCommand("dasgoclient -query=\"instance=prod/global file dataset="+minidatasetList[0]+"\"")
        
        xsec, xsec_unc, eq_lumi, frac_neg_w = xSection(",".join(fileList[0:args.nfiles]), args.cmspath, process)

        xsec_dictionary.write("    {\n")
        xsec_dictionary.write("        \"process_name\": \""+process+"\",\n")
        xsec_dictionary.write("        \"DAS\": \""+minidatasetList[0]+"\",\n")
        xsec_dictionary.write("        \"cross_section\": \""+xsec+"\",\n")
        xsec_dictionary.write("        \"total_uncertainty\": \""+xsec_unc+"\",\n")
        xsec_dictionary.write("        \"energy\": \""+args.energy+"\",\n")
        xsec_dictionary.write("        \"fraction_negative_weight\": \""+frac_neg_w+"\",\n")
        xsec_dictionary.write("        \"equivalent_lumi\": \""+eq_lumi+"\",\n")
        xsec_dictionary.write("        \"comment\": \"from GenXSecAnalyzer\",\n")
        xsec_dictionary.write("    },\n")

    os.system("rm -rf "+args.cmspath)

