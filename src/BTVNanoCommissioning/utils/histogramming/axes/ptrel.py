import hist

axes = {

    # Working points
    "btagdisc" : hist.axis.Regular(22000, -1.1, 1.1, name="btagdisc", label="b-tagging discriminant")

    # Kinematics
    "ptbin" : hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], name="ptbin", label="jet p_{T} bin")
    "jetpt" : hist.axis.Regular(1000, 0., 1000., name="jetpt", label="p_{T} [GeV]")
    "jeteta" : hist.axis.Regular(50, -2.5, 2.5, name="jeteta",label="#mu-jet #eta")
    "mujetdrkin" : hist.axis.Regular(20, 0, 0.5, name="mujetdrkin", label="DeltaR")
    "muopt" : hist.axis.Regular(20, 0., 100., name="muopt", label="mu p_{T} [GeV]")
    "ptrelcut" : hist.axis.IntCategory([0, 1], name="ptrelcut", label="pTrel")
    "mujetdrcut" : hist.axis.IntCategory([0, 1], name="mujetdrcut", label="DeltaR")
    "awayjetbtagdisc" : hist.axis.Regular(50, 0., 1., name="btagdisc", label="btagdisc")
    "awayjetdr" : hist.axis.Regular(30, 0., 3., name="awayjetdr", label="DeltaR")
    "awayjetbtagcut" : hist.axis.IntCategory([0, 1], name="awayjetbtagcut", label="awayjetbtagcut")
    "awayjetdrcut" : hist.axis.IntCategory([0, 1], name="awayjetdrcut", label="awayjetdrcut")

    # Templates
    "ptrel" : hist.axis.Regular(50, 0., 4., name="ptrel", label="p_{T}^{rel} [GeV]")
    "mujetdr" : hist.axis.Regular(16, 0, 0.4, name="mujetdr", label="DeltaR")
    #"btagwp" : hist.axis.IntCategory([0, 1, 2, 3, 4, 5], name="btagwp", label="Pass b-tag WP")
    "btagwp" : hist.axis.Regular(6, 0., 6., name="btagwp", label="Pass b-tag WP")
    "tagawj" : hist.axis.IntCategory([0, 1], name="tagawj", label="Tagged away jet")

}

