# Response to Reviewer 3

Dear Reviewer,

We thank you for the detailed comments and for the constructive suggestions on the physical model and evaluation. We have revised the manuscript accordingly; the changes in the manuscript are marked in blue.

Comments 1: In Eq. (4), the authors correctly include the use of Darcy's friction factor. However, the relation used to calculate this variable (e.g., Colebrook-White) is not very clear. This factor is essential for calculating pressure drops, so it needs to be clarified.

Response 1: We agree. We revised Section 2.4 by adding the explicit Altshul correlation used to calculate the Darcy friction factor, together with definitions of wall roughness, dynamic viscosity, flow velocity, and branch Reynolds number. (lines 195-203, Equations 5-6).

Comments 2: The thermal model appears to use a single control volume per branch to evaluate heat losses and average branch temperature. While acceptable under quasi-steady conditions, the validity of this approximation depends on pipe length, diameter, water volume, and flow velocity. The authors should report the range of pipe dimensions and flow velocities/residence times, and justify why thermal transport delays and pipe thermal inertia can be neglected. Please also clarify whether a multi-volume discretization of long branches was considered, or why the single-volume approximation is sufficient for the simulated fault-location scenarios.

Response 2: Thank you. We revised Section 2.4 to clarify that the present simulator is quasi-stationary and that each sample represents a steady operating regime, not a transient evolution. Equation (10) is now described as a steady branch-integrated solution in which pipe length is accounted for through the heat-transfer area and the exponential attenuation factor. We also explicitly state that thermal transport delays, hot-water-front propagation, and pipe-wall thermal inertia are outside the scope of the present dataset. (lines 222-234).

Comments 3: Given the aim of the paper, reducing the number of nodes/elements to be considered from the total of 187 is entirely appropriate and reasonable. However, how did the authors decide which nodes to select, and why did they choose 30 nodes? In my opinion, it would add even more value to the paper to examine how this number affects the quality of the GNN model; identifying a potential plateau would be excellent.

Response 3: We revised Section 2.5 to explain the 30-node sparse-sensing configuration. The selected nodes correspond to realistic monitoring locations: the heat-source supply and return nodes and the supply/return connection points of the 14 consumers. We agree that a sensor-placement sensitivity study would be valuable, and we now mention testing different sensor layouts as future work. (lines 306-320, lines 512-515).

Comments 4: As mentioned in 2), the authors state that this analysis is based on a real DHN, but no further information is provided. At the very least, the region and a brief description of the case study should be included, because, given the nature of the topological analysis, it is important to know the network's dimensions, etc.

Response 4: We agree. Section 2.4 now states that the modeled case is a local segment of an urban DHN in Arkhangelsk Oblast, Russia. We also added a brief structural description: a compact residential district served by a central heat source, with consumer stations and interconnecting supply/return infrastructure, represented by 187 nodes and 202 pipe segments. (lines 253-257).

Comments 5: The authors correctly describe the training procedure; however, it is unclear whether k-fold cross-validation was performed. It would be necessary to specify and analyze the results for multiple random seeds and consider average values, since the results might otherwise depend on an unlucky (or lucky) seed.

Response 5: We agree. We added the cross-validation result in Section 2.6. The revised manuscript reports 3-fold cross-validation with consistent accuracy scores of 0.913±0.011 averaged over the two subnets, confirming that the reported performance is not tied to a single data split. (lines 367-369).

Comments 6: The evaluation metrics should be more clearly defined in the methodology section. While accuracy is relatively straightforward, the meaning and computation of F1-score and Macro-F1 may not be immediately clear to all readers in the context of this paper.

Response 6: We added metric definitions in the Results and Discussion section. The revised text defines precision, recall, F1-score, and Macro F1, and explains that Macro F1 is the unweighted average of per-class F1 values, making it more sensitive to rare classes than overall accuracy. (lines 394-399).

Comments 7: Figure 5 provides an interesting analysis of errors in terms of topological distance. However, topological distance alone may hide the actual physical magnitude of the localization error, since two graph hops can correspond to very different pipe lengths. The authors should consider also reporting the error in metric distance, e.g., meters along the network path, possibly using the branch midpoint/centroid as the representative fault position. This would make the localization performance more physically interpretable.

Response 7: We agree and added a new metric-distance analysis. The revised manuscript computes physical path distance along the network using pipe lengths as edge weights, with the predicted section midpoint and true defective-pipe midpoint as representative positions. The new Figure 6 reports this metric-distance localization error for both supply and return subnets. (lines 434-441, Figure 6).

Comments 8: In Fig. 5a, does the "1" in section 41 indicate that 100% of the faults in section 41 have been assigned to another section? Why? Please explain.

Response 8: We thank Reviewer for this comment. Interpretation, provided in comment, is correct. We added an explicit explanation in the Results and Discussion section. Segment 41 has 43 test samples, all of which were assigned to neighboring sections 32 or 36. Segment 41 consists of a single pipe, which yields fewer training examples than multi-pipe sections, and it is also topologically the farthest supply-side segment from the heat source. These factors attenuate the pressure/temperature signature and make this segment harder to distinguish. (lines 412-423, Figure 5).

Comments 9: Figure 6 should be discussed in more detail. The current text essentially repeats the caption, while the plot only shows true/predicted class counts. My understanding is that similar true and predicted counts for a given section do not necessarily imply correct localization, since compensating errors between sections may occur. For instance, using Section 18 as an example, some faults truly belonging to Section 19 could be predicted as Section 18, while some true faults from Section 18 could be predicted elsewhere. The authors should clarify this point and explain how Figure 6 should be interpreted in relation to the actual class-wise localization accuracy.

Response 9: We agree. We revised this part of the Results and Discussion section. The manuscript now explicitly states that the class-distribution plot is a marginal count diagnostic and should not be interpreted as class-wise localization accuracy by itself, because compensating errors are possible. We therefore added confusion matrices and discuss the distribution plot together with the actual class-to-class error structure. The revised text also comments on the main return-subnet ambiguity: segment 0 is often confused with nearby segments 1, 2, and 4, which we attribute to smoother hydraulic/thermal signatures in the remote return-side part of the network and to the presence of several long pipes in segment 0. (lines 442-463, Figure 7, Figure 8).

Comments 10: In lines 88-100, rather than listing the references at the beginning and then citing only the authors and year (ref. 27-31), it would be more appropriate to include the references when describing the contributions of the relevant papers.

Response 10: We revised the literature paragraph in the Introduction. The citations are now integrated directly into the discussion of each paper's contribution instead of being grouped at the beginning. (lines 99-110).

Comments 11: Lines 302-303 and 316-318 repeat the same information regarding the 70:15:15 dataset split and the 100-epoch training setup. Given the proximity of the two passages, this repetition should be avoided.

Response 11: We removed the repeated description from the Results section and kept the split and training setup in the Training subsection. (lines 365-379).

Comments 12: In Fig. 6, the text is difficult to read. It would be better to place the two graphs one above the other and enlarge them.

Response 12: We revised the figure layout by placing the supply and return plots vertically and increasing their size. The updated layout is consistent with the distance-matrix figure layout and improves readability. (Figure 7, Figure 8).
