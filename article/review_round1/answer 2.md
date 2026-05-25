# Response to Reviewer 2

Dear Reviewer,

We thank you for the detailed and practically oriented comments. We have revised the manuscript accordingly; the changes in the manuscript are marked in blue.

Comments 1: The whole problem is linked to a certain level of loads and material strength in the DH-system. Weak points must have been identified already by doing the pipe statics in the planning phase. Of course this can also be done at a later stage. Including this would sharpen the view on the leakage points. AI could do that for you.

Response 1: We agree that load level, material strength, pipe-system design, and pipe-static/fatigue assessment are important for identifying weak points in DHNs. We added this context to the Introduction, in the paragraph discussing engineering risk factors, standards, and leakage-detection wires, and clarified in the Conclusions how such information can complement the proposed GNN. These data were not available as input features in the present simulated dataset, but in practical use they can act as prior information, while the GNN ranks likely fault locations from sparse pressure/temperature measurements and network topology. (lines 36-45, lines 516-521).

Comments 2: Since the leakage problem is very much linked to the load level and the pipe system that will be used the introduction must explain both. There is an international definition on DH-generations that could be used, but even more information would be helpful. Furthermore I recommend to give some insights in the introduction about the role of leakage detection wires in the pipe systems. This is also missing.

Response 2: We revised the Introduction to explain that DHN fault localization depends on load level, pipe-system design, material strength, insulation condition, and operating regime. In the same Introduction paragraph, we added a discussion of leakage-detection wires in pre-insulated pipe systems and clarified that the proposed GNN does not replace such installed monitoring systems. Instead, it provides a complementary localization/ranking layer when dense instrumentation or complete condition-history data are unavailable. In Section 2.4, in the case-study description before Figure 2, we further clarified that the modeled object is a local hot-water supply/return DHN segment rather than a complete city-wide heat-supply system. (lines 36-45, lines 253-270, Figure 2).

Comments 3: I recommend to define the type of District Heating Networks according to standards. A very good one would be the GOST 55596-2013 "District Heating Networks" of the Russian Federation. There all the relevant problems for faults are explained and you may calculate fault risks accordingly. In combination with the GNN approach you will have a very good accuracy. Fatigue etc. will be covered. Neglecting the state of the art of design weakens your method and today computers can combine the information you receive.

Response 3: We added representative standards to the Introduction, including GOST R 55596-2013, GOST 30732-2006, GOST 25.101-83, SNiP 41-02-2003, EN 13941, and EN 253, and added the corresponding entries to the References. The revised Introduction text explains that these standards constrain load level, pipe-system design, material strength, insulation, and fatigue-related assessment. We also clarified in the Conclusions that these engineering assessments are complementary to the proposed GNN and can be combined with it as prior risk information in future hybrid workflows. The current model does not yet perform a standard-based risk calculation, because the required design, pipe-static, fatigue, and asset-condition data are outside the available dataset. (lines 39-41, lines 516-521, References lines 558-569).

Comments 4: Actually I am wondering why you do not use standards for your investigation. GOST of 30732-2006 would explain the pipe system, GOST 25.101-83 and SNIP 41-02-2003. would also be helpful. Also EN13941 and EN253 could be used for the definition of load level and the pipe system. Why don't you use data on water losses?

Response 4: We agree that standards and historical water-loss data are important sources of information for practical fault-risk assessment. We added references to the relevant GOST, SNiP, and EN standards in the Introduction and clarified in the Conclusions how such information can complement the GNN in a predictive-maintenance workflow. The manuscript also clarifies the water-loss part of the question in Section 2.4, immediately after the defect-severity description: the current dataset does not simulate mass leakage, but insulation/thermal degradation through an increased heat-transfer coefficient. Therefore, no water-loss term is added to the nodal mass-balance equation, and water-loss measurements are outside the current target variable. Historical water-loss records are now discussed in the Conclusions as a useful prior information source for future hybrid implementations. (lines 39-41, lines 285-290, lines 516-521, References lines 558-569).

Comments 5: line 3: Is it leak oder leakage?

Response 5: We revised the terminology in the Abstract and Keywords. The task studied in this paper is now described as "fault localization", and the Abstract refers to "leakage-related faults" rather than simply "leaks". This wording is more accurate because the simulated defect is not a mass leak; it is represented as increased heat transfer through the pipe wall/insulation. Where "leak" or "leakage" remains, it refers to related literature or broader operational context. (line 3, line 19, lines 285-290).

Comments 6: line 44: measurement of noise

Response 6: We revised the wording in the Introduction paragraph on physics-based methods and clarified the noise simulation procedure in Section 2.4. The manuscript now refers to measurement noise/measurement uncertainty and explains that environmental fluctuations and measurement uncertainty are simulated by applying random variations within 5% of the heat-transfer coefficient on a randomly selected subset of pipes. (line 55, lines 273-274).

Comments 7: line 58: Do you have a reference for this hypothesis?

Response 7: The comment refers to the Introduction statement that models performing well during development may be difficult to transfer to real DHN data. We strengthened this part of the Introduction and retained supporting references for this point, including studies on real-world fault-detection transfer, DHN optimization/topology, and leakage diagnosis under limited sensing. We also clarified, later in the Introduction, the role of network topology in the propagation of thermal and hydraulic disturbances. (lines 68-76, lines 85-98, References lines 588-596).

Comments 8: line 99: The research gab could be formulated more clearly.

Response 8: We revised the final part of the Introduction to state the research gap more explicitly, immediately before the transition to Materials and Methods. The revised text explains that most topology-aware leakage-localization evidence comes from water and gas networks or from settings whose sensing and data conditions do not directly match operational DHNs. The resulting gap is topology-aware fault localization for DHNs under sparse SCADA-like measurements and limited labeled field data. (lines 99-114).

Comments 9: line 198: The water losses due to leakage does influence the formula. m will be changed.

Response 9: We agree for actual mass leakage: in that case, the branch mass flow rate in the thermal equation would change. We clarified this in Section 2.4, immediately after the defect-scenario description. The present study does not simulate mass leakage; it simulates a thermal fault caused by increased heat transfer. We added an explicit statement that a true leakage case would require an additional source/sink term in the continuity equation and changed branch mass flows, and that this case is outside the current dataset. (lines 285-290).

Comments 10: line 232: The alpha variation is not well explained. Please explain better the methodology and the reason for the numbers chosen.

Response 10: We expanded the dataset-generation description in Section 2.4. The revised text explains that environmental fluctuations and measurement uncertainty are simulated by applying up to 5% random variation in the heat-transfer coefficient on 10--20% of randomly selected pipes, while fault cases use stronger degradation ranges: 120--150%, 150--170%, 170--200%, 200--300%, and 300--400% of the baseline heat-transfer coefficient. (lines 273-280).

Comments 11: line 236: What is consumer modeling pipes?

Response 11: We added an explicit definition in Section 2.4, immediately after the defect-severity ranges. Consumer-modeling pipes are auxiliary graph edges that connect the supply and return subnets through a consumer station and represent consumer-side hydraulic resistance and heat extraction. They are not treated as candidate distribution-pipeline fault classes in the present localization task. (lines 281-284).

Comments 12: line 256: Figure 2: The network looks like a subnetwork or a low temperature network. All chosen values for the boundary conditions line 258 ff. are not given. This is a weak point in a strong paper. Figure 2: Legend: A customer is also a heat sink in my understanding.

Response 12: The case-study description in Section 2.4 and Figure 2 have been revised. The manuscript now states, in the paragraph introducing the modeled network, that the system is a local hot-water supply/return DHN segment in Arkhangelsk Oblast, Russia, not a complete city-wide heat-supply system. We also clarified the legend interpretation in the same paragraph: consumer stations act as heat sinks in the thermal model, while the source and sink symbols denote the heat-source supply and return boundary nodes. The following boundary-condition paragraph describes the boundary-condition structure: specified source supply flow rate and temperature, fixed return pressure, and an outdoor-temperature curve used to scale source and consumer demands. We added a clarification that the boundary conditions are scenario-dependent values defined by this operating curve rather than a single fixed operating point; the outdoor-temperature range from -34 to +10 degrees C specifies the simulated load envelope. (lines 253-270, Figure 2).

Comments 13: line 304: More information on the Adam optimizer would be helpful. Please give a reference.

Response 13: We added a reference for the Adam optimizer in Section 2.6, Training, in the sentence introducing the optimizer. The optimizer settings were already specified in the same subsection, including the initial learning rate, beta values, epsilon, weight decay, and learning-rate scheduler parameters. (lines 374-379, Reference 45 line 632).

Comments 14: line 345: How large is the expected search area? Is the accuracy better than using standard leakage detection wires?

Response 14: We added both topological and metric-distance interpretations of the localization error in Section 3.1, Performance on DHN Data. The distance-confusion-matrix discussion now states that most errors are within 1--2 graph sections and that the initial inspection area can be limited to the predicted section and its one- to two-section topological neighbourhood. The following metric-distance analysis and Figure 6 additionally report physical path distance along the pipe network. We also added leakage-detection wires to the Introduction and clarified in the Conclusions that the proposed model does not replace standard leakage-detection wires. The method should be interpreted as a complementary localization/ranking layer based on sparse pressure/temperature measurements and topology. (lines 41-45, lines 424-441, lines 516-521, Figure 5, Figure 6).

Comments 15: line 346: Figure 6: Please put (a) and (b) into the figures or use (left) and (right). What meand "True"? True failures? This is misleading.

Response 15: We revised the visualization layout in Section 3.1. The former Figure 6 has been split into subnet-specific diagnostic figures: Figure 7 for the supply subnet and Figure 8 for the return subnet, each with explicit subfigure labels for the class distribution and normalized confusion matrix. We also clarified in the explanatory paragraph following these figures that "True" denotes the number of test samples whose simulated defect is located in the corresponding section, while "Predicted Right" and "Predicted Wrong" denote correct and incorrect predictions assigned to that section. The revised text further explains that the class-distribution panel is a marginal count diagnostic and should not be interpreted as class-wise localization accuracy by itself, because compensating errors between sections are possible. (lines 442-449, Figure 7, Figure 8).

Comments 16: line 370: Conclusion: Please give an outlook how todays Predictive maintenance methods of DH-systems can be improved.

Response 16: We added a predictive-maintenance outlook to the Conclusions. The revised paragraph explains that the proposed GNN can be used as an additional localization and ranking layer. Standard-based risk estimates, pipe age and material data, pipe-static or fatigue assessments, historical water-loss records, leakage-wire alarms, and operating-history data can define prior fault probabilities, while the GNN uses sparse SCADA-like pressure and temperature measurements together with network topology to narrow the inspection area. (lines 516-521).

Comments 17: References: The year of reference 34 and 37 must be put in bold font.

Response 17: We regenerated the References section using the MDPI bibliography style after updating the bibliography entries. The template formats journal-article years in bold, while some other reference types, such as in-proceedings entries, are formatted without a bold year according to the bibliography-style settings. We therefore kept the automatically generated journal-template formatting. (References lines 609-616).
