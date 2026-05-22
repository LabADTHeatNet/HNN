# Response to Reviewer 2

Dear Reviewer,

We thank you for the detailed and practically oriented comments. We have revised the manuscript accordingly; the changes in the manuscript are marked in blue.

Comments 1: The whole problem is linked to a certain level of loads and material strength in the DH-system. Weak points must have been identified already by doing the pipe statics in the planning phase. Of course this can also be done at a later stage. Including this would sharpen the view on the leakage points. AI could do that for you.

Response 1: We agree that load level, material strength, pipe statics, and weak-point assessment are important for practical predictive maintenance. We added this context to the Introduction and clarified in the Conclusions that standard-based risk estimates and operating-history data can be used as prior information together with the proposed GNN localization model.

Comments 2: Since the leakage problem is very much linked to the load level and the pipe system that will be used the introduction must explain both. There is an international definition on DH-generations that could be used, but even more information would be helpful. Furthermore I recommend to give some insights in the introduction about the role of leakage detection wires in the pipe systems. This is also missing.

Response 2: Thank you. We added a new paragraph in the Introduction explaining that DHN fault risk is linked to load level, pipe-system design, material strength, insulation condition, and operating regime. We also added a short explanation of leakage-detection wires in pre-insulated pipe systems and clarified that our method complements such installed monitoring systems by using sparse pressure/temperature measurements and network topology.

Comments 3: I recommend to define the type of District Heating Networks according to standards. A very good one would be the GOST 55596-2013 "District Heating Networks" of the Russian Federation. There all the relevant problems for faults are explained and you may calculate fault risks accordingly. In combination with the GNN approach you will have a very good accuracy. Fatigue etc. will be covered. Neglecting the state of the art of design weakens your method and today computers can combine the information you receive.

Response 3: We agree and added references to relevant design and pipe-system standards in the Introduction, including GOST R 55596-2013, GOST 30732-2006, EN 13941, and EN 253. The present model does not yet use fatigue/stress calculations as input features, because such historical design and condition data were not available in the simulated dataset. We now explicitly identify standard-based risk estimates as a natural extension for predictive-maintenance workflows.

Comments 4: Actually I am wondering why you do not use standards for your investigation. GOST of 30732-2006 would explain the pipe system, GOST 25.101-83 and SNIP 41-02-2003. would also be helpful. Also EN13941 and EN253 could be used for the definition of load level and the pipe system. Why don't you use data on water losses?

Response 4: We added standard references and clarified the scope. The current dataset models insulation/thermal degradation through an increased heat-transfer coefficient, not actual mass leakage. Therefore, water-loss measurements are not part of the simulated target and no water-loss term is included in the mass-balance equation. We added this clarification in Section 2.4 and noted that historical water-loss records can be integrated as additional prior information in future work.

Comments 5: line 3: Is it leak oder leakage?

Response 5: We revised the terminology and now use "fault localization" for the task studied in this paper. Where leakage is mentioned, it refers to related literature or operational context, not to the simulated target variable.

Comments 6: line 44: measurement of noise

Response 6: We revised the wording to "measurement noise" and clarified the noise simulation procedure in Section 2.4.

Comments 7: line 58: Do you have a reference for this hypothesis?

Response 7: We strengthened the Introduction with additional references on DHN reliability, topology, and monitoring limitations, including Novitsky et al., Tereshchenko and Nord, Postnikov and Stennikov, and Sarbu et al.

Comments 8: line 99: The research gab could be formulated more clearly.

Response 8: We revised the Introduction to state the gap more explicitly: existing data-driven and physics-based approaches are difficult to apply in DHNs with sparse SCADA-like measurements, while topology-aware GNNs can exploit network structure under limited sensing.

Comments 9: line 198: The water losses due to leakage does influence the formula. m will be changed.

Response 9: We agree for actual mass leakage. The present study does not simulate mass leakage; it simulates a thermal fault caused by increased heat transfer. We added an explicit statement that a true leakage case would require an additional source/sink term in the mass-balance equation and changed branch mass flows.

Comments 10: line 232: The alpha variation is not well explained. Please explain better the methodology and the reason for the numbers chosen.

Response 10: We expanded the dataset-generation description. The revised text explains that 10--20% of randomly selected pipes receive up to 5% random variation in $\alpha$ to emulate environmental and measurement noise, while fault cases use five stronger degradation ranges: 120--150%, 150--170%, 170--200%, 200--300%, and 300--400% of the baseline $\alpha$.

Comments 11: line 236: What is consumer modeling pipes?

Response 11: We clarified this in Section 2.4. Consumer-modeling pipes are specialized graph edges connecting the supply and return sub-networks and representing consumer-side hydraulic resistance and heat extraction.

Comments 12: line 256: Figure 2: The network looks like a subnetwork or a low temperature network. All chosen values for the boundary conditions line 258 ff. are not given. This is a weak point in a strong paper. Figure 2: Legend: A customer is also a heat sink in my understanding.

Response 12: We revised the case-study description and redrew Figure 2 with clearer labels. The text now states that the modeled system is a local segment of an urban DHN in Arkhangelsk Oblast and describes the boundary-condition structure: source supply flow rate and temperature, fixed return pressure, and an outdoor-temperature curve used to scale source and consumer demands.

Comments 13: line 304: More information on the Adam optimizer would be helpful. Please give a reference.

Response 13: We added the Adam reference and kept the optimizer parameters in Section 2.6: learning rate, $\beta$, $\varepsilon$, weight decay, and scheduler settings.

Comments 14: line 345: How large is the expected search area? Is the accuracy better than using standard leakage detection wires?

Response 14: We added both topological and metric-distance interpretations of the localization error. The distance-confusion matrix shows that most errors are within 1--2 graph sections, and the new metric-distance figure reports physical path distance along the network. We do not claim that the method replaces leakage-detection wires; rather, it complements such systems by narrowing the inspection area from sparse SCADA-like data.

Comments 15: line 346: Figure 6: Please put (a) and (b) into the figures or use (left) and (right). What meand "True"? True failures? This is misleading.

Response 15: We revised the visualization layout by placing the supply and return plots one above the other with explicit subfigure labels. We also added confusion matrices and clarified that the distribution plot is a marginal count diagnostic and should not be interpreted as class-wise accuracy by itself.

Comments 16: line 370: Conclusion: Please give an outlook how todays Predictive maintenance methods of DH-systems can be improved.

Response 16: We added an outlook paragraph in the Conclusions. It explains how the proposed GNN can be used as an additional ranking layer in predictive maintenance, together with standard-based risk estimates, historical water-loss records, leakage-wire alarms, and operating-history data.

Comments 17: References: The year of reference 34 and 37 must be put in bold font.

Response 17: We checked and regenerated the bibliography. The MDPI bibliography style now formats publication years in bold consistently.
