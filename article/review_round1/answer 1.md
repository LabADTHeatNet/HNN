# Response to Reviewer 1

Dear Reviewer,

We thank you for the constructive comments. We have revised the manuscript accordingly; the changes in the manuscript are marked in blue.

Comments 1: The dataset contains 42,570 samples and 30 classes, including the normal state. However, the manuscript should provide more information on class balance, the number of samples per class, the number of fault severities per pipe/section, and whether operating conditions overlap between training and testing.

Response 1: Thank you for pointing this out. We expanded the dataset description in Section 2.4 and the split description in Section 2.6. The revised text now specifies the outdoor-temperature range, the number of defect-free, defective, and reference samples, the five defect-severity ranges, and the class balance after splitting the full graph into supply and return subnets. We also clarified that the train/validation/test split is random over generated operating states, so the evaluation measures interpolation within the simulated operating envelope rather than extrapolation to fully unseen climate regimes.

Comments 2: Although the ablation study compares different GNN configurations, the manuscript does not sufficiently compare the proposed method with non-GNN or simpler baselines, such as MLP, random forest, gradient boosting, CNN-like models, or physics-based residual localization.

Response 2: We agree. We added non-GNN baselines to Table 1: Random Forest, Histogram Gradient Boosting, and MLP. The proposed GNN outperforms all these topology-agnostic models, especially on the return subnet and Macro F1. We did not include a physics-based residual-localization baseline because the present dataset does not model actual mass leakage or measured field residuals; this limitation is now clarified in the dataset description.

Comments 3: The study retains pressure and temperature readings at only 30 of 187 nodes to represent sparse sensing. However, it is not clear why these 30 nodes were selected, whether they correspond to realistic SCADA locations, and how sensitive the model is to sensor placement.

Response 3: We revised Section 2.5 to explain the selected sparse-sensing layout. Dynamic pressure and temperature values are retained only at realistic monitoring points: the heat-source supply and return nodes and the supply/return connection points of the 14 consumers, for 30 nodes in total. We also added sensor-layout sensitivity as a future-work direction in the Conclusions.

Comments 4: The model is trained and tested entirely on simulated data. Although this is understandable due to the lack of labeled real-world faults, the manuscript should provide a more critical discussion of the simulation-to-reality gap, including model uncertainty, measurement noise, unmodeled operational control actions, pipe aging, unknown boundary conditions, and seasonal load variation.

Response 4: We agree. We added a limitation paragraph in the Conclusions explaining that the current thermohydraulic model is simplified and does not include shut-off/control valves, intermediate switching states, gradual pipe degradation, or the full domain shift caused by real seasonal and load variations. We also clarified the simulated measurement/noise assumptions in Section 2.4.

Comments 5: The phrase “fault localization” may be more appropriate than “leak localization” throughout the manuscript unless actual mass leakage is modeled.

Response 5: We agree and revised the terminology accordingly. The manuscript now consistently frames the task as fault localization. We also added an explicit clarification in Section 2.4 that the simulated defect is not a mass leak: no water-loss term is added to the mass-balance equation; instead, the fault is modeled as a local increase in heat transfer coefficient.

Comments 6: Figure 2 should be enlarged or redrawn with clearer labels, because the current network topology and legends are difficult to read.

Response 6: We redrew and enlarged Figure 2 using the updated network visualization. The revised figure uses a larger layout and clearer labels/legend for the DHN topology.
