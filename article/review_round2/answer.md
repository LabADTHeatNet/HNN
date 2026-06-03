Dear Reviewer,

We thank you again for your thoughtful and constructive comments. We have carefully addressed each of the remaining points, and we have revised the manuscript accordingly.

Comment 1: The authors have correctly specified the location of the district, but it would be more helpful to understand the lengths of the pipes and the distances considered. At a minimum, a scale should be included in Figure 2. This would clarify the subsequent graphs and the comments in the results section. 

Response 1: We agree with the comment. We have replaced Figure 2.

Comment 2: I would like to express my gratitude to the authors for including Figure 6, as it serves to enhance the clarity of the presentation of the results. I agree with the authors' reasoning regarding the supply-side error caused by the drop in temperature due to sections 41, 32 and 36 being far from the heating plant. However, this would not explain why the model predicts sections 37 and 38 well. Graphically, it appears that section 36 is at 19 hops, whereas sections 37 and 38 are at approximately 17 hops. While I noticed that some explanations follow Figure 8, Figures 5 and 6 require clearer explanation, as they risk making the model appear weak. Alternatively, I suggest that the authors conduct further analyses to see if they can improve the model further.

Response 2: We thank the reviewer for this insightful observation. In the revised manuscript, we offer a more detailed analysis of the errors affecting segment 41 and discuss why segments 37 and 38 do not appear to suffer from similar confusion. The explanation, presented in the Results and Discussion section, is that segments 32, 36, and 41 share a single downstream sensor (the node incident to both the consumer edge and the pipe edge of segment 41). We suggest that this topology may create an ambiguity that does not arise for the subgraph containing segments 37-38. (lines 472-485)

Comment 3: Regarding the new Figures 7 and 8, I thank the authors for the added comment and for including the normalized confusion matrix. However, in these figures, rather than "Predicted wrong," the label in the legend should be called "False positives assigned to this section."

Response 3: We have corrected the legend in Figures 7 and 8. The label "Predicted Wrong" has been replaced with "False Positives" as requested.

Comment 4: The conclusions do not address the observed limitations. While it is true that many errors occur within a limited topological distance, errors on the order of hundreds of metres often occur too. I suggest briefly mentioning the analysis related to the distance of the errors. Furthermore, the issue of accuracy could be emphasised by discussing how this information could be used to determine where to place additional monitoring sensors in a DHN.

Response 4: We have extended the Conclusions. A new paragraph have been added that connects analysis of errors in segments 32, 36, 41 (where the largest metric errors tend to concentrate) to the assumption that adding one pressure sensor at the junctions of these segments may eliminate most of occuring missclassifications. This idea is further linked to the optimal sensor placement framework proposed by Rosich et al., suggesting that the confusion patterns revealed by our GNN can serve as a data-driven prior for identifying critical locations where additional sensors would be most beneficial. (lines 553-560)