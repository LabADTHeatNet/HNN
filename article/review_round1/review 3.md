he article presents an innovative method for fault detection in district heating networks that can be applied to a limited number of sensors in the network. The approach, which uses GNN-based methods, avoids the need for highly complex physical models of the network or, conversely, a very large network of sensors and a vast amount of data. The methodology is explained in detail and is clear. However, there are some points that should be clarified further or improved before the article is ready for publication.

Since the purpose of the paper is to assess the accuracy of the proposed GNN-based fault-location algorithm using data generated from a physical network model, the physical reliability of the thermo-hydraulic simulator should be more clearly justified. In particular:

1)  In Eq. (4), the authors correctly include the use of Darcy’s friction factor. However, the relation used to calculate this variable (e.g., Colebrook-White) is not very clear. This factor is essential for calculating pressure drops, so it needs to be clarified.

2) The thermal model appears to use a single control volume per branch to evaluate heat losses and average branch temperature. While acceptable under quasi-steady conditions, the validity of this approximation depends on pipe length, diameter, water volume, and flow velocity. The authors should report the range of pipe dimensions and flow velocities/residence times, and justify why thermal transport delays and pipe thermal inertia can be neglected. Please also clarify whether a multi-volume discretization of long branches was considered, or why the single-volume approximation is sufficient for the simulated fault-location scenarios.

3) Given the aim of the paper, reducing the number of nodes/elements to be considered from the total of 187 is entirely appropriate and reasonable. However, how did the authors decide which nodes to select, and why did they choose 30 nodes? In my opinion, it would add even more value to the paper to examine how this number affects the quality of the GNN model; identifying a potential plateau would be excellent. 

4) As mentioned in 2), the authors state that this analysis is based on a real DHN, but no further information is provided. At the very least, the region and a brief description of the case study should be included, because, given the nature of the topological analysis, it is important to know the network’s dimensions, etc.  

5) The authors correctly describe the training procedure; however, it is unclear whether k-fold cross-validation was performed. It would be necessary to specify and analyze the results for multiple random seeds and consider average values, since the results might otherwise depend on an unlucky (or lucky) seed.

6) The evaluation metrics should be more clearly defined in the methodology section. While accuracy is relatively straightforward, the meaning and computation of F1-score and Macro-F1 may not be immediately clear to all readers in the context of this paper.

7) Figure 5 provides an interesting analysis of errors in terms of topological distance. However, topological distance alone may hide the actual physical magnitude of the localization error, since two graph hops can correspond to very different pipe lengths. The authors should consider also reporting the error in metric distance, e.g., meters along the network path, possibly using the branch midpoint/centroid as the representative fault position. This would make the localization performance more physically interpretable.

8) In Fig. 5a, does the “1” in section 41 indicate that 100% of the faults in section 41 have been assigned to another section? Why? Please explain.

9) Figure 6 should be discussed in more detail. The current text essentially repeats the caption, while the plot only shows true/predicted class counts. My understanding is that similar true and predicted counts for a given section do not necessarily imply correct localization, since compensating errors between sections may occur. For instance, using Section 18 as an example, some faults truly belonging to Section 19 could be predicted as Section 18, while some true faults from Section 18 could be predicted elsewhere. The authors should clarify this point and explain how Figure 6 should be interpreted in relation to the actual class-wise localization accuracy.

Minor revisions

- In lines 88–100, rather than listing the references at the beginning and then citing only the authors and year (ref. 27–31), it would be more appropriate to include the references when describing the contributions of the relevant papers. 

- Lines 302–303 and 316–318 repeat the same information regarding the 70:15:15 dataset split and the 100-epoch training setup. Given the proximity of the two passages, this repetition should be avoided.

- In Fig. 6, the text is difficult to read. It would be better to place the two graphs one above the other and enlarge them.