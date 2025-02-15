### Dimension Reduction Analysis  

In this assignment, we applied UMAP for dimensionality reduction to visualize patterns in the data. We first ran the algorithm on both the original and tuned models, analyzing how the visualization changed with different random seeds. We observed that using the same seed produced identical plots, while changing the seed altered the layout. However, the relative distances between names remained roughly similar, though this was assessed visually rather than quantitatively. This suggests that while UMAP introduces some randomness, the overall structure is somewhat preserved.  
![Figure_1](https://github.com/user-attachments/assets/1f3922bd-5d34-453f-8120-a731e5b751c4)
![Figure_2](https://github.com/user-attachments/assets/ddace54e-67da-4e4c-9557-eae2a6c035ac)
For the tuned model, we optimized hyperparameters and evaluated the impact on Spearman correlation. Despite achieving a better correlation score, the UMAP visualization did not provide clear conclusions about improvements. We experimented with increasing `n_neighbors` and decreasing `min_distance` to tighten clusters, but these adjustments primarily affect the grouping density rather than correlation.  
![Figure_3](https://github.com/user-attachments/assets/d70d3292-539e-4486-8c96-b7738c57b223)

Our dataset consists of only 19 records, with each individual contributing just one or two lines of data. Given this limited sample size, any conclusions drawn from UMAP must be interpreted with caution. While the method helps in identifying potential structures, its sensitivity to initialization and small dataset limitations prevent us from making strong claims about model stability based purely on visualization.
