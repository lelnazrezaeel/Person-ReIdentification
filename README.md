# Person-ReIdentification
This project enhances person re-identification by incorporating temporal information, specifically considering time gaps between a person’s disappearance and reappearance across disjoint cameras. By adding a temporal model to an existing pretrained re-identification system, we aim to improve the accuracy of matching individuals in our custom dataset from the Iran University of Science and Technology.

## <img width="25" height="25" src="https://img.icons8.com/dotty/80/41b883/overview-pages-2.png" alt="overview-pages-2"/> Overview
Person re-identification (Re-ID) is a critical task in surveillance systems, aiming to track and match individuals across multiple camera views. While appearance-based methods dominate the field, our research introduces a temporal model to enhance the matching accuracy by considering the time difference of appearances between disjoint cameras.

This temporal model complements existing appearance-based features, leading to more accurate and reliable re-identification. Our customized dataset and limited identities allowed us to test the effectiveness of the model in real-world scenarios.

## <img width="20" height="20" src="https://img.icons8.com/ios/50/41b883/features-list.png" alt="features-list"/> Features
- **Temporal Modeling**: Incorporates the time difference between appearances in two cameras to improve matching accuracy.
- **Pretrained Re-ID Base**: Built on top of the [SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID) model for appearance-based person re-identification.
- **Custom Dataset**: Used a private dataset with 35 identities from cameras at two doors at Iran University of Science and Technology.
- **Error Reduction**: By combining temporal and appearance features, we achieve a reduction in identification errors.

## <img width="20" height="20" src="https://img.icons8.com/ios/50/41b883/database-options.png" alt="database-options"/> Data 
Our dataset consists of 35 unique identities, captured from two disjoint camera locations (door 1 and door 3) within Iran University of Science and Technology. Due to the limited size of the dataset, additional preprocessing was performed to ensure compatibility with the pretrained model used in this project.

**Note:** The dataset is private and not available for public use.

## <img width="20" height="20" src="https://img.icons8.com/ios/50/test-passed--v1.png" alt="test-passed--v1"/> Results
By incorporating temporal features into the SOLIDER-REID base model, we observed an improvement in identification accuracy.

We used the **Leave-One-Out** method in the temporal model for better performance and generalization. In the **Few-Shot Learning** method, we employed two procedures: 
- **Few-Shot Learning ReID (1)**: Averaging the scores of the galleries corresponding to the query when we have 4 gallery images per query.
- **Few-Shot Learning ReID (2)**: Selecting the highest score among the galleries corresponding to the query.

The table below summarizes the performance of the model in terms of Rank-1, Rank-5, and Rank-10 accuracies:

| Method                                                    | Rank 1 | Rank 5 | Rank 10 |
|-----------------------------------------------------------|--------|--------|---------|
| Leave-One-Out Temporal + One-Shot Learning ReID            | 87.1%  | 93.5%  | 93.5%   |
| Leave-One-Out Temporal + Few-Shot Learning ReID (1)        | 87.1%  | 93.5%  | 96.8%   |
| Leave-One-Out Temporal + Few-Shot Learning ReID (2)        | 87.1%  | 93.5%  | 96.8%   |
