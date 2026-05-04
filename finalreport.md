# Project name
Probabilistic CT Liver Atlas for Distance to Vasculature Estimation

Tristan Jones

Capstone Spring 2026

## Executive Summary
Vasculature infiltration is a clinically significant event that is suspected to be the final stage of cancer. This project seeks a method to approximate the distance to the vasculature from a standard clinical CT scan by constructing a population-level probabilistic atlas of the liver and its vasculature. Rather than registering one patient's anatomy directly onto another's, all subjects are aligned into a shared coordinate frame and accumulated into a density model, analogous to an electron cloud, that captures where liver tissue and vasculature typically exist across a population. The pipeline for this project uses TotalSegmentator to automatically extract the liver parenchyma, all eight Couinaud segments, the portal vein, and hepatic vein segments from standard CT volumes retrieved from the TotalSegmentator website. Volumes are rigidly aligned using centroid-based translation and NIfTI affine direction normalization, and accumulated into sex-stratified probabilistic atlases for male and female cohorts. Rigid alignment improved median Dice similarity from 0.146 to 0.724 across 261 subjects. Distance-to-vasculature maps were computed per subject in native space and then warped into atlas coordinates. The project focuses on the hepatic and portal venous systems; angiogenesis and capillary structures are explicitly out of scope. The resulting atlas provides a foundational framework for estimating tumor-to-vasculature proximity in new patients.  




## Project Goals
The Project originated from a collaboration with Dr. Ghersi on a core clinical question: can we determine the distance between a tumor and the surrounding vasculature from a standard CT scan? The initial proposed deliverable was a reproducible registration pipeline covering:
CT preprocessing and orientation normalization
TotalSegmentator execution for automated segmentation
Surface and landmark extraction
Staged registration (rigid -> affine -> non-rigid)
Transformation application to segmented structures
Distance-to-vasculature computation

As the project progressed, TPS-based non-rigid registration was largely abandoned due to the fundamental point-correspondence problem Dr. Hale had pointed out. It was very difficult to ensure that a specific point in one liver corresponded 100% to a point in another liver. This was possible to some extent with landmarks, but covering the entire liver would be a difficult endeavor entirely. From there, the scope shifted into creating more of an electron cloud model, which allowed me to create the population-level probabilistic atlas model. As such, the final goals become:
Build a sex-stratified probabilistic liver atlas from a large clinical CT cohort
Compute population-level distance-to-vasculature maps for the hepatic and portal venous systems
Establish a foundational framework for future tumor localization in atlas space






## Project Methodology
Dataset and Preprocessing 
CT volumes were sourced from the publicly available TotalSegmentator dataset, this is comprised of 1228 clinically acquired scans. A manual integrity verification step was performed to exclude cases with truncation artifacts or incomplete liver volumes, retaining a total of 303 cases after this first step. From there, following automated segmentation and a second quality control pass to detect silent segmentation failures, the final analytic cohort consisted of 261 subjects (107 female and 154 male). All of the volumes were padded into a common array grid based on the maximum observed volume extents.
Automated Segmentation 
Three separate Total Segmentator subtask calls were used to extract the liver parenchyma, the portal/splenic vasculature, the eight Couinaud liver segments, and the hepatic vessel masks, respectively 
Rigid Alignment
Alignment was performed in two separate passes using centroid-based translation and a canonical orientation matrix derived from each subject's NIfTI affine direction. Isotropic scaling was intentionally omitted to preserve natural size variation across the population. 
Atlas Construction 
Aligned liver masks were accumulated into a shared global grid and normalized by patient count to produce a probabilistic density volume. A consensus liver boundary was defined at the 0.5 density threshold. However, it should be addressed in future work. 
Distance to vasculature Mapping
Per-subject distance maps were computed in native space using a k-nearest neighbor approach (k=5) implemented with a k-D Tree, averaging the 5 nearest neighbors. Then warped into atlas space via trilinear interpolation and then averaged across the population. 
Approaches Considered and Abandoned 
Thin Plate Spline non-rigid registration was explored but abandoned due to the point correspondence problem; effectively, there was no reliable method to achieve 100% point correspondence between every voxel in the entire dataset and every voxel in the new target subjects' dataset. You could get some landmarks but this step needs a lot of further refinement and iteration to be useful. 

I was confused about what this was I thought this was the milesone_3.md kind of thing you wanted so thats why I added all that my brain is clearly a little fried I'll keep the rest of this cause its fine but Im getting rid of theat resutls section cause Its really supposed to be in the milestone3.md


## Install Instructions

### Requirements
- Python 3.9+
- ~25GB disk space for the full dataset
- CPU hardware (no GPU required, however GPU will be faster for segmentations)
- nibabel
- numpy
- scipy
- plotly
- matplotlib
- totalsegmentator

### Installation Instructions
1. Clone the repository from GitHub
2. Run `pip install nibabel numpy totalsegmentator matplotlib scipy plotly`
3. Download `Totalsegmentator_dataset_v201.zip` from https://zenodo.org/records/10047292
4. Extract into a `Data/` folder in the project directory

### Getting Started
```bash
# Step 1 - Run segmentation on CT volumes
python Segmentation/run_totalsegmentator.py

# Step 2 - Validate dataset and filter usable patients
python Validation/validate_dataset.py --source zip --zip-path Data/segmentations.zip

# Step 3 - Build the liver atlas
python -m Atlas.liver_atlas

# Step 4 - Build the distance-to-vasculature cloud
python -m Atlas.vascular_distance
```


I had Claude help me format a few of these things in markdown 



