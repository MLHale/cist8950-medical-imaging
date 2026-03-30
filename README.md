# Medical Image Registration — CT-to-CT Non-Rigid Pipeline
**Author:** Tristan Jones | Spring 2026

## Environment Setup

Non-rigid CT-to-CT registration pipeline using Thin Plate Splines (TPS) applied to organ segmentations from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). Currently focused on liver segmentation registration between subjects.

## Installation
```bash
pip install nibabel numpy totalsegmentator matplotlib scipy
```

## Data Setup

1. Go to [https://zenodo.org/records/10047292](https://zenodo.org/records/10047292) and download `Totalsegmentator_dataset_v201.zip`
2. Extract it into your project directory
3. Run `Testing_total_segmentator.py` — this pulls the specific segmentation files needed and sets up the working directory automatically

## Running the Pipeline
```bash
# Step 1 - Extract and set up data (run once)
python Testing_total_segmentator.py

# Step 2 - Run the registration
python Total_Segmentator_3D_Registration.py

# Step 3 - View output visualizations
python testing_output_mainfile.py
```

Output is saved as `transformed_liver_seg.nii.gz`.

I had Claude help me to format this in .md 
