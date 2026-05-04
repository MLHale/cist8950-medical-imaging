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
2. Extract it into your project directory make sure theres another folder in the "Code folder" named Data dont know why that didn't get added as well. 
3. Run `Testing_total_segmentator.py` — this pulls the specific segmentation files needed and sets up the working directory automatically

## Running the Pipeline
```bash
# Step 1 - Extract and set up data (run once)
python Testing_total_segmentator.py

# Step 2 - Run the registration
python "Total Segmentator 3D Registration.py"

# Step 3 - View output visualizations
python testing_output_mainfile.py
```

Output is saved as `transformed_liver_seg.nii.gz`.

## Conceptual model

![Conceptual model diagram](images/conceptual_Model.png)


Link to Visio Embedding this as a link isnt working for some reason 

https://uofnebraska-my.sharepoint.com/personal/75885073_nebraska_edu/_layouts/15/Doc.aspx?sourcedoc={cf3f133e-cb40-4e76-9683-9e982aa7d600}&action=embedview

Planning on also creating a more in-depth model of the pipeline itself. 

I had Claude help me to format this in .md 


I will probably be updating this before next monday 
