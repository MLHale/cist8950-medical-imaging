# Project name
Probabilistic CT Liver Atlas for Distance to Vasculature Estimation

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

## Results / Findings
Milestone 2:
outcome 1 - CT Preprocessing to get the two organs with the same space A big challenge I didn't initially realize is that voxel spaces are actually unique in that they are defined by their own set of coordinates and values, so there's an issue right away. We can alleviate this by changing from voxels to mm and then centering the organs around the origin. However, this may still need more work. I need to look into it more. Additionally, another major issue was the question of how to rotate these things to align. I did a lot of searching to figure this one out and a lot of dead ends, basically, but I came to find a rotation matrix through SVD, then applied it. I know that sounds obvious in retrospect, but it is what it is. I was just so locked into normal vectors, aligning principal components, and other things that really didn't make a lot of sense.

outcome 2 - TotalSegmentator execution I used Claude to make a wrapper that essentially calls the functions from the total segmentator to segment things appropriately. This was originally just me testing things out, but at this point, it's become pretty pivotal to the pipeline's operation. The biggest hangup here was that I was originally just using the presegmented files, but I later found out there were some issues with them. Things are being randomly cut off, etc. This was also an issue with 3d slicer. I probably spent like 9+ hours just trying to use 3D Slicer to view things or figure out how things look. Then I gave up and had Claude make me some visualizations so I could look at the shapes of things. This was made through voxel triangulation, a form of downsampling, but it makes it work, and I can generally see what things look like. This is not a main part of the code and is just for my own viewing, really, at this point. Essentially, though, the Total Segmentator wrapper is the main thing here.
outcome 3 - Surface or landmark extraction As I mentioned earlier, I concluded that the best approach was to select landmarks in a few specific locations on the Liver. Alongside this, I also implemented Landmarks that were in alignment with the Centroids, or the most central point within the 8 liver segments. I think there's still a lot of room for improvement for this. This was based on the question Dr. Hale asked, which was “what is a corresponding point,” so I really wanted to hone in on that by extracting specific points that were kind of anatomically relevant, so to speak.

outcome 4/5 - Staged registration (rigid → affine → non-rigid)/Transformation application to segmented structures I took out some of the transformation aspects I had originally, which, as I said earlier, were really just based on a random 3D shape. This part needs to be updated, and in part with section 3, more landmarks. Currently, it still technically operates but the result is wonky to say the least and the reason for that is specifically because is now only focused on the landmarks and not the entire shape. I've also considered implementing other methods beyond TPS. I originally implemented this one because I thought it was the best at the time, and it's good, but maybe not ideal for this specific point cloud problem. I originally implemented that like probably 6 months ago, though, at this point Realized this was also talking about the transformation, so added 5 here as well. But yeah, the transformation was originally just sampled points, and, at best, it was questionable how well they corresponded to one another, as Dr. Hale pointed out. There's now stronger evidence that we have corresponding points. But also, I think there's more work to be done in this. Possibly, instead of just centroids on the 8 liver segments, I could also grab landmarks on those as well, even just the tips of those would be interesting landmarks, I think, to focus on.

outcome 6 - Distance-to-vasculature computation Technically, the end goal of the project, but not quite there; however, I have done a lot of research and work towards a better understanding of how to get there. There are only a few intervasculature methods in the Total Segmentator, one of which is the liver. The Portal vein is actually on the outside of the liver and is connected to the Hepatic Vein(s), which I was misled about a long time ago. Essentially, the portal vein drains into the Hepatic Veins; hence, my understanding of the different branches as far as I understand, most people have 3 Hepatic Vein branches, but some people may have 4 hepatic vein branches. For context, when I first really started working on this part of my research, I was looking through videos from med students and additionally reading some anatomy textbooks for med students. For the purposes of this project, Dr. Ghersi advised me to essentially ignore angiogenesis, so I’m going to try my best to focus on the kind of main branches rather than focusing on things like the capillaries or smaller subsystems of the circulatory system, as that's what Dr. Ghersi advised me to do for the purposes of this project. This project was originally intended to be a stepping stone to determine whether this is possible and how useful it would be. But later on, the plan was to involve radiologists in the project to verify the results and move forward to the next phase, which would focus on the generalizability of this function. This is a very big rant here, which is maybe not necessary, but all that is to say that I think the distance-to-vasculature computation should be possible. The purpose of which was, of course, to see how long someone has before vasculature infiltration. I am unsure whether I’ll be able to do all of that, but I have ideas for how it could be implemented. This calculation should be relatively easy, though my initial thought is to do Euclidean distance. There also might be some argument to do some kind of Euclidean Nearest Neighbor kind of shenanigans, which is what I’m calling it now, but essentially finding how many nearest neighboring vessels there are and returning the distance to, like, let's say, 5 different ones. This would be a better implementation in practice.

Milestone 3:

Outcome 1- TPS abandoned, this is still very much a part of the code itself and there is a module in there for the TPS so its still possible to implement this down the line as an additional step. However, this was a really big point of contention not necessarily between Dr. Hale and me, but just with the project itself. I talk a lot about the “correspondence problem” I think that this is a really difficult problem to solve. I also think quite possibly that the issue is likely in my TPS implementation, it was originally made with just 3D objects such as a sphere or random organ-ish object in mind, but maybe there's a way to implement this where it actually does work, and we can still capture the variance between patients? Who knows its an interesting point though and Its unfortunate that it got abandoned 
Outcome 2 - Kind of in the same vein, but redesigning the whole pipeline around this probabilistic atlas model. I think this was a very fast, sudden shift in a whole new direction that really just happened, I mean, less than a month ago tbh. However, the electron cloud thing really clicked for me. 

Outcome 3- manual integrity verification of 1228 CT scans and getting a final cohort of 261. I'm tired of typing things out, so I’m not going to explain this all again. Its a thing I did it took a long time to complete and my neck hurt after

Outcome 4- Two pass rigid alignent into a conical coordinate frame using centroid translation and NIfTI affine direction normalization. Isotropic scaling was intentionally omitted to preserve natural size variation across the population. This probably should be addressed theres so much going on with this but effectively the Isotropic scaling was implemented at one point its just not ever applied to building the atlas itself. Effectively the full prosecutes rigid registration method is translation, rotation and scaling; the nifti stuff can apply scaling as well and shearing and also rotation. So originally, SVD was used to get the rotation matrix and I was trying to calculate the rotation based on the Couinaud centroids. This was kind of hastily removed and patched with just using the nifti affine rotation. The reason it was removed though was because it kept calculating I think possibly in order or something and so instead of getting small little discreet rotations it would end up getting a rotation that was like 150+ degrees so it kept applying it and then calculating etc. I just removed it at the time but It could come back in theres a lot of skeletons in the closet so to speak about this part of the code since it was supposed to be something else but I just wanted to ensure that it worked first. Just know there was a lot of testing that went into making sure this part actually worked. I think essentially I just need to if I try to fix this implement a different way of calculating the rotation matrix from the segments like getting that extra bit of alignment will be beneficial I think but it comes down to just making sure this calculation is very clear. 

Outcome 5- sex stratified atlases for the male and female cohorts, im just glad this worked out as well as it did I think other papers do this largely in a way that is implementing ML algorithms and what not to try and get this level of accuracy. 

Outcome 6- Dice similarity coefficient improved from median 0.146 pre-alignment to 0.724 post-alignment, this was a pretty big one and I think that the overall spread of the alignment was also very tight on the box and whisker plot which is a really good sign as well. All of this really came down to nailing the rotation aspect though I think that if I’m able to fix the Couinaud segments centroid rotation calculation then itll be even better

Outcome 7- Population-level distance-to-vasculature maps computed for hepatic and portal venous systems mean distance 18.29mm male, 16.48mm female I think that the implementation using the mean distance seemed really wise at first but I think in reality its actually just skewed this calculation. It was meant to make sure that there was stability so that we werent calculating random kind of points off in space but ended up not really working out. It gives a min distance that is ~5mm which is 3x larger than 1.5mm which is the voxel grid step size, so in all reality the actual min distance should be 1.5mm I’ll probably bring this up during the presentation 







## Install Instructions (if applicable)
### Requirements
Python 3.9+
~25GB disk space for the full dataset
CPU hardware (no GPU required however could be faster for segmentations)
nibabel
numpy
scipy
plotly
matplotlib
totalsegmentator



### Installation Instructions
pip install nibabel numpy totalsegmentator matplotlib scipy
Clone the repository from GitHub
Download Totalsegmentator_dataset_v201.zip from https://zenodo.org/records/10047292
Extract into a Data/ folder in the project directory


### Getting started
# Step 1 - Run segmentation on CT volumes
python Segmentation/run_totalsegmentator.py

# Step 2 - Validate dataset and filter usable patients
python Validation/validate_dataset.py --source zip --zip-path Data/segmentations.zip

# Step 3 - Build the liver atlas
python -m Atlas.liver_atlas

# Step 4 - Build the distance-to-vasculature cloud
python -m Atlas.vascular_distance



