# Progress Report 3/30/26
## Overview
So a lot of this Milestone was centered around two things: the first being, of course, figuring out the exact direction of the project, since this is a new project, and the second being implementing those items. On the first day, I met with Dr. Hale, and I brought a lot of ideas about what I thought the future of this project would look like. Essentially, though, I brought forward this original 6-step pipeline idea:
CT preprocessing if needed? (spacing, orientation normalization) 
TotalSegmentator execution (currently the CTs Ive worked with have been preuploaded from the total segmentator website)
Surface or landmark extraction
Staged registration (rigid → affine → non-rigid)
Transformation application to segmented structures
Distance-to-vasculature computation 

When I started the project, I already had a semi-working TPS that could be used on randomly generated 3D shapes, which I had been working on implementing for use with organ shapes. Some of that code still resides. I think the TPS code is largely the same, but the way it implements the transformation has changed completely, for better or worse. I will get to why I’m saying this later. 

After bringing this pipeline and project idea to Dr. Hale, I was instructed to focus primarily on the preprocessing component, which I feel I have mostly completed. I did pretty extensive research on ways I could preprocess, only to realize I was running in circles for no reason, since many of the things I wanted to implement were already covered in one way or another by Total Segmentator. Anyways, I think I probably spent too much time on research and preliminaries, and not enough time just testing things out. I spent far too much time trying to use 3D Slicer to view the segmented files and the CT scans themselves, which really slowed me down as well. 

While I had an idea of what I wanted to do, I had no kind of compass or direction on how to get there, which is kind of what this project has always been about. Even when I first started it, it was the same way. However, through the process of researching things, I did rule out a lot of other options that might be unhelpful for me to implement, which was great, I think, and again par for the course for this project in a lot of ways. 

The biggest thing that kept coming back up was how the computer or the program knows which way is up. I did a lot of research on this, but again, I think I was largely looking in all the wrong places because, of course, the real answer I came to was that I just needed to indicate, or find ways to indicate, special markers of landmarks on the organ itself. So through my research, I came to two conclusions, the first being that I needed to utilize the Couinaud classification system for subsegmenting the liver into several “lobes” or segments, and two that I could indicate “which way is up” by just getting data on where certain aspects of the organ were located in point space.

## Outcomes


also list them out like this:
* outcome 1 - CT Preprocessing to get the two organs with the same space
A big challenge I didn't initially realize is that voxel spaces are actually unique in that they are defined by their own set of coordinates and values, so there's an issue right away. We can alleviate this by changing from voxels to mm and then centering the organs around the origin. However, this may still need more work. I need to look into it more. 
Additionally, another major issue was the question of how to rotate these things to align. I did a lot of searching to figure this one out and a lot of dead ends, basically, but I came to find a rotation matrix through SVD, then applied it. I know that sounds obvious in retrospect, but it is what it is. I was just so locked into normal vectors, aligning principal components, and other things that really didn't make a lot of sense. 

* outcome 2 - TotalSegmentator execution 
I used Claude to make a wrapper that essentially calls the functions from the total segmentator to segment things appropriately. This was originally just me testing things out, but at this point, it's become pretty pivotal to the pipeline's operation. The biggest hangup here was that I was originally just using the presegmented files, but I later found out there were some issues with them. Things are being randomly cut off, etc. This was also an issue with 3d slicer. I probably spent like 9+ hours just trying to use 3D Slicer to view things or figure out how things look. Then I gave up and had Claude make me some visualizations so I could look at the shapes of things. This was made through voxel triangulation, a form of downsampling, but it makes it work, and I can generally see what things look like. This is not a main part of the code and is just for my own viewing, really, at this point. 
Essentially, though, the Total Segmentator wrapper is the main thing here. 

* outcome 3 - Surface or landmark extraction 
As I mentioned earlier, I concluded that the best approach was to select landmarks in a few specific locations on the Liver. Alongside this, I also implemented Landmarks that were in alignment with the Centroids, or the most central point within the 8 liver segments. I think there's still a lot of room for improvement for this. 
This was based on the question Dr. Hale asked, which was “what is a corresponding point,” so I really wanted to hone in on that by extracting specific points that were kind of anatomically relevant, so to speak. 

* outcome 4/5 - Staged registration (rigid → affine → non-rigid)/Transformation application to segmented structures
I took out some of the transformation aspects I had originally, which, as I said earlier, were really just based on a random 3D shape. This part needs to be updated, and in part with section 3, more landmarks. Currently, it still technically operates but the result is wonky to say the least and the reason for that is specifically because is now only focused on the landmarks and not the entire shape. 
I've also considered implementing other methods beyond TPS. I originally implemented this one because I thought it was the best at the time, and it's good, but maybe not ideal for this specific point cloud problem. I originally implemented that like probably 6 months ago, though, at this point 
Realized this was also talking about the transformation, so added 5 here as well. But yeah, the transformation was originally just sampled points, and, at best, it was questionable how well they corresponded to one another, as Dr. Hale pointed out. There's now stronger evidence that we have corresponding points. But also, I think there's more work to be done in this. Possibly, instead of just centroids on the 8 liver segments, I could also grab landmarks on those as well, even just the tips of those would be interesting landmarks, I think, to focus on. 

* outcome 6 - Distance-to-vasculature computation 
Technically, the end goal of the project, but not quite there; however, I have done a lot of research and work towards a better understanding of how to get there. 
There are only a few intervasculature methods in the Total Segmentator, one of which is the liver. The Portal vein is actually on the outside of the liver and is connected to the Hepatic Vein(s), which I was misled about a long time ago. Essentially, the portal vein drains into the Hepatic Veins; hence, my understanding of the different branches as far as I understand, most people have 3 Hepatic Vein branches, but some people may have 4 hepatic vein branches. For context, when I first really started working on this part of my research, I was looking through videos from med students and additionally reading some anatomy textbooks for med students. 
For the purposes of this project, Dr. Ghersi advised me to essentially ignore angiogenesis, so I’m going to try my best to focus on the kind of main branches rather than focusing on things like the capillaries or smaller subsystems of the circulatory system, as that's what Dr. Ghersi advised me to do for the purposes of this project. This project was originally intended to be a stepping stone to determine whether this is possible and how useful it would be. But later on, the plan was to involve radiologists in the project to verify the results and move forward to the next phase, which would focus on the generalizability of this function. 
This is a very big rant here, which is maybe not necessary, but all that is to say that I think the distance-to-vasculature computation should be possible. The purpose of which was, of course, to see how long someone has before vasculature infiltration. I am unsure whether I’ll be able to do all of that, but I have ideas for how it could be implemented. 
This calculation should be relatively easy, though my initial thought is to do Euclidean distance. There also might be some argument to do some kind of Euclidean Nearest Neighbor kind of shenanigans, which is what I’m calling it now, but essentially finding how many nearest neighboring vessels there are and returning the distance to, like, let's say, 5 different ones. This would be a better implementation in practice. 


## Hinderances
As I stated before, I think the main hindrance I had was a lack of direction, which is a big part of this project overall. Just not knowing exactly what the next steps are along the path. It feels like it's been a lot more of ruling out options than implementing things, which I get that sometimes that's how things are, but it's very overwhelming for me when I feel like I have more options than pathways. However, I think I have a bit more momentum now than when I started. But there are still many problems that need to be addressed in the context of the project. 

I think that's the largest hindrance, though. Also, being a solo project, it's hard when you can't bounce ideas off other people, and another hindrance has just been the lack of time. I had one week where I had like multiple big things due and a midterm, and I have just been feeling very overwhelmed lately. Largely like psychological issues, but I definitely think that lately this has been the most stressed I’ve been in a long time. So part of me wishes I had just coasted things out on the other project, but like that's just not who I am, I just have to do things the hard way. 

Another hindrance, probably just that prior to picking this back up, it had been months since I’d last worked on all of this, and so I had forgotten, actually, more than I realized about certain aspects of the project. So, in some ways, it felt like I was kind of starting over, needing to reiterate and parse through things I'd forgotten about. Or even re-researching things and finding that actually something I had originally ruled out might actually be better than what I’m currently doing. 

## Ongoing Risks
(address your project risks identified from Milestone 1 and update them based on your current progress, this should be a table)

|Risk name (value)  | Impact     | Likelihood | Description |
|-------------------|------------|------------|-------------|
| Anatomical variability exceeds TPS correction (64) | 8 | 8 | Inter-subject liver shape differences may be too large for TPS to produce anatomically valid mappings, particularly without dense, reliable landmarks. Current results show instability. Mitigation: add more anatomically grounded landmarks (segment tips, vessel entry points). |
| Vasculature not segmentable via TotalSegmentator (40) | 10 | 4 | The hepatic/portal vein coverage in TotalSegmentator is limited and may not provide sufficient vascular structure for distance computation, the project's core deliverable. I’ve only looked at the vasculature on one subject, but it's particularly spotty, and it may be difficult to identify the 3 hepatic branches. Additionally, you may need to accept that distance computation may be approximate or deferred. |
| Landmark correspondence remains anatomically ambiguous (36) | 9 | 4 | Couinaud centroid landmarks may not be sufficiently distinctive or stable across subjects to anchor non-rigid registration. Mitigation: incorporate additional landmarks (e.g., vessel branching points) and refine the extraction logic. |
| Pipeline generalizability beyond liver unvalidated (28) | 7 | 4 |All development and testing (so far) have been mostly liver-specific, currently leveraging Couinaud segments as anatomical anchors. Other organs lack equivalent subdivision frameworks, making landmark extraction and correspondence much harder. If generalization is expected, this is unaddressed. Mitigation: accept liver-only scope explicitly in documentation, or prototype one additional organ to test transferability. Additionally, I think this methodology is likely to be generalizable to the entire body system and may be used only for local alignment comprehension. |
| Solo project/knowledge gaps cause delays (24) | 6 | 4 | Competing academic obligations further compress available time. Accept: scope may need narrowing; focus on pipeline correctness over breadth. |


I had Claude help me to make this risk list because I really didn’t know what to put here, but I updated it. 
