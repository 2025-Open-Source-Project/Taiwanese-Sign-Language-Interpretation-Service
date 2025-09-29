### How to make virtual avatar sign lang video
+ Open Source app: [Blender](https://www.blender.org/)
+ Open Source avatar from [Mixamo](https://www.mixamo.com/) and [MakeHuman](http://makehumancommunity.org/content/downloads.html)  

![](MakeHuman_avatar.png)  

### Why virtual avatar sign lang video instead of using real person video directly
- makes it possible to cover unlimited vocabulary and sentences
- Consistency: same avatar through out the whole system
- Recorded human videos may involve rights management, consent, and contracts with the signer

### What we tried
+ [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) keypoint combined with [MocapNET](https://github.com/FORTH-ModelBasedTracker/MocapNET), from csv to bvh

+ [ThreeDPoseTracker](https://freedom3d.art/post-category/category-artificial-intelligence-ai/threedposetracker-v0-6-2/) generate bvh from video

+ [MocapNET](https://github.com/FORTH-ModelBasedTracker/MocapNET) directly trans video to bvh

+ [rokoko Blender Plugin](https://www.rokoko.com/integrations/blender): combine bones of souce(bvh) with avatar

### Current using 
+ Blender + [ThreeDPoseTracker](https://freedom3d.art/post-category/category-artificial-intelligence-ai/threedposetracker-v0-6-2/)
+ Blender avatar (dif avatar body may have dif looking in result due to dif body scale): 
    + Mixamo_avatar(w) -> woman
    + Mixamo_avata -> man  

![](body_scale_result_in_dif_poses.png)  
video source: [台灣手語線上辭典](https://twtsl.ccu.edu.tw/) 