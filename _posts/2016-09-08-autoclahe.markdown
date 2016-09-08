---
layout: default 
title:  "Auto Mutli-scale Contrast Limited Adaptive Histogram Equalization"
date:   2016-09-08 10:07:35 -0700
categories: 
---
## Overview
### Abstract
One common step in image editing is global histogram equalization. This increases the contrast of a image by spreading out the dens area of the histogram giving them more dynamic range. 

Contrast limited adaptive histogram equalization (CLAHE) is a image filter is similar to global histogram equalization. Except that the equalization is done per pixel depending on the histogram of the neighboring pixels and it limits the local contrast. 
Both methods are well established and are available in most image editing suites. The goal of this project is to improve the effect and usability of CLAHE. Since it is often hard to setup CLAHE without getting unrealistic effects. It does so simplifying the interface to the filter. The created interface is a minimum image similarity limit. Then extending it to act on multiple image scales given by the Gaussian pyramid. This makes the filter easier to use and creates a realistic but still useful effect. 

### Results
The results are shown below. The effect on this photo is purposefully understated as the intent is to keep the photo realistic. The most note-able changes are around the city. 
Before (right)/After(left):  
![before](/assets/CLAHEoriginalscaled.jpg){:class="img-results"}
![before](/assets/CLAHEpostscaled.jpg){:class="img-results"}

Full resolution:
[Before]({{ site.baseurl }}/assets/CLAHEoriginal.jpg),
[Post]({{ site.baseurl }}/assets/CLAHEpost.jpg)

## Technical Details
### Interface simplification
CLAHE has two inputs. Clip limit, the limit of local contrast with the algorithm enforces. Block size, the size of the neighborhood used to calculate the local histogram and transform. Although functional the effect on the output image is hard to predict given clip limit and block size. Which makes it hard to find the optimal parameters settings if the user is editing them manually. A simpler interface is maximum effect size. To create this interface clip limit and block size must be automatically determined given the limit of effect size. 

Effect size can be defined as inversely related to image similarity. Image similarity is a wide studied research topic which has applications to compression, and image enhancement. There are a large number of metrics that can measure image similarity. The one the worked the best for this application is structural similarity (SSIM). So the user give a minimum SSIM which effectively limits the filters maximum effect. 

To be able to search the clip limit, and block size space, there needs to be heuristic of effectiveness. In this case that heuristic is pretty direct. Since the goal of the filter is to increase localized contrast. Local contrast can be measured by local standard deviation. So the effectiveness of the filter is the increase is local standard deviation. 

Given these two metrics the clip limit and block size can be searched to maximize contrast while keeping the structural similarity above a minimum level. Although a grid search, even a limited one, of these two parameters is too costly computationally. Luckily given a block size, the SSIM decreases monotonically with clip limit and contrast increases monotonically with clip limit. This makes sense since the clip limit was created to limit the over amplification of noise which reduces SSIM. This was found through experimentation. This allows for a binary search of clip limit to find the limit which is as close to the SSIM minimum without going under. 

Unfortunately block size was not monotonic. So a linear search was the best that is possible. 
So to find the best block size and clip limit, the program checks all block sizes and for each block size does a binary search of clip limit. Then returns the image with the highest contrast gain. 


### Multi scale enhancement 
Multi-scale algorithms use multiple copies of the same images at different scales. The scales are calculated by doing a blur and subsample pass. Each pass gives a new image at a smaller scale with a smaller resolution. For this filter the CLAHE filter process is applied on each scale (with the automatic parameter generation). Then the scales are merged back together to produce the final output image. 

This helps limit over editing effect that CLAHE filter has which can make images unrealistic. 

  
