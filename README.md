# A Look into the Past


## Pipeline

### I/O

INPUTS: 2 images of random shape + rectangle box for overlapped objects

- One image is what we call "new" one, indicating it was shot in recent days.
- The other image is the "old", meaning it was captured in the past.

OUTPUT: 1 image that combines these 2 images

- We always bring the "old" one to front of the "new" one, where the objects are from the "old" and the background is from the "new".
- Objects need annotated by humans.

### Step 1, Annotate overlapped objects

Assume that 2 images have some objects in common, otherwise there's no such "a look into the past".

Thus, we require the users to first annotate the objects, in case that there are too many objects to match and get the undesired outcomes.

Schema:
a tuple indicate upper-left and bottom-right points on the diagonal `--box_[src] 'y_upper-left, x_upper-left, y_bottom-right, x_bottom-right'`. For example

```shell
--box_old '158,61,426,539'
--box_new '85,156,455,700' 
```

### Step 2, Extract local feature

We use SIFT to extract scale-invarient local features. See more about SIFT and how to emplement it in Python, see refer

- Emplement: [Feature Matching in OpenCV](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
- Theory: [Tutorial 2: Image Matching](https://ai.stanford.edu/~syyeung/cvweb/tutorial2.html)

Quick learning:

An image --> SIFT --> key points (loc_x, loc_y) and feature vectors of shape (num_feature, 128)

Note that different images may have different number of features

### Step 3, Match the feature

Compute Euclidean distance of each feature vector and using KNN, where we set k=2, to find most 'similar' key points

This is automatically emplemented by calling

```python
 bf = cv2.BFMatcher()
 matches = bf.knnMatch(feature_vectors_1, feature_vectors_2, k=2)
```

Even though we have KNN to avoid extreme values or key points, we still add a rule-based filter to control the quality of the key points, where we call them "good points"

### Step 4, Align the new and the old

We've extract the similar feature from 2 images. Now we need to align them in the pixel. For example, feature A is on (3, 5) in the new image while it's on (5, 6) in the old one, so we need to learn a transfomation that maps the old image to adapt to the new one.

Basically, we use RANdom SAmple Consensus (RANSAC) to estimate the transformation. [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) in Wikipedia

For emplement, just call `H, status = cv2.findHomography(kp_old, kp_new, cv2.RANSAC, 5.0)`, where it will return a 3X3 transformation matrix H

### Step 5, Smooth the edge and stitch them


