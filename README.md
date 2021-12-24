# A Look into the Past
Version 1

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

**We may develop auto match algorithm in the future**

Schema:
a tuple indicate upper-left and bottom-right points on the diagonal `(p_ul,p_br)`

### Step 2, Scale and search

### Step 3, Smooth the edge




