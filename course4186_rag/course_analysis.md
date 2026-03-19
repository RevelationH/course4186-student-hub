# Course 4186 Lecture Analysis

## Scope reviewed

Analyzed folder:

`D:\digital_human\4186\4186`

Observed structure:

- `Week1` to `Week12`
- `Tutorial\Week5` to `Tutorial\Week11`
- file formats: `pdf`, `pptx`, `docx`, and one legacy `ppt`

## What this project is about

Course 4186 is a computer vision course centered on image formation, feature extraction, recognition, geometry, and motion analysis. The lecture sequence is consistent with a standard modern computer vision pipeline:

1. low-level image processing
2. edges and texture
3. feature extraction and matching
4. geometric transformations and alignment
5. camera models and stereo geometry
6. structure from motion
7. motion and optical flow

The materials are not a general programming course. They are specialized lecture notes for computer vision, with tutorials and Q&A materials that can be reused as high-value supervision for knowledge-point extraction and exercise generation.

## Evidence from lecture materials

Representative extracted lecture titles and opening text:

- Week 1: `Lecture 1`, textbook reference to Szeliski, introduction to vision and image filtering
- Week 2: `Edge detection`, `Image Resampling`
- Week 3: `Natural Textures`, `Convolutional Neural Network`
- Week 4: `ImageSegmentation`
- Week 5: `Harris corner detection`, feature extraction, local features
- Week 6: `Scale Invariant Feature Transform`, `Bag of Words`, image matching
- Week 7: `Transformations`
- Week 8: `Image alignment`, `Cameras`
- Week 9: vector geometry and `Stereo Vision`
- Week 10: `Epipolar constraint example`, `Structure from motion`
- Week 11: `Motion and Optical Flow`
- Week 12: course-wide `Revision`

There is also a special lecture deck:

- `Week7\LLM_Deepseek.pptx`

That file appears to be an additional topical presentation on LLM evolution and prompting, not the main computer vision syllabus.

## Tutorial material characteristics

The tutorial folders contain exercises and answers in `docx` and `pptx`, for example:

- convolution exercises
- Harris corner detector conceptual questions
- SIFT invariance questions
- camera model and vanishing point questions

This is important because the tutorial set provides ready-made patterns for:

1. identifying the expected granularity of knowledge points
2. generating question styles aligned with the course
3. validating whether generated exercises are on-topic

## Suggested knowledge hierarchy

For RAG storage, the course is best represented in a hierarchy:

1. Week
2. Lecture file
3. Chunk
4. Knowledge point

Suggested knowledge-point families:

- Image representation and filtering
- Edge detection and resampling
- Texture analysis
- CNN-based image classification
- Segmentation
- Corners, blobs, and local features
- SIFT and invariant descriptors
- Bag-of-words image retrieval
- 2D transformations and alignment
- Camera model and projection
- Vector geometry for vision
- Stereo correspondence and depth
- Epipolar geometry
- Structure from motion
- Motion estimation and optical flow

## Why RAG is suitable here

This course is a strong fit for RAG because:

1. the knowledge is document-grounded and mostly declarative
2. terminology is stable and repeated across lectures and tutorials
3. exercise generation should stay close to the lecture evidence
4. later topics depend on earlier definitions, so retrieval helps preserve prerequisites

## Risks found during analysis

1. Mixed formats mean a single PDF-only parser is not enough.
2. One legacy `.ppt` file needs fallback handling and may fail on machines without PowerPoint COM support.
3. Some extracted PDF pages contain noisy text such as `Ignoring wrong pointing object...`; chunk cleaning is required.
4. Revision slides duplicate earlier topics, so chunk deduplication or overlap-aware retrieval is useful.

## Design conclusion

The correct implementation path is not to reuse the original project RAG directly. Instead, Course 4186 should use a separate pipeline that:

1. reads the real folder structure recursively
2. stores both chunks and knowledge points
3. retrieves evidence first, then generates exercises from that evidence
4. keeps all new code and artifacts outside the original project paths
