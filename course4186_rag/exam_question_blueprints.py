def mcq(prompt, correct, distractors, explanation, use_image=False, review_terms=None):
    item = {
        "prompt": prompt,
        "correct": correct,
        "distractors": distractors,
        "explanation": explanation,
    }
    if use_image:
        item["use_image"] = True
    if review_terms:
        item["review_terms"] = list(review_terms)
    return item


QUESTION_BLUEPRINTS = {
    "Computer vision foundations": [
        mcq(
            prompt="A system receives an image and outputs the identity of the person in it. Which statement best describes the computer vision goal being used?",
            correct="The system is extracting high-level semantic meaning from visual data rather than just storing pixels.",
            distractors=[
                "The system is only changing the file format of the image.",
                "The system is only measuring the camera focal length.",
                "The system is only increasing image resolution without interpretation.",
            ],
            explanation="In the introductory lecture, computer vision is defined as making computers gain high-level understanding from images and videos. Identity recognition is a semantic interpretation task, not a storage task.",
            review_terms=["goal of computer vision", "high-level understanding", "recognize objects and people"],
        ),
        mcq(
            prompt="Why can the same object category still be difficult to recognize across different photos?",
            correct="Because viewpoint, illumination, scale, clutter, and occlusion can change the image appearance significantly.",
            distractors=[
                "Because digital images cannot be represented numerically.",
                "Because computer vision methods only work on black-and-white images.",
                "Because every object always appears at the same scale and under the same lighting.",
            ],
            explanation="The lecture lists viewpoint variation, illumination, scale, background clutter, motion, and occlusion as core reasons why computer vision is difficult.",
            review_terms=["why is computer vision difficult", "viewpoint variation", "occlusion"],
        ),
        mcq(
            prompt="Which task is closest to the course definition of computer vision?",
            correct="Recognizing objects or people from image evidence.",
            distractors=[
                "Renaming an image file in the operating system.",
                "Counting how many bytes the image file uses on disk.",
                "Changing the wallpaper on a laptop screen.",
            ],
            explanation="The Week 1 introduction uses examples such as recognizing objects and people to explain what it means to understand the physical world from images.",
            review_terms=["recognize objects and people", "goal of computer vision"],
        ),
        mcq(
            prompt="Why does the course argue that computer vision is worth studying even before discussing specific algorithms?",
            correct="Because enormous numbers of images and videos are captured every day and many useful applications depend on analyzing them.",
            distractors=[
                "Because every camera already produces perfect scene understanding automatically.",
                "Because the field is limited to one single application such as face recognition.",
                "Because computer vision removes the need for any machine learning or geometry.",
            ],
            explanation="The Week 1 motivation slide points to the scale of visual data and the large number of useful applications as the practical reason for studying the field.",
            review_terms=["why study computer vision", "billions of images", "applications"],
        ),
        mcq(
            prompt="The application figure includes robotics, medical imaging, recognition, and marine-animal counting. What is the main message of that figure?",
            correct="Computer vision is a general field with many application areas that all rely on interpreting visual information.",
            distractors=[
                "Computer vision is only about face recognition.",
                "All computer vision problems reduce to the same single filter.",
                "The figure proves that camera calibration is the only important topic in the course.",
            ],
            explanation="The figure is used to show the breadth of the field. The point is not one algorithm, but that many different applications depend on extracting meaning from images.",
            use_image=True,
            review_terms=["applications", "robotics", "medical imaging", "marine mammal recognition"],
        ),
    ],
    "Image filtering and convolution": [
        mcq(
            prompt="A 3x3 kernel is applied at one interior pixel. Before the final sum is produced, how many pixel-weight multiplications are performed?",
            correct="9 multiplications.",
            distractors=[
                "3 multiplications.",
                "6 multiplications.",
                "18 multiplications.",
            ],
            explanation="A 3x3 filter contains 9 weights, so one output value is formed from 9 multiply-and-add terms over the local neighborhood.",
            review_terms=["linear filtering", "kernel", "weighted sum"],
        ),
        mcq(
            prompt="What is the mathematical difference between convolution and cross-correlation in the lecture notes?",
            correct="Convolution flips the kernel before applying it, while cross-correlation uses the kernel in its original order.",
            distractors=[
                "Cross-correlation flips the kernel, while convolution never does.",
                "They are identical for every kernel and every image.",
                "Convolution can only be used on color images, while cross-correlation can only be used on grayscale images.",
            ],
            explanation="The lecture explicitly states that convolution is the same local weighted computation as cross-correlation except that the kernel is flipped horizontally and vertically first.",
            review_terms=["convolution", "cross-correlation", "kernel is flipped"],
        ),
        mcq(
            prompt="If a 2D filter can be written as h(x,y) = a(x)b(y), what is the main computational benefit?",
            correct="The 2D filtering can be split into two 1D passes, which reduces computation.",
            distractors=[
                "The image no longer needs any padding or boundary handling.",
                "The filter automatically becomes nonlinear.",
                "The filter can then be applied only to binary images.",
            ],
            explanation="A separable filter replaces one expensive 2D operation with two cheaper 1D convolutions, which is exactly why separability is useful in practice.",
            review_terms=["separable", "2d filter", "two 1d convolutions"],
        ),
        mcq(
            prompt="Why is the associative property of convolution useful when smoothing and differentiating an image?",
            correct="Because the filters can be combined first, reducing the number of image-wide operations.",
            distractors=[
                "Because it guarantees zero noise in the final result.",
                "Because it makes convolution identical to median filtering.",
                "Because it removes the need to choose a derivative filter.",
            ],
            explanation="The lecture notes point out that if filtering operations are associative, we can combine filters before applying them to the image and save one full pass over the data.",
            review_terms=["associative property", "differentiation is convolution"],
        ),
        mcq(
            prompt="In the local-neighborhood figure, what does the output pixel represent?",
            correct="A weighted sum of the nearby input pixels defined by the kernel values.",
            distractors=[
                "The exact 3D location of the observed point.",
                "The camera intrinsic matrix for that neighborhood.",
                "A histogram of visual words from the image patch.",
            ],
            explanation="The diagram is illustrating local linear filtering: the output intensity is computed from neighboring pixels and kernel weights, not from geometry or recognition labels.",
            use_image=True,
            review_terms=["linear filtering", "kernel", "weighted sum"],
        ),
    ],
    "Edge detection": [
        mcq(
            prompt="Where should a strong edge response occur in an image?",
            correct="At locations where intensity changes sharply over a short distance.",
            distractors=[
                "At any perfectly uniform flat region.",
                "Only at the optical center of the camera.",
                "At every pixel, regardless of local structure.",
            ],
            explanation="The lecture models images as intensity functions and explains edges as steep changes in that function, not flat regions.",
            review_terms=["edges look like steep cliffs", "edge detection"],
        ),
        mcq(
            prompt="Which set of conditions best matches the design criteria for a good edge detector?",
            correct="Good detection, good localization, and only one response per true edge.",
            distractors=[
                "Maximum blur, minimum contrast, and repeated responses.",
                "Large histogram bins, low memory, and no derivatives.",
                "Perfect segmentation masks from a single threshold only.",
            ],
            explanation="The Week 2 edge-detection lecture explicitly lists these three design criteria for a good detector.",
            review_terms=["designing an edge detector", "good detection", "good localization", "one point only"],
        ),
        mcq(
            prompt="Why is smoothing often applied before taking image derivatives for edge detection?",
            correct="Because derivatives amplify noise, so smoothing produces a more stable edge response.",
            distractors=[
                "Because smoothing directly converts the image into a segmentation mask.",
                "Because derivatives can only be computed after upsampling the image.",
                "Because smoothing guarantees perfect subpixel localization.",
            ],
            explanation="The derivative operator is sensitive to noise. Smoothing reduces high-frequency noise so that the derivative responds more to meaningful structure.",
            review_terms=["edge detection", "noise", "derivative"],
        ),
        mcq(
            prompt="Which of the following can produce an image edge according to the lecture?",
            correct="A change in depth, surface color, illumination, or surface normal.",
            distractors=[
                "Only a change in file size on disk.",
                "Only the image resolution chosen by the viewer.",
                "Only the camera brand used to take the photo.",
            ],
            explanation="The edge-origin slide lists several physical causes of edges, including depth, color, illumination, and surface-orientation changes.",
            review_terms=["origin of edges", "depth", "surface color", "illumination", "surface normal"],
        ),
        mcq(
            prompt="If the figure highlights object outlines while suppressing flat image regions, what result is the algorithm trying to produce?",
            correct="A compact boundary-based representation that keeps salient structure and removes much of the less informative interior texture.",
            distractors=[
                "A full 3D reconstruction of the scene.",
                "A bag-of-words histogram for image retrieval.",
                "A camera-calibration matrix estimated from one frame.",
            ],
            explanation="The lecture motivates edge detection as converting a 2D image into a set of curves so that the scene is represented more compactly than raw pixels.",
            use_image=True,
            review_terms=["convert a 2d image into a set of curves", "more compact than pixels"],
        ),
    ],
    "Image resampling": [
        mcq(
            prompt="What is the main risk of reducing image resolution without low-pass filtering first?",
            correct="Aliasing can appear because high-frequency content is sampled too coarsely.",
            distractors=[
                "The image automatically gains new geometric detail.",
                "The camera focal length changes during resampling.",
                "All edges become perfectly sharp after subsampling.",
            ],
            explanation="When high-frequency image content is sampled on a coarser grid without prefiltering, it folds into misleading low-frequency patterns. That is the aliasing problem discussed in the lecture.",
            review_terms=["image sub-sampling", "aliasing", "gaussian pre-filtering"],
        ),
        mcq(
            prompt="Before downsampling by a factor of 2, what does the lecture recommend doing first?",
            correct="Apply a low-pass filter such as a Gaussian, then subsample.",
            distractors=[
                "Differentiate the image twice, then downsample.",
                "Run Harris corner detection, then downsample.",
                "Sharpen the image to emphasize high frequencies, then downsample.",
            ],
            explanation="The course notes explicitly present Gaussian prefiltering as the solution before subsampling, because it removes frequencies that would otherwise alias.",
            review_terms=["gaussian pre-filtering", "solution: filter the image, then subsample"],
        ),
        mcq(
            prompt="If you keep every other row and every other column from an image, what fraction of the original pixels remains?",
            correct="One quarter of the original pixels.",
            distractors=[
                "One half of the original pixels.",
                "One eighth of the original pixels.",
                "The same number of pixels remains.",
            ],
            explanation="Keeping every other row leaves half the rows, and keeping every other column leaves half the columns. The combined effect is one half times one half, which is one quarter.",
            review_terms=["sub-sampling", "every other row and column"],
        ),
        mcq(
            prompt="When an image is enlarged, what problem does interpolation solve?",
            correct="It estimates intensity values at new sample locations that were not present on the original grid.",
            distractors=[
                "It estimates the camera baseline from the enlarged image.",
                "It converts local descriptors into visual words.",
                "It removes the need for any image sampling model.",
            ],
            explanation="Upsampling creates a denser grid. Interpolation provides the new pixel values on that grid using nearby known samples.",
            review_terms=["upsampling", "interpolation", "image scaling"],
        ),
        mcq(
            prompt="A figure compares a coarse grid, a filtered image, and a resized output. What concept is that figure primarily designed to test?",
            correct="How sampling density and prefiltering affect the quality of the resampled image.",
            distractors=[
                "How epipolar geometry reduces stereo search.",
                "How bag-of-words histograms are normalized.",
                "How object detectors classify sliding windows.",
            ],
            explanation="The Week 2 resampling lecture uses these figures to show that grid spacing alone is not enough; proper filtering before subsampling changes the final visual result.",
            use_image=True,
            review_terms=["image resampling", "sampling", "gaussian pre-filtering"],
        ),
    ],
    "Texture analysis": [
        mcq(
            prompt="What kind of visual evidence is texture analysis designed to describe?",
            correct="Repeated local appearance patterns over a region, such as grass, leaves, or fabric-like surfaces.",
            distractors=[
                "Only one single isolated corner in the entire image.",
                "The exact focal length of the camera lens.",
                "Only the 3D coordinates of scene points.",
            ],
            explanation="Texture analysis is about region-level repeated appearance structure, not about a single keypoint or a camera parameter.",
            review_terms=["natural textures", "grass", "leaves"],
        ),
        mcq(
            prompt="In a normalized gray-level histogram P[i], what must the sum over all bins equal?",
            correct="1.",
            distractors=[
                "The image width in pixels.",
                "The number of corners in the image.",
                "The maximum gray value times the number of bins.",
            ],
            explanation="A normalized histogram represents proportions or probabilities, so all bin values must sum to 1.",
            review_terms=["gray image histogram", "normalized histogram"],
        ),
        mcq(
            prompt="Why can two image regions have the same average intensity but still look like different textures?",
            correct="Because texture depends on the distribution and arrangement of local intensity patterns, not only on the mean intensity.",
            distractors=[
                "Because average intensity already determines the exact texture uniquely.",
                "Because texture can only be measured in color images, never grayscale images.",
                "Because texture is defined only by camera motion.",
            ],
            explanation="Texture is a richer description than a single mean value. Histograms and other local statistics capture variation that the mean alone misses.",
            review_terms=["texture", "gray image histogram", "distribution"],
        ),
        mcq(
            prompt="Which example is most naturally treated as a texture-analysis problem?",
            correct="Distinguishing regions of grass from regions of leaves using repeated local appearance patterns.",
            distractors=[
                "Estimating an affine transformation from point correspondences.",
                "Recovering stereo depth from disparity.",
                "Computing an essential matrix from two calibrated cameras.",
            ],
            explanation="Texture analysis is introduced through natural surfaces such as grass and leaves, where the key cue is repeated local appearance.",
            review_terms=["natural textures", "grass", "leaves"],
        ),
        mcq(
            prompt="A figure shows surfaces such as grass and leaves rather than isolated keypoints. What should a student infer from that figure?",
            correct="The lecture is focusing on describing regional surface patterns instead of single-point features.",
            distractors=[
                "The lecture is introducing camera calibration.",
                "The lecture is explaining epipolar lines.",
                "The lecture is showing a fully convolutional segmentation network.",
            ],
            explanation="The figure is there to shift attention from isolated local points to repeated surface patterns spread across an image region.",
            use_image=True,
            review_terms=["natural textures", "how to describe the textures"],
        ),
    ],
    "Convolutional neural networks": [
        mcq(
            prompt="In the sliding-window approach shown in the lecture, what is the CNN doing at each window location?",
            correct="It is classifying the image crop to decide whether the target object is present.",
            distractors=[
                "It is computing the camera intrinsic matrix of the crop.",
                "It is converting the crop directly into a bag-of-words histogram.",
                "It is estimating dense optical flow from one still image.",
            ],
            explanation="The object-detection slide frames detection as repeated classification over candidate windows. The CNN is acting as the classifier for each cropped region.",
            review_terms=["object detection as classification", "sliding window", "cnn"],
        ),
        mcq(
            prompt="Why are convolutional layers particularly appropriate for images?",
            correct="Because the same learned local filter can be reused across many spatial locations.",
            distractors=[
                "Because every pixel requires a completely separate filter bank.",
                "Because convolutional layers eliminate the need for training data.",
                "Because they can only be applied after camera calibration.",
            ],
            explanation="Images contain repeated local structure. Shared filters let the network detect the same pattern at different positions without learning separate weights everywhere.",
            review_terms=["convolutional neural network", "feature maps", "filters"],
        ),
        mcq(
            prompt="What is the practical role of pooling or strided downsampling inside a CNN?",
            correct="It reduces spatial resolution while retaining stronger feature evidence and broadening the receptive field.",
            distractors=[
                "It estimates the scene depth directly from one feature map.",
                "It converts every hidden layer into a segmentation mask immediately.",
                "It guarantees invariance to every geometric transformation.",
            ],
            explanation="Pooling and strided downsampling compress the spatial grid so that deeper layers can aggregate information more efficiently.",
            review_terms=["pooling", "downsampling", "feature maps"],
        ),
        mcq(
            prompt="How do deeper CNN layers usually differ from early layers in the course discussion?",
            correct="They tend to encode more abstract, task-relevant visual patterns built from simpler earlier responses.",
            distractors=[
                "They stop using any spatial information at all.",
                "They can only respond to raw RGB values without intermediate features.",
                "They always have fewer channels than the input image.",
            ],
            explanation="The main idea is hierarchical representation: early layers detect simple local cues and deeper layers combine them into more meaningful features for the task.",
            review_terms=["convolutional neural network", "feature maps", "learned filters"],
        ),
        mcq(
            prompt="If a figure shows stacked learned filters and intermediate feature maps, what core point is it making?",
            correct="CNNs build image understanding through a hierarchy of local convolutional responses.",
            distractors=[
                "Stereo correspondence is restricted to a single epipolar line.",
                "A camera model is projecting 3D points into the image.",
                "Feature descriptors are being quantized into visual words.",
            ],
            explanation="Feature-map figures are used to show that the network learns and composes visual detectors layer by layer rather than reasoning from raw pixels alone.",
            use_image=True,
            review_terms=["feature maps", "learned filters", "cnn"],
        ),
    ],
    "Image segmentation": [
        mcq(
            prompt="Why is the lecture's 'How many zebras?' example a segmentation problem rather than a plain classification problem?",
            correct="Because the system must separate image regions or object instances before counting them.",
            distractors=[
                "Because the image must first be converted into a grayscale histogram only.",
                "Because segmentation ignores object boundaries and counts directly from file names.",
                "Because classification and segmentation are exactly the same task in the lecture.",
            ],
            explanation="The zebra example is meant to show that counting requires knowing which pixels or regions belong to each object, not just assigning one label to the whole image.",
            review_terms=["how many zebras", "image segmentation"],
        ),
        mcq(
            prompt="What is the defining output of an image-segmentation system?",
            correct="A partition of the image into meaningful regions or labeled pixels.",
            distractors=[
                "A single class label for the entire image only.",
                "A camera focal length estimate.",
                "A histogram of descriptor clusters for retrieval.",
            ],
            explanation="Segmentation differs from whole-image classification because it produces spatially organized region or pixel assignments.",
            review_terms=["image segmentation", "regions", "masks"],
        ),
        mcq(
            prompt="What does 'binary segmentation' mean in the Week 4 materials?",
            correct="Separating the image into two groups, typically foreground and background.",
            distractors=[
                "Representing every pixel with exactly 8 bits.",
                "Restricting the image to two camera views.",
                "Using only two convolutional layers in a network.",
            ],
            explanation="Binary segmentation is about dividing the image into two semantic groups, usually the object of interest and everything else.",
            review_terms=["semi-automatic binary segmentation", "foreground", "background"],
        ),
        mcq(
            prompt="In the toy intensity-based segmentation example, what is the key idea behind grouping pixels?",
            correct="Pixels are assigned to groups based on an image feature such as intensity, but this may be too simple for real scenes.",
            distractors=[
                "Pixels are grouped only by file order in memory.",
                "Every region must contain exactly the same number of pixels.",
                "Intensity segmentation always solves every real image correctly.",
            ],
            explanation="The slide shows that intensity can define groups in a toy case, but the lecture also warns that realistic images are more complicated than that simple setup.",
            review_terms=["auto segmentation: toy example", "intensity", "three groups"],
        ),
        mcq(
            prompt="If a figure colors different object regions or masks separately, what task is it illustrating?",
            correct="Assigning pixels or regions to different objects or semantic groups.",
            distractors=[
                "Estimating motion vectors between video frames.",
                "Detecting SIFT keypoints at multiple scales.",
                "Computing a convolution kernel from image patches.",
            ],
            explanation="Colored region masks are the standard way to visualize segmentation output, because they show which pixels belong to which region or object.",
            use_image=True,
            review_terms=["image segmentation", "masks", "regions"],
        ),
    ],
    "Harris corners and local features": [
        mcq(
            prompt="In the Harris-corner formulation, what does the SSD error E(u,v) measure?",
            correct="How much the image window changes when it is shifted by a small displacement (u,v).",
            distractors=[
                "How much the camera focal length changes during motion.",
                "How similar two different object categories are in a classifier.",
                "How many visual words are present in the patch.",
            ],
            explanation="The Harris detector studies how the local window changes under small shifts. The SSD error is precisely the shifted-window change measure used in that derivation.",
            review_terms=["ssd error", "corner detection: the math", "shift the window"],
        ),
        mcq(
            prompt="If a local image window changes very little for shifts in both x and y directions, what kind of region is it?",
            correct="A flat region.",
            distractors=[
                "A corner.",
                "A strong edge.",
                "A descriptor histogram.",
            ],
            explanation="A flat region has little intensity variation, so shifting the window in any small direction changes the patch very little.",
            review_terms=["feature extraction: corners and blobs", "flat region", "shift"],
        ),
        mcq(
            prompt="If a window changes strongly for shifts in both orthogonal directions, what image structure is most likely present?",
            correct="A corner.",
            distractors=[
                "A flat patch with constant intensity.",
                "A pure edge with variation in only one main direction.",
                "A bag-of-words vocabulary cluster center.",
            ],
            explanation="A corner is distinctive in two directions. That is exactly why the shifted-window error becomes large no matter which small 2D direction is used.",
            review_terms=["harris corner detection", "shift the window", "corner detection: the math"],
        ),
        mcq(
            prompt="Why are corners usually more useful than flat regions for matching between two views?",
            correct="Because they are locally distinctive in two directions and therefore easier to localize repeatably.",
            distractors=[
                "Because flat regions contain more unique information than corners.",
                "Because corners never change under viewpoint variation.",
                "Because corners eliminate the need for descriptors.",
            ],
            explanation="Matching needs points that can be localized reliably. Corners are more repeatable than flat regions because the image intensity changes around them in multiple directions.",
            review_terms=["feature extraction", "corners and blobs", "repeatable"],
        ),
        mcq(
            prompt="In the Harris example, why is the highlighted local window evaluated under several small shifts?",
            correct="To determine whether the local patch behaves like a flat region, an edge, or a corner.",
            distractors=[
                "To estimate the scene depth from one image.",
                "To quantize the patch into a visual word.",
                "To compute the global histogram of the whole image.",
            ],
            explanation="The entire point of the shifted-window analysis is to separate flat regions, edges, and corners based on how the patch response changes with displacement.",
            use_image=True,
            review_terms=["corner detection: the math", "shift the window", "local window"],
        ),
    ],
    "SIFT and scale-invariant features": [
        mcq(
            prompt="Which stage comes immediately before the SIFT descriptor is formed?",
            correct="Orientation assignment.",
            distractors=[
                "Epipolar-line estimation.",
                "Visual-word clustering.",
                "Semantic segmentation.",
            ],
            explanation="The SIFT pipeline in the lecture is: scale-space peak selection, keypoint localization, orientation assignment, and then keypoint descriptor construction.",
            review_terms=["scale invariant feature transform", "orientation assignment", "key point descriptor"],
        ),
        mcq(
            prompt="Why does SIFT assign a dominant orientation to each keypoint?",
            correct="To make the descriptor more robust to image rotation.",
            distractors=[
                "To estimate the 3D camera translation vector.",
                "To force every keypoint into the same image location.",
                "To remove the need for scale-space detection.",
            ],
            explanation="Orientation assignment normalizes the descriptor frame so that the same local structure can still match after rotation.",
            review_terms=["orientation assignment", "rotation"],
        ),
        mcq(
            prompt="A SIFT descriptor uses a 16x16 window divided into a 4x4 grid, and each cell stores an 8-bin orientation histogram. What is the final descriptor dimension?",
            correct="128.",
            distractors=[
                "32.",
                "64.",
                "256.",
            ],
            explanation="There are 16 cells in the 4x4 grid, and each cell contributes 8 orientation bins. Therefore the descriptor dimension is 16 times 8, which is 128.",
            review_terms=["sift descriptor", "16x16", "4x4 grid", "8 bin", "128"],
        ),
        mcq(
            prompt="Why is a SIFT descriptor generally more reliable than matching raw square windows directly?",
            correct="Because it is designed to be more stable to scale and rotation changes while summarizing local gradient structure.",
            distractors=[
                "Because it stores the entire image rather than a local neighborhood.",
                "Because it uses no local gradients at all.",
                "Because it guarantees perfect correspondence without any distance test.",
            ],
            explanation="The lecture contrasts simple square-window matching with SIFT precisely to motivate a descriptor that is more invariant and therefore more matchable across views.",
            review_terms=["feature descriptors", "simple option: match square windows", "popular approach: sift"],
        ),
        mcq(
            prompt="Why does the lecture introduce Difference of Gaussian (DoG) in the SIFT section?",
            correct="Because DoG provides a practical approximation to LoG for detecting features across scale.",
            distractors=[
                "Because DoG is used to estimate the camera baseline in stereo vision.",
                "Because DoG converts descriptors into visual words.",
                "Because DoG is the standard classifier used in sliding-window CNN detection.",
            ],
            explanation="The SIFT lecture explicitly connects LoG and DoG and uses DoG as the practical mechanism for finding scale-space extrema.",
            use_image=True,
            review_terms=["difference of gaussian", "log-dog", "scale space peak selection"],
        ),
    ],
    "Bag-of-words image retrieval": [
        mcq(
            prompt="What is the representation stored for one image in a plain bag-of-words retrieval model?",
            correct="A histogram counting how many local descriptors fall into each visual-word bin.",
            distractors=[
                "A single corner point with no descriptor information.",
                "A semantic segmentation mask for every object class.",
                "A camera intrinsic matrix estimated from the image.",
            ],
            explanation="The image is represented by counts of quantized local features, not by geometry or dense per-pixel labels.",
            review_terms=["bag of words", "histogram", "visual words"],
        ),
        mcq(
            prompt="What is a visual vocabulary in the bag-of-words pipeline?",
            correct="A set of prototype descriptor clusters used to quantize local features into visual words.",
            distractors=[
                "A list of object names that appear in the dataset.",
                "A set of camera poses used for alignment.",
                "A collection of gradient filters used for edge detection.",
            ],
            explanation="The vocabulary is the codebook of representative local-descriptor types. Each extracted local feature is assigned to one of these visual words.",
            review_terms=["visual vocabulary", "visual words", "descriptor clusters"],
        ),
        mcq(
            prompt="Why is bag-of-words useful for large-scale image retrieval?",
            correct="Because histogram representations can be compared efficiently across a large database.",
            distractors=[
                "Because the method stores the exact geometric layout of every local feature.",
                "Because it removes the need to extract any local features.",
                "Because it solves depth reconstruction at the same time.",
            ],
            explanation="Large-scale retrieval needs a compact searchable representation. A histogram of visual words is much cheaper to compare than a full pairwise local-feature matching procedure for every image pair.",
            review_terms=["large-scale image matching", "bag of words", "retrieval"],
        ),
        mcq(
            prompt="What important information is lost in a plain bag-of-words model?",
            correct="The exact spatial arrangement of the local features in the image.",
            distractors=[
                "Whether the image contains any pixels at all.",
                "Whether the descriptors were extracted from local image patches.",
                "Whether the image can be represented as a histogram.",
            ],
            explanation="A plain bag-of-words histogram keeps counts of descriptor types, but not where those descriptors occurred in the image.",
            review_terms=["bag of words", "histogram", "spatial arrangement"],
        ),
        mcq(
            prompt="If local descriptors are grouped into bins and then counted, which part of the retrieval pipeline is being shown?",
            correct="Quantizing descriptors into visual words and building the bag-of-words histogram.",
            distractors=[
                "Estimating the essential matrix from stereo matches.",
                "Assigning orientations to SIFT keypoints.",
                "Upsampling a low-resolution feature map.",
            ],
            explanation="The grouped-bin figure is illustrating the step that converts many local descriptors into a fixed-length histogram representation for retrieval.",
            use_image=True,
            review_terms=["bag of words", "visual words", "histogram"],
        ),
    ],
    "Geometric transformations": [
        mcq(
            prompt="Which transformation class consists of a linear transformation followed by a translation?",
            correct="An affine transformation.",
            distractors=[
                "A median transformation.",
                "A bag-of-words transformation.",
                "A histogram equalization transformation.",
            ],
            explanation="Affine transformations extend linear maps by adding translation, which is why homogeneous coordinates are so useful in this topic.",
            review_terms=["image transformations", "affine", "translation"],
        ),
        mcq(
            prompt="Which homogeneous coordinate triple represents the same 2D point as (3, 2, 1)?",
            correct="(6, 4, 2).",
            distractors=[
                "(6, 2, 1).",
                "(3, 2, 2).",
                "(4, 6, 2).",
            ],
            explanation="Homogeneous coordinates represent the same Cartesian point if one triple is a nonzero scalar multiple of the other. Doubling all three entries keeps the same point.",
            review_terms=["homogeneous coordinates"],
        ),
        mcq(
            prompt="Which of the following belongs to photometric rather than geometric change in the lecture taxonomy?",
            correct="Brightness or exposure change.",
            distractors=[
                "Rotation.",
                "Translation.",
                "Uniform scale.",
            ],
            explanation="The course separates geometric transformations such as rotation and scale from photometric changes such as brightness and exposure variation.",
            review_terms=["geometric", "photometric", "brightness", "exposure"],
        ),
        mcq(
            prompt="Why can translation not be expressed by a 2x2 matrix acting on ordinary 2D coordinates alone?",
            correct="Because translation is not a linear operation in standard 2D coordinates and needs an augmented representation such as homogeneous coordinates.",
            distractors=[
                "Because translation changes the dimensionality of the image.",
                "Because translation is always nonlinear in homogeneous coordinates.",
                "Because translation can only be defined for 3D points.",
            ],
            explanation="A 2x2 linear matrix can rotate, scale, or shear about the origin, but adding translation requires moving to homogeneous coordinates so that the translation term is included in matrix form.",
            review_terms=["translation", "homogeneous coordinates", "image transformations"],
        ),
        mcq(
            prompt="If one image differs from another by rotation, scaling, or translation, what is the lecture emphasizing?",
            correct="The need for feature descriptions that remain stable under geometric transformations.",
            distractors=[
                "The need to convert all local features into binary masks.",
                "The fact that only intensity histograms matter for matching.",
                "The fact that every geometric change can be ignored completely during recognition.",
            ],
            explanation="The transformations slide is connected to invariance: a good local-feature pipeline should still function when these geometric changes occur between views.",
            use_image=True,
            review_terms=["image transformations", "geometric", "invariance"],
        ),
    ],
    "Image alignment": [
        mcq(
            prompt="In the panorama-stitching workflow from the lecture, which step comes immediately after extracting features?",
            correct="Match features between the two images.",
            distractors=[
                "Compute dense optical flow over the whole video.",
                "Assign a semantic label to every pixel.",
                "Estimate a bag-of-words histogram for each crop first.",
            ],
            explanation="The pipeline shown in the lecture is organized as interest-point detection, feature extraction, feature matching, and then image alignment.",
            review_terms=["panorama stitching", "extract features", "match features", "align images"],
        ),
        mcq(
            prompt="Why does panorama stitching need local features rather than only raw pixel overlap?",
            correct="Because corresponding features provide reliable matches that can be used to estimate the transformation between images.",
            distractors=[
                "Because raw pixels cannot be represented numerically.",
                "Because panorama stitching ignores geometric transformations entirely.",
                "Because local features automatically label every object in the scene.",
            ],
            explanation="The lecture motivates feature extraction precisely because two images need reliable correspondences before they can be aligned into one panorama.",
            review_terms=["why extract features", "panorama stitching", "combine them"],
        ),
        mcq(
            prompt="What is the main goal of the final alignment step in the lecture pipeline?",
            correct="To find a transformation that makes corresponding scene content line up across the images.",
            distractors=[
                "To estimate a gray-level histogram for each image independently.",
                "To assign one class label to the entire panorama.",
                "To remove all local features before stitching.",
            ],
            explanation="Alignment is successful when corresponding structures from the two views overlap after applying the estimated transformation.",
            review_terms=["align images", "panorama stitching", "transformation"],
        ),
        mcq(
            prompt="If feature matches are wrong, what is the most direct consequence for image alignment?",
            correct="The estimated transformation is likely to be wrong, so the images will not line up correctly.",
            distractors=[
                "The image automatically becomes segmentation-ready.",
                "The descriptors become rotation invariant automatically.",
                "The bag-of-words histogram becomes denser.",
            ],
            explanation="Alignment depends on correct correspondences. If the matches are wrong, the transformation fitted from those matches will also be wrong.",
            review_terms=["match features", "align images", "panorama stitching"],
        ),
        mcq(
            prompt="If the before-and-after overlay lines up after warping, what has the alignment stage achieved?",
            correct="It has brought corresponding image content into a common coordinate frame closely enough for stitching or comparison.",
            distractors=[
                "It has converted the image pair into a texture histogram.",
                "It has found every object category in the scene.",
                "It has replaced feature matching with edge detection.",
            ],
            explanation="The visual sign of good alignment is that corresponding content overlays correctly after the estimated transform is applied.",
            use_image=True,
            review_terms=["align images", "automatic panoramas", "panorama stitching"],
        ),
    ],
}
