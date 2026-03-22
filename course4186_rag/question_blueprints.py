QUESTION_BLUEPRINTS = {
    "Computer vision foundations": [
        {
            "prompt": "Which description best captures the aim of computer vision in this course?",
            "correct": "To build methods that interpret images and video for tasks such as recognition, measurement, and scene understanding.",
            "distractors": [
                "To store every image as an uncompressed pixel table without any interpretation.",
                "To guarantee perfect 3D reconstruction from a single pixel.",
                "To replace all machine learning with manual camera calibration formulas.",
            ],
            "explanation": "The introductory material frames computer vision as extracting useful understanding from visual data.",
        },
        {
            "prompt": "Why is computer vision considered difficult in the introductory lecture?",
            "correct": "Because real images contain variation, clutter, occlusion, and other factors that make the same object look very different.",
            "distractors": [
                "Because digital images cannot be represented as numeric arrays.",
                "Because every vision task uses exactly the same algorithm and data representation.",
                "Because objects always appear in front of a clean and uniform background.",
            ],
            "explanation": "The lecture highlights intra-class variation, clutter, and occlusion as central sources of difficulty.",
        },
        {
            "prompt": "Which of the following is a typical computer vision task rather than a low-level storage operation?",
            "correct": "Recognizing an object category from an input image.",
            "distractors": [
                "Saving the same image to disk under a different file name.",
                "Replacing a folder path in the operating system.",
                "Counting how many bytes a file occupies without using the image content.",
            ],
            "explanation": "Computer vision is about interpreting image content, such as recognition and scene understanding.",
        },
        {
            "prompt": "What makes a vision system different from simply recording pixels?",
            "correct": "A vision system tries to infer structure or meaning from the visual input, not just keep the raw image values.",
            "distractors": [
                "A vision system can only operate on handwritten images.",
                "A vision system never uses any numerical representation.",
                "A vision system is defined only by the camera focal length.",
            ],
            "explanation": "The course introduction distinguishes understanding visual content from merely storing raw image measurements.",
        },
        {
            "prompt": "The figure shows applications such as robotics, medical imaging, and recognition. What is it meant to show about computer vision as a field?",
            "correct": "Computer vision supports many different application areas that rely on interpreting visual data.",
            "distractors": [
                "Only camera calibration matters in computer vision.",
                "Every computer vision problem reduces to one fixed convolution kernel.",
                "Computer vision is limited to one single application domain.",
            ],
            "explanation": "The overview material uses applications to show the breadth of computer vision as a field.",
            "use_image": True,
        },
    ],
    "Image filtering and convolution": [
        {
            "prompt": "If a 3x3 filter is applied to one interior pixel, how many multiplications are needed before the results are summed?",
            "correct": "9 multiplications.",
            "distractors": [
                "3 multiplications.",
                "6 multiplications.",
                "18 multiplications.",
            ],
            "explanation": "A 3x3 kernel contains 9 weights, so one output value uses 9 multiply terms before summation.",
        },
        {
            "prompt": "If a 2D filter can be written as h(x,y) = a(x)b(y), what is the main computational advantage?",
            "correct": "The 2D convolution can be carried out as two 1D convolutions with fewer operations.",
            "distractors": [
                "The filter must be applied four times instead of once.",
                "The filter can only be used on binary images.",
                "The image no longer needs any interpolation or resampling.",
            ],
            "explanation": "A separable filter reduces the cost by replacing one 2D filtering pass with two cheaper 1D passes.",
        },
        {
            "prompt": "Which statement correctly distinguishes convolution from cross-correlation?",
            "correct": "Convolution flips the kernel, while cross-correlation uses the kernel in its original order.",
            "distractors": [
                "Cross-correlation flips the kernel, while convolution keeps it unchanged.",
                "Both operations always produce identical results for every kernel.",
                "Convolution can only be used for color images, while cross-correlation can only be used for grayscale images.",
            ],
            "explanation": "The lecture material explicitly distinguishes convolution from cross-correlation by the kernel flip.",
        },
        {
            "prompt": "What is the main tradeoff when a mean or Gaussian filter is applied before edge detection?",
            "correct": "Noise is reduced, but some edge sharpness is also blurred.",
            "distractors": [
                "Noise is increased, but edges become perfectly exact.",
                "The image becomes scale invariant without any descriptor.",
                "The field of view becomes narrower.",
            ],
            "explanation": "Smoothing stabilizes derivatives and later processing, but it also softens high-frequency structures such as sharp edges.",
        },
        {
            "prompt": "In the local-neighborhood and kernel figure, how is one output pixel computed?",
            "correct": "An output pixel is obtained as a weighted sum of nearby input pixels.",
            "distractors": [
                "A 3D point is triangulated from stereo disparity.",
                "A camera intrinsic matrix is being calibrated.",
                "A feature descriptor is being quantized into visual words.",
            ],
            "explanation": "Kernel-and-neighborhood diagrams show local filtering, where each output value depends on a weighted neighborhood.",
            "use_image": True,
        },
    ],
    "Edge detection": [
        {
            "prompt": "What is the main goal of edge detection?",
            "correct": "To locate strong visual discontinuities that usually mark object boundaries or structure changes.",
            "distractors": [
                "To estimate the focal length from a single pixel intensity.",
                "To quantize local descriptors into visual words.",
                "To align two images by solving for a camera baseline.",
            ],
            "explanation": "Edge detection highlights abrupt intensity changes, which often correspond to meaningful boundaries.",
        },
        {
            "prompt": "Why is smoothing often applied before computing image derivatives for edge detection?",
            "correct": "Because derivatives amplify noise, so smoothing makes the edge response more stable.",
            "distractors": [
                "Because smoothing converts edges directly into segmentation masks.",
                "Because derivatives can only be computed on binary images.",
                "Because smoothing guarantees the image has no corners.",
            ],
            "explanation": "Noise creates unstable derivative responses, so smoothing is commonly used before edge detection.",
        },
        {
            "prompt": "Where should the strongest edge response appear?",
            "correct": "At locations where the image intensity changes abruptly over a short distance.",
            "distractors": [
                "At perfectly uniform flat regions with constant intensity.",
                "At every pixel with the same response regardless of structure.",
                "Only at the optical center of the camera.",
            ],
            "explanation": "Edges correspond to rapid local intensity variation rather than flat regions.",
        },
        {
            "prompt": "Why are edge maps useful in later vision tasks?",
            "correct": "They preserve important boundary structure while discarding much of the less informative interior appearance.",
            "distractors": [
                "They recover absolute scene depth without any second view.",
                "They replace the need for feature descriptors in all matching tasks.",
                "They always provide the final semantic label of every object.",
            ],
            "explanation": "Edge maps provide a compact representation of salient structural changes in the image.",
        },
        {
            "prompt": "If a figure highlights object outlines while suppressing flat regions, what output is it trying to produce?",
            "correct": "Detecting boundaries by finding large local intensity changes.",
            "distractors": [
                "Estimating optical flow between frames.",
                "Building a bag-of-words histogram for retrieval.",
                "Projecting a 3D line through a pinhole camera.",
            ],
            "explanation": "Boundary-outline figures are the classic visual explanation for edge detection.",
            "use_image": True,
        },
    ],
    "Image resampling": [
        {
            "prompt": "What is the main issue when an image is downsampled without adequate low-pass filtering?",
            "correct": "Aliasing artifacts can appear because high-frequency content is sampled too coarsely.",
            "distractors": [
                "The image automatically gains more geometric detail.",
                "The camera focal length becomes smaller.",
                "The image is forced into a corner detector response map.",
            ],
            "explanation": "Downsampling without proper filtering can fold high frequencies into misleading lower-frequency patterns.",
        },
        {
            "prompt": "What is interpolation used for during image resampling?",
            "correct": "To estimate pixel values at new sample locations that were not originally present.",
            "distractors": [
                "To recover the exact 3D pose of the camera.",
                "To convert edges into texture descriptors.",
                "To cluster SIFT features into visual words.",
            ],
            "explanation": "Interpolation fills in values when the image grid changes, especially during resizing or warping.",
        },
        {
            "prompt": "If an image is enlarged, what does resampling mainly need to provide?",
            "correct": "New pixel values on a denser grid using an interpolation rule.",
            "distractors": [
                "A disparity map from two rectified images.",
                "A semantic segmentation label for each region.",
                "A fundamental matrix between the original and enlarged images.",
            ],
            "explanation": "Upsampling creates new sample positions that require estimated intensities.",
        },
        {
            "prompt": "Which statement best describes image resampling?",
            "correct": "It changes how an image is sampled in space while trying to preserve useful visual information.",
            "distractors": [
                "It learns a decision boundary directly from labeled training data.",
                "It computes corner strength from second-moment statistics.",
                "It enforces the coplanarity relation between two camera rays.",
            ],
            "explanation": "Resampling is about changing spatial sampling or resolution rather than recognition or stereo geometry.",
        },
        {
            "prompt": "A figure comparing different sampling grids or resized outputs is mainly asking you to reason about which issue?",
            "correct": "How sampling density and interpolation affect the resampled image.",
            "distractors": [
                "How epipolar lines reduce the correspondence search space.",
                "How CNN pooling creates invariance to translation.",
                "How camera motion is recovered from feature tracks.",
            ],
            "explanation": "Grid-comparison figures are commonly used to explain sampling, resizing, and interpolation behavior.",
            "use_image": True,
        },
    ],
    "Texture analysis": [
        {
            "prompt": "What type of visual signal is texture analysis designed to capture?",
            "correct": "Repeated local appearance patterns such as grass, fabric, or brick-like structure.",
            "distractors": [
                "Exact camera intrinsic parameters.",
                "The optical center of the imaging system.",
                "Only the strongest single edge in the whole image.",
            ],
            "explanation": "Texture analysis focuses on repeated local structure rather than isolated geometric primitives.",
        },
        {
            "prompt": "Why can texture be useful for recognition?",
            "correct": "Because different surfaces often produce characteristic repeated local patterns that are discriminative.",
            "distractors": [
                "Because texture removes the need for any segmentation or matching.",
                "Because texture directly returns a 3D point cloud.",
                "Because texture always stays unchanged under every possible transformation.",
            ],
            "explanation": "Texture cues help distinguish surfaces and categories when local appearance patterns differ.",
        },
        {
            "prompt": "Which example best matches a texture-analysis problem?",
            "correct": "Distinguishing grass, leaves, or other repeated surface patterns from one another.",
            "distractors": [
                "Estimating the baseline between two calibrated cameras from one point.",
                "Choosing a focal length to widen the field of view.",
                "Projecting a 3D point through a pinhole model.",
            ],
            "explanation": "Texture analysis is about repeated appearance patterns, not camera geometry.",
        },
        {
            "prompt": "Why is texture analysis different from corner detection?",
            "correct": "Texture analysis describes repeated appearance over regions, while corner detection looks for localized point-like structure.",
            "distractors": [
                "Texture analysis uses only 3D points, while corner detection uses only 2D lines.",
                "Texture analysis estimates depth, while corner detection estimates focal length.",
                "Texture analysis requires two cameras, while corner detection requires four cameras.",
            ],
            "explanation": "Texture and corners capture different kinds of information: regional pattern versus localized feature points.",
        },
        {
            "prompt": "A figure showing surfaces such as grass or leaves is best used to test understanding of which topic?",
            "correct": "Describing repeated local patterns for texture recognition or classification.",
            "distractors": [
                "Recovering optical flow between adjacent video frames.",
                "Constraining correspondences with epipolar lines.",
                "Aligning two views by estimating a global warp.",
            ],
            "explanation": "Natural-surface imagery is a standard example when explaining texture analysis.",
            "use_image": True,
        },
    ],
    "Convolutional neural networks": [
        {
            "prompt": "Why are convolutional layers well suited for images?",
            "correct": "They reuse local filters across the image, allowing shared feature detectors at many locations.",
            "distractors": [
                "They require a different weight matrix for every pixel location.",
                "They replace the need for any labeled data.",
                "They compute the essential matrix from a stereo pair.",
            ],
            "explanation": "Convolution exploits local connectivity and weight sharing, which matches the structure of images.",
        },
        {
            "prompt": "What is the role of pooling in a CNN?",
            "correct": "To reduce spatial resolution while preserving strong local evidence and improving robustness.",
            "distractors": [
                "To calibrate the camera's focal length.",
                "To generate a semantic segmentation mask directly from a single pixel.",
                "To convert every descriptor into a geometric transformation matrix.",
            ],
            "explanation": "Pooling compresses feature maps and can make the network less sensitive to small shifts.",
        },
        {
            "prompt": "How do deeper CNN layers usually differ from earlier layers?",
            "correct": "They tend to represent more abstract and task-specific visual patterns.",
            "distractors": [
                "They always contain fewer parameters than the input image.",
                "They stop using any local spatial information at all.",
                "They can only respond to raw grayscale intensity values.",
            ],
            "explanation": "Early layers often capture simple patterns while deeper layers combine them into more semantic features.",
        },
        {
            "prompt": "Which statement best matches a classification use of CNNs?",
            "correct": "The network learns feature hierarchies that help separate image categories with a decision boundary.",
            "distractors": [
                "The network guarantees exact geometric alignment between any two views.",
                "The network enforces brightness constancy for optical flow.",
                "The network computes vanishing points by analytic projection only.",
            ],
            "explanation": "CNNs for classification learn increasingly useful representations for separating classes.",
        },
        {
            "prompt": "If a figure shows feature maps or stacked learned filters, what core CNN idea is being emphasized?",
            "correct": "CNNs build hierarchical visual features from local convolutional responses.",
            "distractors": [
                "Stereo matching is restricted to an epipolar line.",
                "A bag-of-words model quantizes descriptors into a histogram.",
                "A pinhole camera maps 3D points by perspective projection.",
            ],
            "explanation": "Feature-map diagrams are standard visual explanations for convolutional neural networks.",
            "use_image": True,
        },
    ],
    "Image segmentation": [
        {
            "prompt": "What is the main goal of image segmentation?",
            "correct": "To divide the image into meaningful regions or objects.",
            "distractors": [
                "To estimate the camera baseline from a single edge.",
                "To build a scale-space descriptor for local keypoints.",
                "To recover optical flow by brightness constancy alone.",
            ],
            "explanation": "Segmentation partitions an image into coherent parts rather than isolated feature points.",
        },
        {
            "prompt": "Which output best matches a segmentation system?",
            "correct": "A region or mask assignment for pixels belonging to different objects or classes.",
            "distractors": [
                "A single focal-length value for the whole image.",
                "A histogram of visual words only.",
                "A list of camera rotations without any pixel labels.",
            ],
            "explanation": "Segmentation outputs pixel- or region-level assignments, not just image-level labels.",
        },
        {
            "prompt": "Why is segmentation harder than simple edge detection?",
            "correct": "Because segmentation must group pixels into coherent regions or objects, not just mark local discontinuities.",
            "distractors": [
                "Because segmentation works only on binary images.",
                "Because segmentation cannot use any appearance information.",
                "Because segmentation requires every scene to contain stereo pairs.",
            ],
            "explanation": "Segmentation goes beyond local boundaries and reasons about region consistency and object extent.",
        },
        {
            "prompt": "Which statement best matches semantic segmentation?",
            "correct": "Pixels are assigned to categories such as road, zebra, sky, or person.",
            "distractors": [
                "Only a single corner point is marked in each object.",
                "Each image is reduced to a 3D point cloud directly.",
                "Every descriptor is converted into a camera matrix.",
            ],
            "explanation": "Semantic segmentation labels pixels or regions with category information.",
        },
        {
            "prompt": "If a figure colors each object region or mask separately, what task is being demonstrated?",
            "correct": "Partitioning the image into meaningful object or semantic regions.",
            "distractors": [
                "Estimating the disparity of corresponding pixels between two views.",
                "Describing repeated local texture for retrieval.",
                "Normalizing a feature descriptor for scale and rotation.",
            ],
            "explanation": "Colored regions or masks are the clearest visual cue for segmentation.",
            "use_image": True,
        },
    ],
    "Harris corners and local features": [
        {
            "prompt": "In Harris corner detection, what does it usually mean if one eigenvalue is large and the other is small?",
            "correct": "The local patch is likely an edge rather than a corner.",
            "distractors": [
                "The local patch is a strong corner.",
                "The local patch is completely flat.",
                "The camera focal length is too small.",
            ],
            "explanation": "An edge changes strongly in one direction only, which gives one large and one small eigenvalue.",
        },
        {
            "prompt": "What does it usually mean if both eigenvalues of the Harris matrix are large?",
            "correct": "The local patch is a corner because intensity changes strongly in two directions.",
            "distractors": [
                "The local patch is flat.",
                "The local patch is a pure edge.",
                "The image has already been rectified for stereo.",
            ],
            "explanation": "Corners are the locations where the image patch varies strongly under small shifts in both x and y directions.",
        },
        {
            "prompt": "What does it usually mean if both eigenvalues of the Harris matrix are small?",
            "correct": "The local patch is a flat region with little intensity change.",
            "distractors": [
                "The local patch is a strong corner.",
                "The local patch is a strong edge.",
                "The descriptor has become rotation invariant.",
            ],
            "explanation": "Flat regions do not change much under small shifts, so both directional variations remain small.",
        },
        {
            "prompt": "According to the revision material, which property is Harris corner detection not invariant to?",
            "correct": "Scaling.",
            "distractors": [
                "Image translation.",
                "Image rotation.",
                "Intensity shift.",
            ],
            "explanation": "The revision slides note that Harris is shift- and rotation-friendly in common settings, but not scale invariant.",
        },
        {
            "prompt": "In the Harris example, what quantity is the highlighted local window used to evaluate under small shifts?",
            "correct": "The Harris matrix and the resulting corner response for that window.",
            "distractors": [
                "A stereo disparity map for the whole image.",
                "A CNN pooling score.",
                "A camera projection matrix.",
            ],
            "explanation": "Tutorial Harris questions ask students to compute derivatives, form the Harris matrix, and then evaluate the corner response inside a selected window.",
            "use_image": True,
        },
    ],
    "SIFT and scale-invariant features": [
        {
            "prompt": "What role does the Difference-of-Gaussian stage play in SIFT?",
            "correct": "It detects candidate keypoints as extrema across scale space.",
            "distractors": [
                "It computes the final bag-of-words histogram directly.",
                "It rectifies a stereo pair into scanlines.",
                "It replaces the descriptor with a semantic class label.",
            ],
            "explanation": "SIFT first searches for stable extrema across scale space before building descriptors.",
        },
        {
            "prompt": "A full SIFT descriptor uses a 4x4 grid of cells with 8 orientation bins per cell. What is the descriptor dimension?",
            "correct": "128.",
            "distractors": [
                "32.",
                "64.",
                "256.",
            ],
            "explanation": "The lecture slides state 16 cells times 8 orientation bins, which gives 128 dimensions.",
        },
        {
            "prompt": "Why can SIFT remain fairly robust under illumination change?",
            "correct": "Because the descriptor is built from local gradients rather than raw pixel values alone.",
            "distractors": [
                "Because SIFT assumes every image is captured with the same exposure.",
                "Because SIFT always normalizes the camera focal length.",
                "Because SIFT ignores all local structure inside the patch.",
            ],
            "explanation": "The tutorial answer notes that SIFT depends on gradient information, which is more stable than raw brightness values.",
        },
        {
            "prompt": "How does SIFT achieve rotation invariance?",
            "correct": "It aligns the descriptor to a dominant local orientation before describing the patch.",
            "distractors": [
                "It removes all orientation information from the image gradients.",
                "It estimates rotation by comparing stereo disparity values.",
                "It requires the input images to be physically rotated back before matching.",
            ],
            "explanation": "Orientation assignment normalizes the patch before descriptor formation, which makes matching more stable under rotation.",
        },
        {
            "prompt": "In a SIFT scale-space figure, what step is being decided before the descriptor is built?",
            "correct": "Finding keypoints as local extrema across neighboring locations and scales.",
            "distractors": [
                "Computing a stereo baseline from two cameras.",
                "Estimating optical flow under brightness constancy.",
                "Projecting a 3D point through a pinhole camera.",
            ],
            "explanation": "The tutorial questions explicitly refer to identifying a single keypoint by checking extrema in scale space.",
            "use_image": True,
        },
    ],
    "Bag-of-words image retrieval": [
        {
            "prompt": "What is the main idea of a bag-of-words model for image retrieval?",
            "correct": "Represent the image by a histogram of visual words built from local descriptors.",
            "distractors": [
                "Represent the image only by a single corner point.",
                "Represent the image as a full semantic segmentation mask.",
                "Represent the image by the camera intrinsic matrix alone.",
            ],
            "explanation": "The image is summarized by counts of quantized local visual patterns.",
        },
        {
            "prompt": "What is a visual vocabulary in a bag-of-words retrieval pipeline?",
            "correct": "A set of prototype descriptor clusters used to quantize local features into visual words.",
            "distractors": [
                "A list of all object names in the dataset.",
                "A matrix that enforces the epipolar constraint.",
                "A collection of camera baselines from stereo pairs.",
            ],
            "explanation": "Visual words come from clustering local descriptors into representative prototypes.",
        },
        {
            "prompt": "Why can bag-of-words support large-scale retrieval?",
            "correct": "Because histograms of visual words give a compact representation that can be compared efficiently.",
            "distractors": [
                "Because it stores every image as a full 3D point cloud.",
                "Because it removes the need for any feature extraction.",
                "Because it estimates optical flow between all image pairs.",
            ],
            "explanation": "Histogram representations are compact and efficient for indexing and matching.",
        },
        {
            "prompt": "What information does a plain bag-of-words model usually lose?",
            "correct": "The exact spatial arrangement of the local features in the image.",
            "distractors": [
                "Whether the image contains any pixel values at all.",
                "Whether the camera uses perspective projection.",
                "Whether edges correspond to intensity changes.",
            ],
            "explanation": "A plain histogram keeps counts of visual words but not their exact geometry.",
        },
        {
            "prompt": "If local descriptors are grouped into bins or histogram counts, which retrieval pipeline step is being shown?",
            "correct": "Quantizing local features into visual words for image retrieval.",
            "distractors": [
                "Estimating depth from disparity.",
                "Computing optical flow under brightness constancy.",
                "Projecting 3D points by a pinhole camera.",
            ],
            "explanation": "Cluster-and-histogram figures are the standard visual explanation for bag-of-words retrieval.",
            "use_image": True,
        },
    ],
    "Geometric transformations": [
        {
            "prompt": "Which homogeneous coordinate represents the same 2D point as (3, 2, 1)?",
            "correct": "(6, 4, 2).",
            "distractors": [
                "(6, 2, 1).",
                "(3, 2, 2).",
                "(4, 6, 2).",
            ],
            "explanation": "Homogeneous coordinates that differ by a nonzero scale factor represent the same Cartesian point.",
        },
        {
            "prompt": "Which transformation class combines linear transformations with translation?",
            "correct": "An affine transformation.",
            "distractors": [
                "A Gaussian transformation.",
                "A bag-of-words transformation.",
                "A disparity transformation.",
            ],
            "explanation": "The course notes define affine transformations as linear transforms together with translations.",
        },
        {
            "prompt": "Which motion cannot be represented by a 2x2 linear matrix alone and is usually added via homogeneous coordinates?",
            "correct": "Translation.",
            "distractors": [
                "Rotation about the origin.",
                "Uniform scaling.",
                "Shear.",
            ],
            "explanation": "A pure 2x2 linear transform handles operations about the origin, while translation is added naturally in homogeneous coordinates.",
        },
        {
            "prompt": "In the simple image-alignment case where the transform is pure translation, what is the least-squares estimate from feature matches?",
            "correct": "The mean displacement vector across the matches.",
            "distractors": [
                "The largest single displacement vector only.",
                "The disparity with the smallest magnitude.",
                "The camera principal point.",
            ],
            "explanation": "The lecture slides state that for translation, the least-squares solution is the average displacement of the matched points.",
        },
        {
            "prompt": "If one image is translated, rotated, or scaled relative to another, what concept is the transformed-image figure explaining?",
            "correct": "Image coordinates are being changed by a global geometric transform such as rotation, scaling, shear, or translation.",
            "distractors": [
                "A SIFT descriptor is being converted into a histogram.",
                "Stereo disparity is being converted into depth.",
                "A CNN is assigning semantic classes to pixels.",
            ],
            "explanation": "The course uses warped coordinates and transformed images to explain planar transforms and warping.",
            "use_image": True,
        },
    ],
    "Image alignment": [
        {
            "prompt": "Given matched features between images A and B, what is the direct goal of image alignment?",
            "correct": "Estimate a transform T that best agrees with the matches.",
            "distractors": [
                "Estimate a segmentation mask for each image independently.",
                "Replace all feature matches with a CNN classifier.",
                "Recover depth from a single image without any geometry.",
            ],
            "explanation": "The alignment lectures frame the task as solving for the transform that best fits corresponding matches.",
        },
        {
            "prompt": "In the lecture alignment pipeline, what is typically estimated after computing and matching image features?",
            "correct": "A homography or other geometric transform between the two images.",
            "distractors": [
                "A bag-of-words vocabulary.",
                "A dense semantic segmentation map.",
                "A stereo baseline from a calibrated rig.",
            ],
            "explanation": "The course alignment algorithm computes features, matches them, and then estimates a homography from those matches.",
        },
        {
            "prompt": "Why is RANSAC used before the final least-squares fit in image alignment?",
            "correct": "To remove outlier matches and keep the transform supported by the largest inlier set.",
            "distractors": [
                "To increase the image brightness before matching.",
                "To convert the images into grayscale histograms.",
                "To replace the transform with semantic labels.",
            ],
            "explanation": "RANSAC is used to reject mismatches so the final transform is estimated from geometrically consistent correspondences.",
        },
        {
            "prompt": "In inverse warping, where does each output pixel g(x', y') sample its value from?",
            "correct": "From the source image location T^-1(x', y') in the input image.",
            "distractors": [
                "From the output image location T(x', y') itself.",
                "From the camera center in 3D space.",
                "From the largest SIFT bin only.",
            ],
            "explanation": "Inverse warping fills each output pixel by mapping backward through the inverse transform into the source image.",
        },
        {
            "prompt": "If the before-and-after overlay lines up after warping, what does that tell you about the alignment result?",
            "correct": "Corresponding structures from the two images overlap consistently after warping.",
            "distractors": [
                "Every image edge disappears after filtering.",
                "The camera focal length changes automatically.",
                "Optical flow is computed for every video frame.",
            ],
            "explanation": "Image alignment is successful when the warped source and target image content line up well.",
            "use_image": True,
        },
    ],
    "Camera model and projection": [
        {
            "prompt": "A pinhole camera has focal length 5 mm and pixel size 0.02 mm. What is the focal length in pixel units?",
            "correct": "250 pixels.",
            "distractors": [
                "25 pixels.",
                "100 pixels.",
                "500 pixels.",
            ],
            "explanation": "Dividing 5 mm by 0.02 mm per pixel gives 250 pixels for both fx and fy.",
        },
        {
            "prompt": "Using focal length 5 mm, pixel size 0.02 mm, and principal point (500, 500), what is the image projection of the 3D point (100, 150, 800) when world and camera coordinates are aligned?",
            "correct": "(531.25, 546.875).",
            "distractors": [
                "(31.25, 46.875).",
                "(525, 537.5).",
                "(546.875, 531.25).",
            ],
            "explanation": "fx = fy = 250, so x = 250*(100/800)+500 = 531.25 and y = 250*(150/800)+500 = 546.875.",
        },
        {
            "prompt": "What is the effect of increasing focal length while keeping the sensor size fixed?",
            "correct": "The field of view becomes narrower and objects appear larger in the image.",
            "distractors": [
                "The field of view becomes wider and objects appear smaller.",
                "The camera no longer needs a principal point.",
                "Stereo disparity becomes zero everywhere.",
            ],
            "explanation": "The course materials link larger focal length with narrower viewing angle and larger projected image scale.",
        },
        {
            "prompt": "Why do vanishing points appear in perspective images?",
            "correct": "Parallel 3D lines that are not parallel to the image plane can project to image lines that meet at a point.",
            "distractors": [
                "Because every 3D line stays parallel after perspective projection.",
                "Because vanishing points are caused only by Gaussian filtering.",
                "Because the principal point always lies on every 3D line.",
            ],
            "explanation": "Vanishing points are a direct consequence of perspective projection of sets of parallel scene lines.",
        },
        {
            "prompt": "The pinhole-camera figure with rays passing through a small aperture is most likely illustrating which process?",
            "correct": "Perspective projection from 3D scene points onto a 2D image plane.",
            "distractors": [
                "Optical flow estimation between adjacent video frames.",
                "Quantization of SIFT descriptors into visual words.",
                "Semantic segmentation of foreground objects.",
            ],
            "explanation": "Ray diagrams with a small aperture are the standard visualization of the pinhole camera model.",
            "use_image": True,
        },
    ],
    "Stereo vision and depth": [
        {
            "prompt": "In the stereo relation z = fB / d, what happens to depth if disparity doubles while f and B stay fixed?",
            "correct": "The depth is halved.",
            "distractors": [
                "The depth doubles.",
                "The depth stays the same.",
                "The depth becomes zero regardless of the scene point.",
            ],
            "explanation": "Depth is inversely proportional to disparity, so doubling disparity cuts depth by a factor of two.",
        },
        {
            "prompt": "Why is stereo rectification applied before dense matching in a calibrated stereo pair?",
            "correct": "It turns epipolar lines into horizontal scanlines, so the search becomes one-dimensional.",
            "distractors": [
                "It removes the need for camera calibration entirely.",
                "It converts disparity directly into a segmentation mask.",
                "It makes every image patch rotation invariant.",
            ],
            "explanation": "The stereo lecture states that rectification makes pixel motion horizontal and simplifies correspondence search.",
        },
        {
            "prompt": "Two points in the same stereo pair have disparities 8 and 16 pixels. Which point is closer to the cameras?",
            "correct": "The point with disparity 16 pixels.",
            "distractors": [
                "The point with disparity 8 pixels.",
                "They are at the same depth.",
                "The answer depends only on the principal point, not disparity.",
            ],
            "explanation": "Larger disparity means the scene point is closer when the stereo rig is fixed.",
        },
        {
            "prompt": "In the basic stereo matching algorithm, which quantity is computed immediately before depth?",
            "correct": "Disparity, computed as the horizontal difference x - x'.",
            "distractors": [
                "The Harris corner response.",
                "The dominant SIFT orientation.",
                "The vanishing point of the scene.",
            ],
            "explanation": "The lecture pipeline is: find correspondence, compute disparity, then convert disparity to depth.",
        },
        {
            "prompt": "The paired stereo images with a horizontal shift of the same object are most likely illustrating which idea?",
            "correct": "Depth can be inferred from disparity between corresponding image points.",
            "distractors": [
                "A local filter kernel is convolved with the image.",
                "A semantic segmentation mask has been assigned to every pixel.",
                "A bag-of-words histogram is being constructed for retrieval.",
            ],
            "explanation": "Horizontal offset between stereo views is the classic cue for disparity and depth.",
            "use_image": True,
        },
    ],
    "Epipolar geometry": [
        {
            "prompt": "For a point selected in image 1, where can its corresponding point lie in image 2 according to epipolar geometry?",
            "correct": "On the associated epipolar line.",
            "distractors": [
                "At any pixel in the second image with no geometric restriction.",
                "Only at the principal point.",
                "Only at a Harris corner with the largest response.",
            ],
            "explanation": "A point in one view maps to an epipolar line in the other view, which is the core search constraint.",
        },
        {
            "prompt": "Which matrix is used in the calibrated two-view case?",
            "correct": "The essential matrix.",
            "distractors": [
                "The homography voting matrix.",
                "The Harris corner matrix.",
                "The pooling matrix.",
            ],
            "explanation": "The essential matrix encodes the calibrated epipolar relation between corresponding points.",
        },
        {
            "prompt": "Which matrix is used to express the epipolar relation in image coordinates when calibration is not assumed?",
            "correct": "The fundamental matrix.",
            "distractors": [
                "The Gaussian smoothing matrix.",
                "The segmentation confusion matrix.",
                "The SIFT orientation matrix.",
            ],
            "explanation": "The fundamental matrix links image points across uncalibrated views.",
        },
        {
            "prompt": "What geometric plane contains the two camera centers and a 3D scene point observed in both views?",
            "correct": "The epipolar plane.",
            "distractors": [
                "The image plane only.",
                "The focal plane only.",
                "A segmentation region.",
            ],
            "explanation": "The two camera centers and the 3D point define the epipolar plane, from which the epipolar lines are induced.",
        },
        {
            "prompt": "The two-camera figure with a baseline and matching lines is most likely illustrating which practical benefit?",
            "correct": "Correspondence search is reduced from a 2D area to a 1D search along an epipolar line.",
            "distractors": [
                "A local descriptor is normalized for scale and orientation.",
                "A bag-of-words pipeline is clustering local features.",
                "A classifier is reducing feature maps by pooling.",
            ],
            "explanation": "Epipolar geometry is used in practice to constrain where a valid match can appear in the second image.",
            "use_image": True,
        },
    ],
    "Structure from motion": [
        {
            "prompt": "How does structure from motion differ from traditional stereo vision in the course materials?",
            "correct": "SfM can work with multiple images and arbitrary camera motion while jointly estimating camera poses and 3D structure.",
            "distractors": [
                "SfM requires exactly two rectified cameras with fixed known baseline.",
                "SfM ignores camera motion and estimates only image smoothing.",
                "SfM is only another name for edge detection.",
            ],
            "explanation": "The tutorial notes contrast SfM with stereo by emphasizing arbitrary motion and joint estimation of structure and camera pose.",
        },
        {
            "prompt": "What kind of input does an SfM pipeline need?",
            "correct": "Multiple overlapping 2D images of the same scene, optionally with known intrinsics.",
            "distractors": [
                "A single segmented image only.",
                "A 1D signal with no image coordinates.",
                "Only a disparity map with no original images.",
            ],
            "explanation": "The tutorial Q&A states that SfM uses multiple 2D images of the same scene and may also use known camera intrinsics.",
        },
        {
            "prompt": "Why is RANSAC applied to feature matches in a typical SfM pipeline?",
            "correct": "To reject outlier correspondences before estimating the two-view geometry.",
            "distractors": [
                "To enlarge the camera baseline automatically.",
                "To convert the matches into CNN feature maps.",
                "To remove the need for camera pose estimation.",
            ],
            "explanation": "The tutorial material describes RANSAC as fitting a model, counting inliers, and rejecting outliers among matches.",
        },
        {
            "prompt": "What does the reconstructed sparse point cloud in SfM represent?",
            "correct": "Estimated 3D scene points recovered from multiple images.",
            "distractors": [
                "A histogram of visual words for retrieval.",
                "A dense semantic segmentation map.",
                "Only the set of camera intrinsics.",
            ],
            "explanation": "Sparse 3D points are a standard SfM output, reconstructed from repeated observations across views.",
        },
        {
            "prompt": "The figure with several camera poses around a sparse 3D point cloud is most likely illustrating which pipeline output?",
            "correct": "Recovering both scene structure and camera motion from multiple images.",
            "distractors": [
                "Computing dense optical flow between two adjacent frames.",
                "Applying a local smoothing kernel to an image.",
                "Quantizing descriptors into a retrieval histogram.",
            ],
            "explanation": "Camera trajectories plus sparse 3D points are the signature visual for structure from motion.",
            "use_image": True,
        },
    ],
    "Motion and optical flow": [
        {
            "prompt": "Which statement best matches the brightness-constancy assumption used in Lucas-Kanade style optical flow?",
            "correct": "The same moving point keeps approximately the same image brightness between nearby frames.",
            "distractors": [
                "Every moving point keeps exactly the same depth in world coordinates.",
                "Every frame has the same histogram after smoothing.",
                "The camera focal length changes to match the motion.",
            ],
            "explanation": "Brightness constancy is one of the core assumptions used to derive the optical-flow constraint equation.",
        },
        {
            "prompt": "If motion is observed only through a small aperture on a straight edge, which component is usually recoverable?",
            "correct": "Only the motion component perpendicular to the edge.",
            "distractors": [
                "Only the motion component parallel to the edge.",
                "The full 2D motion vector with no ambiguity.",
                "Only the scene depth of the point.",
            ],
            "explanation": "This is the aperture problem: a straight edge gives insufficient information to determine full 2D motion.",
        },
        {
            "prompt": "Why does Lucas-Kanade usually aggregate information over a small neighborhood instead of using a single pixel only?",
            "correct": "Because one pixel gives too little information to solve for the two motion components reliably.",
            "distractors": [
                "Because a single pixel already gives three independent motion equations.",
                "Because neighborhoods remove the need for brightness constancy.",
                "Because neighborhoods guarantee perfect semantic segmentation.",
            ],
            "explanation": "The optical-flow equation alone is underconstrained, so Lucas-Kanade uses nearby pixels and spatial coherence to estimate motion.",
        },
        {
            "prompt": "What is the purpose of the coarse-to-fine pyramid used in iterative optical flow estimation?",
            "correct": "It allows larger motions to be handled by first solving them at coarse resolution and then refining at finer levels.",
            "distractors": [
                "It removes the need to warp images during refinement.",
                "It converts optical flow into stereo disparity.",
                "It guarantees that brightness constancy is always exact.",
            ],
            "explanation": "The lecture shows Gaussian pyramids and iterative refinement so that large displacements become easier to estimate.",
        },
        {
            "prompt": "The arrow field overlaid on successive frames is most likely visualizing which quantity?",
            "correct": "The direction and magnitude of apparent image motion estimated as optical flow.",
            "distractors": [
                "The segmentation class of each object region.",
                "The projection rays of a pinhole camera.",
                "The cluster centers of a visual vocabulary.",
            ],
            "explanation": "Arrow fields are the standard visualization for optical flow.",
            "use_image": True,
        },
    ],
    "LLM prompting and DeepSeek overview": [
        {
            "prompt": "What is the main purpose of prompt design in an LLM workflow?",
            "correct": "To steer the model toward the desired task, format, and reasoning behavior.",
            "distractors": [
                "To estimate camera motion from multiple views.",
                "To replace local descriptors with epipolar geometry.",
                "To compute disparity from a rectified stereo pair.",
            ],
            "explanation": "Prompting controls how the model interprets and responds to the task.",
        },
        {
            "prompt": "What is the difference between zero-shot and few-shot prompting?",
            "correct": "Few-shot prompting provides example demonstrations, while zero-shot prompting does not.",
            "distractors": [
                "Zero-shot prompting uses images only, while few-shot prompting uses text only.",
                "Few-shot prompting works only for segmentation tasks.",
                "Zero-shot prompting requires a calibrated stereo rig.",
            ],
            "explanation": "Few-shot prompting supplies examples to guide the model's behavior.",
        },
        {
            "prompt": "Why can a more explicit prompt improve an LLM answer?",
            "correct": "Because clearer task instructions reduce ambiguity about the expected output.",
            "distractors": [
                "Because explicit prompts guarantee perfect factual accuracy.",
                "Because explicit prompts eliminate the need for any model weights.",
                "Because explicit prompts convert LLMs into optical-flow solvers.",
            ],
            "explanation": "Specific instructions often help the model format and focus its response more appropriately.",
        },
        {
            "prompt": "What is the main idea behind chain-of-thought prompting in this lecture topic?",
            "correct": "Encourage the model to expose intermediate reasoning steps when solving a task.",
            "distractors": [
                "Force the model to output only single-token answers.",
                "Replace all language understanding with geometric warping.",
                "Compute a bag-of-words histogram before every response.",
            ],
            "explanation": "Chain-of-thought prompting is about eliciting a more explicit reasoning process.",
        },
        {
            "prompt": "A slide comparing prompt styles, model capabilities, or reasoning behavior is most likely illustrating which lecture theme?",
            "correct": "How prompting choices and model design influence LLM outputs and reasoning quality.",
            "distractors": [
                "How disparity changes with stereo depth.",
                "How Harris corners respond to intensity variation.",
                "How a pinhole camera projects 3D points onto the image.",
            ],
            "explanation": "Comparison charts or reasoning examples are typical visuals for an LLM prompting lecture.",
            "use_image": True,
        },
    ],
}
