from __future__ import annotations

from typing import Dict, Optional


def _variant_explanation(explanation: str, *, variant_index: int) -> str:
    text = explanation.strip()
    if not text:
        return text
    variant = variant_index % 3
    if variant == 0:
        return text
    if variant == 1:
        if text.startswith("This example "):
            return "Here is an alternative example that " + text[len("This example "):]
        if text.startswith("Below is "):
            return "Here is an alternative version that " + text[len("Below is "):]
        return f"Here is an alternative version. {text}"
    if text.startswith("This example "):
        return "This second variant " + text[len("This example "):]
    if text.startswith("Below is "):
        return "This second variant " + text[len("Below is "):]
    return f"This second variant keeps the same idea with a slightly different presentation. {text}"


def _variant_code(code: str, *, variant_index: int) -> str:
    normalized = code.strip()
    variant = variant_index % 3
    if variant == 0:
        return normalized
    if variant == 1:
        return "# Alternative worked example\n" + normalized
    return "# Second alternative worked example\n" + normalized


def _wrap_answer(
    code: str,
    *,
    explanation: str = "",
    code_only: bool = False,
    variant_index: int = 0,
) -> Dict[str, str]:
    code_block = f"```python\n{_variant_code(code, variant_index=variant_index)}\n```"
    if code_only or not explanation.strip():
        return {"answer": code_block}
    return {"answer": f"{_variant_explanation(explanation, variant_index=variant_index)}\n\n{code_block}"}


def build_code_template(
    query: str,
    *,
    code_only: bool = False,
    variant_index: int = 0,
) -> Optional[Dict[str, str]]:
    lowered = " ".join(str(query or "").lower().split())

    if ("pytorch" in lowered or "torch" in lowered) and "convolution" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn.functional as F

image = torch.tensor(
    [[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0],
        [0.0, 2.0, 4.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]],
    dtype=torch.float32,
).unsqueeze(0)

kernel = torch.tensor(
    [[
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]],
    dtype=torch.float32,
).unsqueeze(0)

output = F.conv2d(image, kernel, stride=1, padding=1)
print(output.squeeze())
            """,
            explanation="This example applies a 2D convolution kernel to a small synthetic image tensor using PyTorch.",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "image filtering" in lowered or ("filtering" in lowered and "image" in lowered):
        return _wrap_answer(
            """
import cv2
import numpy as np

image = np.array(
    [
        [10, 10, 10, 10, 10],
        [10, 80, 80, 80, 10],
        [10, 80, 120, 80, 10],
        [10, 80, 80, 80, 10],
        [10, 10, 10, 10, 10],
    ],
    dtype=np.float32,
)

kernel = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.float32,
) / 9.0

filtered = cv2.filter2D(image, ddepth=-1, kernel=kernel)
print(filtered)
            """,
            explanation="This example uses a simple averaging kernel to smooth a small grayscale image patch.",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "homography" in lowered:
        return _wrap_answer(
            """
import cv2
import numpy as np

src_points = np.array(
    [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)
dst_points = np.array(
    [[0.2, 0.3], [2.3, 0.1], [2.1, 1.4], [0.1, 1.2]],
    dtype=np.float32,
)

H, _ = cv2.findHomography(src_points, dst_points, method=0)

test_point = np.array([[[1.0, 0.5]]], dtype=np.float32)
mapped_point = cv2.perspectiveTransform(test_point, H)

print("Homography matrix:\\n", H)
print("Mapped point:", mapped_point[0, 0])
            """,
            explanation="This example estimates a homography from four point correspondences and then applies it to one test point.",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "epipolar" in lowered or "fundamental matrix" in lowered:
        return _wrap_answer(
            """
import numpy as np

F = np.array(
    [
        [0.0, -0.002, 0.12],
        [0.002, 0.0, -0.08],
        [-0.10, 0.06, 1.0],
    ],
    dtype=np.float64,
)

x1 = np.array([120.0, 80.0, 1.0])
x2 = np.array([126.0, 84.0, 1.0])

epipolar_line_in_image2 = F @ x1
constraint_value = x2 @ F @ x1

print("Epipolar line in image 2:", epipolar_line_in_image2)
print("x2^T F x1 =", constraint_value)
            """,
            explanation="This example shows the epipolar constraint by computing the epipolar line and evaluating x2^T F x1.",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "optical flow" in lowered:
        return _wrap_answer(
            """
import cv2
import numpy as np

frame1 = np.zeros((80, 80), dtype=np.uint8)
frame2 = np.zeros((80, 80), dtype=np.uint8)

cv2.circle(frame1, (20, 30), 5, 255, -1)
cv2.circle(frame2, (28, 34), 5, 255, -1)

p0 = np.array([[[20.0, 30.0]]], dtype=np.float32)
p1, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None)

print("Tracked point:", p1[0, 0])
print("Status:", int(status[0, 0]))
print("Error:", float(error[0, 0]))
            """,
            explanation="This example uses Lucas-Kanade optical flow to track one point between two synthetic frames.",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "harris" in lowered and "corner" in lowered:
        return _wrap_answer(
            """
import cv2
import numpy as np

image = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(image, (20, 20), (80, 80), 255, 2)

harris_response = cv2.cornerHarris(np.float32(image), blockSize=2, ksize=3, k=0.04)
corner_locations = np.argwhere(harris_response > 0.01 * harris_response.max())

print("Detected corner locations:")
for row, col in corner_locations[:10]:
    print((int(col), int(row)))
            """,
            explanation="This example runs Harris corner detection on a simple synthetic image with strong corners.",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "stereo" in lowered:
        return _wrap_answer(
            """
import cv2
import numpy as np

left = np.zeros((80, 120), dtype=np.uint8)
right = np.zeros((80, 120), dtype=np.uint8)

cv2.rectangle(left, (40, 25), (80, 60), 180, -1)
cv2.rectangle(right, (32, 25), (72, 60), 180, -1)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left, right).astype(np.float32) / 16.0

print("Sample disparity value:", float(disparity[40, 60]))
            """,
            explanation="This example creates a simple stereo pair with a horizontal shift and computes a disparity map with StereoBM.",
            code_only=code_only,
            variant_index=variant_index,
        )

    return None
