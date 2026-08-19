### 2. HPNN-Guided Adaptive Image Enhancement (AIE)

Underwater images are affected by **light scattering, dimming, color degradation, and loss of visual details** caused by suspended particles and the underwater imaging environment. These effects can make obstacles harder for an object detector to identify.

To address this problem, the proposed system introduces an **HPNN-guided Adaptive Image Enhancement (AIE)** module before YOLOv8 detection.

The key idea is that instead of applying one fixed image-enhancement operation to every frame, the HPNN analyzes each input frame and **predicts enhancement parameters adaptively for that frame**.

#### HPNN

A downsampled version of the input frame is provided to the HPNN. The network predicts **five scalar parameters** that control the subsequent image-enhancement operations:

```text
                    Input Frame
                         │
                  Downsampled Frame
                         │
                         ▼
                  ┌─────────────┐
                  │    HPNN     │
                  │ Hyperparameter
                  │   Network   │
                  └──────┬──────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     Wr, Wg, Wb          ω              λ
   Channel Weights   Contrast       Sharpening
                     Coefficient      Factor
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                 Adaptive Enhancement
                         │
                         ▼
                  Enhanced Frame
                         │
                         ▼
                      YOLOv8
```

The five predicted parameters are:

* **`Wr`** – red-channel weight
* **`Wg`** – green-channel weight
* **`Wb`** – blue-channel weight
* **`ω`** – contrast enhancement coefficient
* **`λ`** – sharpening factor

These parameters are generated dynamically from the input frame rather than being fixed manually.

### Adaptive Enhancement Operations

The HPNN-generated parameters control three sequential enhancement operations.

#### 1. Channel-Wise Color Balancing

The first operation adjusts the contribution of each RGB channel:

```text
Iw = (Wr Ir, Wg Ig, Wb Ib)
```

where:

* `Ir` = red channel
* `Ig` = green channel
* `Ib` = blue channel
* `Wr, Wg, Wb` = channel weights predicted by HPNN

This allows the enhancement process to compensate for color imbalance in the underwater image.

#### 2. Contrast Enhancement

The color-balanced image is then enhanced using:

```text
Ic = ω En(I) + (1 − ω)I
```

where:

* `I` = input image
* `En(I)` = enhanced version of the image
* `ω` = contrast coefficient predicted by HPNN

The parameter `ω` determines how strongly the enhanced image contributes relative to the original image.

#### 3. Detail Sharpening

The final enhancement stage increases image details and edges:

```text
Is = I + λ(I − Gaussian(I))
```

where:

* `I` = image
* `Gaussian(I)` = Gaussian-blurred version of the image
* `λ` = sharpening factor predicted by HPNN

The resulting sharpened image `Is` is passed to the YOLOv8 detector.

### Complete AIE Pipeline

The complete process is therefore:

```text
Raw Underwater Frame
        │
        ▼
      HPNN
        │
        ├───────────────┐
        │               │
        ▼               ▼
Channel Weights    Enhancement
(Wr, Wg, Wb)       Parameters
                    (ω, λ)
        │               │
        ▼               ▼
Color Balancing
        │
        ▼
Contrast Enhancement
        │
        ▼
Detail Sharpening
        │
        ▼
Enhanced Frame
        │
        ▼
      YOLOv8
        │
        ▼
Obstacle Detection
```

This architecture is illustrated as the **HPNN-guided adaptive image enhancement pipeline** in Figure 2 of the paper. The HPNN predicts the channel weights, contrast coefficient, and sharpening factor, after which the three enhancement operations generate the frame used by YOLOv8.

### End-to-End Training

An important aspect of the proposed approach is that the HPNN/AIE module and the YOLOv8 detection backbone are **trained together**.

The training process is:

```text
Input Frame
     │
     ▼
    HPNN
     │
     ▼
Enhancement Parameters
     │
     ▼
Differentiable AIE
     │
     ▼
Enhanced Image
     │
     ▼
   YOLOv8
     │
     ▼
Detection Loss
     │
     └──────────────► Backpropagation
                          │
                    HPNN + YOLOv8
                       Parameters
```

The enhancement operations are differentiable, allowing the detection loss from YOLOv8 to be propagated backward through the AIE module into the HPNN.

Therefore, the HPNN learns which enhancement parameters are useful for improving **object-detection performance**, rather than simply optimizing image appearance.

A separate clear or enhanced-image ground truth is **not required**. The HPNN is trained in tandem with the detector using the detection objective.

### Why HPNN + AIE is Used

The purpose of this module is not simply to make underwater images visually better. Its role is to produce an image representation that makes the obstacles **easier for YOLOv8 to detect**.

The overall relationship is:

```text
Underwater Image
      │
      │ poor visibility
      ▼
     HPNN
      │
      │ predicts frame-specific
      │ enhancement parameters
      ▼
     AIE
      │
      │ color + contrast + details
      ▼
Enhanced Image
      │
      ▼
    YOLOv8
      │
      ▼
Improved Obstacle Perception
```

Thus, the HPNN acts as an adaptive parameter generator, while AIE performs the actual image enhancement. The enhanced output is then used by YOLOv8 for obstacle detection.

This HPNN-AIE module forms the **visual preprocessing stage** of the larger perception–decision–control pipeline.
