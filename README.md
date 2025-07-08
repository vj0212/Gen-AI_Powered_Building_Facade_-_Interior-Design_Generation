# Gen-AI_Powered_Building_Facade_-_Interior-Design_Generation

This repository contains three distinct Jupyter Notebook projects (`.ipynb` files) that leverage generative AI for various design-related tasks:

1.  **`GANs_Facade_Generation.ipynb`**: Focuses on generating building facade images using Generative Adversarial Networks (GANs).
2.  **`Interior_Design_Stable_Diffusion.ipynb`**: Implements Stable Diffusion for creating interior design images and integrates Segformers for semantic segmentation.
3.  **`ControlNet_&_Zero-Shot_Learning.ipynb`**: Demonstrates ControlNet for conditional image generation combined with zero-shot learning techniques.

Each project is designed for Google Colab with GPU acceleration and utilizes specific AI models and libraries to achieve its objectives.

---

## 1. GANs Facade Generation

This notebook implements a Generative Adversarial Network (GAN) framework to generate realistic building facade images. It's suitable for architectural visualization and design exploration. 🎨🏗️

### Notebook Metadata

* **Format**: Jupyter Notebook (`nbformat 4, nbformat_minor 0`)
* **Environment**: Google Colab with GPU acceleration (specifically **L4 GPU** and high-memory machine shape `hm`).
* **Kernel**: Python 3
* **Accelerator**: NVIDIA L4 GPU for efficient GAN training and inference.
* **Widgets**: Incorporates Jupyter widgets (`@jupyter-widgets/controls, version 1.5.0`) for interactive UI elements like authentication prompts and progress tracking.

### Purpose

The notebook's primary goal is to generate high-quality **building facade designs** from random noise or conditional inputs, likely inspired by datasets such as CMP Facade. It achieves this by combining a Generator and a Discriminator model.

### Dependencies and Setup

* **Libraries**:
    * `torch`: For GPU-accelerated tensor operations and neural network implementation.
    * `torchvision`: For image preprocessing and dataset handling (e.g., loading facade datasets).
    * `numpy`: For numerical operations.
    * `matplotlib` or `PIL`: For visualizing generated images.
    * `huggingface_hub`: For accessing pre-trained models or datasets from Hugging Face.
* **Hugging Face Integration**: Includes a widget for Hugging Face token authentication, allowing access to models or datasets. It supports login via a password field and optional Git credential storage.
* **GPU Acceleration**: Configured for an **L4 GPU** to handle the significant memory and processing power required for GAN training.
* **Widget Layout**: Uses `VBoxModel` and `LayoutModel` for a centered, user-friendly authentication interface with 50% width responsiveness.

### Key Components

* **Hugging Face Authentication**: Displays an HTML prompt with a Hugging Face logo and instructions to paste a token securely. Includes a password input, a checkbox for storing the token as a Git credential, and a login button.
* **GAN Architecture (Assumed)**:
    * **Generator**: A neural network that maps random noise (or conditional inputs) to facade images, likely using convolutional transpose layers (`nn.ConvTranspose2d` in PyTorch).
    * **Discriminator**: A convolutional neural network that distinguishes real facade images from generated ones, outputting a probability score.
    * **Loss Function**: Uses **adversarial loss** (e.g., Binary Cross-Entropy) to train both models in a minimax game.
* **Dataset Handling**: Likely loads a facade dataset (e.g., CMP Facade Database) and preprocesses images (resizing, normalization) to a standard size (e.g., 256x256).
* **Image Output**: Includes base64-encoded PNG images, indicating successful generation, likely displayed inline or saved to disk.

### Execution Flow

1.  **Initialization**: Installs dependencies and configures the L4 GPU environment. Authenticates with Hugging Face.
2.  **Model Setup**: Defines Generator and Discriminator architectures in PyTorch, and loads pre-trained weights or initializes models. Downloads necessary files from Hugging Face.
3.  **Training**: Iteratively trains the GAN by alternating Discriminator and Generator updates, using optimizers (e.g., `torch.optim.Adam`) and adversarial loss. Monitors progress with loss metrics and periodic image generation.
4.  **Inference**: Generates facade images by passing random noise through the trained Generator.
5.  **Completion**: Displays or saves generated facade images as PNGs.

### Technical Details

* **GAN Framework**: Implements a standard GAN or a variant (e.g., DCGAN, Conditional GAN).
    * **Generator**: Upsamples noise vectors to produce facade images (e.g., 256x256) using convolutional transpose layers.
    * **Discriminator**: Downsamples images to a single scalar for real vs. fake prediction.
* **Hugging Face Integration**: Uses `huggingface_hub` for secure access to models or datasets via token authentication.
* **Base64 Image**: Embedded PNG images are likely rendered using `matplotlib.pyplot` or `IPython.display.Image`.
* **Widgets**: The authentication interface uses `@jupyter-widgets/controls` for interactivity with a clean, centered layout.

### Usage Instructions

1.  **Environment Setup**: Run in Google Colab with an **L4 GPU** enabled.
2.  **Dependencies**: Install required libraries via `!pip install torch torchvision huggingface_hub`.
3.  **Authentication**: Obtain a Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), paste it into the password field, and click "Login."
4.  **Input**: Provide random noise or conditional inputs (if using Conditional GAN) to generate images.
5.  **Output**: View generated facade images inline or save them to files.

### Potential Improvements

* Add error handling for authentication failures or GPU unavailability.
* Include options to customize GAN hyperparameters (e.g., learning rate, batch size, epochs).
* Support conditional inputs (e.g., architectural styles or segmentation masks) for controlled generation.
* Implement checkpointing and provide mechanisms to export generated images.

### Limitations

* Requires an **L4 GPU** for efficient training.
* Assumes familiarity with Jupyter, Colab, and PyTorch.
* Base64-encoded images increase notebook size, impacting portability.
* GAN training can be unstable, requiring careful hyperparameter tuning.
* Hugging Face token authentication requires an active account and internet access.

### Output Example

Generates realistic building facade images, such as modern glass-paneled designs or classical brick facades, displayed as 256x256 PNGs.

---

## 2. Interior Design Stable Diffusion and Segformer

This notebook implements Stable Diffusion for generating interior design images and integrates Segformers for semantic segmentation, combining generative AI with image analysis. 🛋️🖼️

### Notebook Metadata

* **Format**: Jupyter Notebook (`nbformat 4, nbformat_minor 0`)
* **Environment**: Google Colab with GPU acceleration (specifically **T4 GPU**).
* **Kernel**: Python 3
* **Accelerator**: T4 GPU for efficient computation of Stable Diffusion and Segformer models.
* **Widgets**: Utilizes Jupyter widgets (`@jupyter-widgets/controls, version 1.5.0`) for progress bars and interactive UI elements during model loading and processing.

### Purpose

The notebook generates high-quality **interior design images** using Stable Diffusion (a text-to-image diffusion model) and applies Segformers for **semantic segmentation**. This allows for both the creation and analysis of interior design layouts (e.g., identifying furniture, walls, decor).

### Dependencies and Setup

* **Libraries**:
    * `diffusers`: For Stable Diffusion model loading and inference.
    * `transformers`: For Segformer model implementation and semantic segmentation.
    * `torch`: For GPU-accelerated tensor operations.
    * `PIL`: For image processing and handling.
    * `numpy`, `matplotlib` (potential for data manipulation and visualization).
* **GPU Acceleration**: Configured for a **T4 GPU** to handle the computational demands of both Stable Diffusion and Segformer models.
* **Jupyter Widgets**: Progress bars (e.g., `HBoxModel`, `FloatProgressModel`) provide status updates for model downloads, inference, or segmentation tasks.

### Key Components

* **Stable Diffusion Model**: Loads a pre-trained Stable Diffusion model via the `diffusers` library. It generates interior design images from text prompts (e.g., "cozy Scandinavian bedroom").
* **Segformer Model**: Utilizes a Segformer model (from the `transformers` library) for semantic segmentation. It segments images into meaningful regions (e.g., furniture, walls, floors) to support tasks like layout analysis.
* **Progress Tracking**: Employs widgets to monitor tasks such as model downloads, image generation, and segmentation.
* **Image Output**: Generates and displays interior design images as base64-encoded PNGs, embedded in the notebook. Segformer outputs may include segmented masks, visualized as overlays or separate images.

### Execution Flow

1.  **Initialization**: Installs dependencies and configures the GPU environment.
2.  **Model Setup**: Loads the Stable Diffusion pipeline for image generation and the Segformer model for semantic segmentation (potentially with pre-trained weights like `nvidia/segformer-b0-finetuned-ade-512-512`).
3.  **Image Generation**: Takes text prompts to generate interior design images via Stable Diffusion.
4.  **Segmentation**: Applies Segformer to segment the generated images, identifying key elements.
5.  **Output Display**: Renders generated images and segmentation masks in the notebook, with progress tracked by widgets.
6.  **Completion**: Outputs a confirmation message upon successful execution.

### Technical Details

* **Stable Diffusion**: A latent diffusion model that generates high-resolution images from text prompts by denoising latent representations, using `StableDiffusionPipeline`.
* **Segformer**: A transformer-based model for semantic segmentation, efficient for pixel-level classification, likely fine-tuned for interior design elements (e.g., on the ADE20K dataset).
* **Integration**: The notebook likely combines Stable Diffusion outputs with Segformer to analyze or refine generated designs.
* **Base64 Image**: Embedded PNG images are viewable via `IPython.display` or similar.
* **Widgets**: Real-time feedback is provided via `@jupyter-widgets/controls` for various tasks.

### Usage Instructions

1.  **Environment Setup**: Run in Google Colab with a **T4 GPU** enabled.
2.  **Dependencies**: Install `diffusers`, `transformers`, `torch`, and `PIL` via `!pip install` commands.
3.  **Input**: Provide text prompts for Stable Diffusion (e.g., "modern minimalist living room"). Optionally, input images for Segformer to perform segmentation.
4.  **Output**: View generated images and segmentation masks inline or save them to files.
5.  **Example Prompt**: "Generate a cozy living room and segment furniture and walls."

### Potential Improvements

* Add error handling for GPU unavailability or model download failures.
* Include options to fine-tune Segformer for specific interior design categories.
* Allow customization of Stable Diffusion parameters (e.g., inference steps, guidance scale) and Segformer thresholds.
* Implement saving mechanisms for generated images and segmentation masks to cloud storage.

### Limitations

* Requires a GPU for efficient execution.
* Assumes familiarity with Jupyter/Colab and Python-based AI libraries.
* Base64-encoded images may increase notebook size, impacting portability.
* Segformer performance depends on the quality of pre-trained weights and dataset relevance.

### Output Example

* **Stable Diffusion**: Generates images such as a modern kitchen or bedroom based on text prompts.
* **Segformer**: Produces segmentation masks highlighting furniture, walls, or decor, potentially overlaid on generated images.
* **Visualization**: Inline display of images and masks via base64-encoded PNGs.

---

## 3. ControlNet & Zero-Shot Learning

This notebook implements ControlNet for conditional image generation and integrates zero-shot learning techniques, offering precise control over generated images. ⚙️🧠

### Notebook Metadata

* **Format**: Jupyter Notebook (`nbformat 4, nbformat_minor 0`)
* **Environment**: Google Colab with GPU acceleration (specifically **L4 GPU**).
* **Kernel**: Python 3
* **Accelerator**: NVIDIA L4 GPU to handle the computational demands of ControlNet and zero-shot learning models.
* **Widgets**: Incorporates Jupyter widgets (`@jupyter-widgets/controls, version 1.5.0`) for interactive UI elements, such as authentication prompts for Hugging Face integration.

### Purpose

The notebook implements **ControlNet**, a neural network architecture that enhances diffusion models with **conditional control** (e.g., edge maps, pose estimation) for precise image generation. It also integrates **zero-shot learning** to enable the model to generalize to unseen tasks or classes without additional training, likely for tasks like image generation with novel prompts.

### Dependencies and Setup

* **Libraries**:
    * `torch`: For GPU-accelerated tensor operations and neural network implementation.
    * `torchvision`: For image preprocessing and dataset handling.
    * `diffusers`: For diffusion model pipelines, including ControlNet integration.
    * `transformers`: For zero-shot learning components, such as CLIP for text-to-image alignment.
    * `huggingface_hub`: For accessing pre-trained models or datasets from Hugging Face.
    * `numpy`, `PIL`, `matplotlib`: For numerical operations, image handling, and visualization.
* **Hugging Face Integration**: Includes a widget for Hugging Face token authentication, supporting login via a password field and optional Git credential storage. It's likely used to download pre-trained ControlNet models, diffusion pipelines, or CLIP models.
* **GPU Acceleration**: Configured for an **L4 GPU** to support the computationally intensive training and inference of diffusion-based models.
* **Widget Layout**: Uses `VBoxModel` with a centered layout (`align_items: "center", width: "50%"`) for a user-friendly authentication interface.

### Key Components

* **Hugging Face Authentication**: Displays an HTML prompt with a Hugging Face logo and instructions to paste a token securely. Includes a password input, a checkbox for storing the token as a Git credential, and a login button.
* **ControlNet Architecture (Assumed)**: Builds on a **Stable Diffusion model**, augmented with ControlNet to condition image generation on additional inputs (e.g., Canny edge maps, depth maps, or pose keypoints). It consists of a neural network that adds control signals to the diffusion process, enabling precise manipulation of generated images, likely using a pre-trained Stable Diffusion model with ControlNet weights from Hugging Face.
* **Zero-Shot Learning (Assumed)**: Integrates a model like **CLIP** (Contrastive Language-Image Pretraining) to align text prompts with generated images, enabling zero-shot generation for unseen prompts. This allows the model to generate images for novel tasks or categories without task-specific fine-tuning, likely by using CLIP's text encoder to guide the diffusion process.
* **Dataset Handling**: Loads datasets (e.g., COCO, custom image datasets) for training or evaluation, with control inputs like edge maps or segmentation masks. Preprocesses images (e.g., resizing to 512x512, normalization) and control conditions.
* **Image Output**: Includes a base64-encoded PNG image, indicating successful generation of a conditioned image, likely displayed inline or saved to disk.

### Execution Flow

1.  **Initialization**: Installs dependencies and configures the L4 GPU environment. Authenticates with Hugging Face.
2.  **Model Setup**: Loads a pre-trained Stable Diffusion model with ControlNet using the `diffusers` library. Initializes CLIP or a similar model for zero-shot learning capabilities. Downloads necessary model weights or datasets from Hugging Face.
3.  **Processing**: Prepares control inputs (e.g., edge maps, pose estimations) and text prompts for zero-shot generation. Runs the diffusion pipeline with ControlNet to generate images conditioned on both control inputs and text prompts.
4.  **Output**: Generates high-quality images based on the provided conditions and prompts. Displays results inline as PNG images or saves them to files.
5.  **Completion**: Outputs generated images, potentially with a confirmation message or evaluation metrics.

### Technical Details

* **ControlNet**: Enhances Stable Diffusion by adding a control branch that processes auxiliary inputs (e.g., edge maps, depth maps). It trains a separate network to condition the diffusion process, preserving the original diffusion model's weights, and uses convolutional layers to integrate control inputs into the U-Net architecture.
* **Zero-Shot Learning**: Leverages **CLIP's text-image alignment** to generate images from text prompts without task-specific training. It encodes text prompts into embeddings using CLIP's text encoder, guiding the diffusion process for flexible generation.
* **Diffusion Process**: Employs a denoising process over multiple timesteps to generate images from noise, conditioned on control inputs and text prompts, using a scheduler (e.g., DDIM, PNDM) for efficiency and quality.
* **Base64 Image**: The embedded PNG suggests successful image generation, likely visualized using `matplotlib.pyplot` or `IPython.display.Image`.
* **Widgets**: The authentication interface uses `@jupyter-widgets/controls` for interactivity, with a clean, centered layout.

### Usage Instructions

1.  **Environment Setup**: Run in Google Colab with an **L4 GPU** enabled.
2.  **Dependencies**: Install required libraries via `!pip install diffusers transformers huggingface_hub`.
3.  **Authentication**: Obtain a Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), paste it into the password field, and click "Login."
4.  **Input**: Provide control inputs (e.g., Canny edge maps, pose estimations) and text prompts for image generation.
    * **Example**: Generate an image of a "modern building" conditioned on an edge map.
5.  **Output**: View generated images inline or save them to files.
    * **Example**: Generate a realistic image of a person in a specific pose or a scene based on a novel text prompt without prior training.

### Potential Improvements

* Add support for multiple control conditions (e.g., combining edge maps and depth maps).
* Implement fine-tuning options for specific tasks to enhance performance beyond zero-shot capabilities.
* Include error handling for invalid control inputs or failed authentication.
* Optimize memory usage for large-scale generation on limited GPU resources.
* Provide a mechanism to save generated images to cloud storage or local files.

### Limitations

* Requires an **L4 GPU** for efficient processing.
* Zero-shot performance depends on the quality of pre-trained models like CLIP, which may struggle with highly specific or niche prompts.
* ControlNet requires high-quality control inputs (e.g., accurate edge maps) for optimal results.
* Base64-encoded images increase notebook size, impacting portability.
* Hugging Face token authentication requires an active account and internet access.

### Output Example

Generates high-quality images conditioned on inputs like edge maps or text prompts. Examples include a 512x512 image of a "futuristic cityscape" conditioned on a sketch, or a person in a specific pose based on a text prompt. Outputs are displayed as PNG images inline.
