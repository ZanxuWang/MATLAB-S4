# S4 (Structured State Space) Model Implementation for POS Tagging (non-selective)

This README explains how my implementation follows the architecture and methodology described in the S4 paper by Albert Gu et al.https://arxiv.org/abs/2111.00396 
The implementation consists of two main components: data pre-processing in Python and the S4 model implementation in MATLAB.

## Results after 50 epochs

Evaluating on training data...
Training Loss: 0.7795, Training Accuracy: 0.7111
Class 1 - Precision: 0.5747, Recall: 0.6072
Class 2 - Precision: 0.6733, Recall: 0.6018
Class 3 - Precision: 0.4749, Recall: 0.1932
Class 4 - Precision: 0.7850, Recall: 0.8361

Evaluating on validation data...
Validation Loss: 0.8016, Validation Accuracy: 0.7043
Class 1 - Precision: 0.5705, Recall: 0.6116
Class 2 - Precision: 0.6715, Recall: 0.5916
Class 3 - Precision: 0.4229, Recall: 0.1624
Class 4 - Precision: 0.7789, Recall: 0.8304

Evaluating on test data...
Test Loss: 0.7757, Test Accuracy: 0.7117
Class 1 - Precision: 0.5816, Recall: 0.6083
Class 2 - Precision: 0.6708, Recall: 0.5950
Class 3 - Precision: 0.4527, Recall: 0.1810
Class 4 - Precision: 0.7844, Recall: 0.8327


## Usage

1. First run data_processing.py
   
   This creates `processed_data.mat` containing tokenized and embedded sequences.
   processed_data.mat is included in the zip file, you can also directly run the S4_SSM.m

2. Train and evaluate model:
   Run S4_SSM.m in MATLAB
   This trains the model and outputs accuracy, precision, and recall metrics for each class.


## Data Preprocessing

The data_processing.py handles the following steps:

1. **Token Processing**:
   - Processes sequences into overlapping windows of length 4
   - Uses `<start>` token for padding sequences shorter than 4 tokens
   - Converts each token to its corresponding 64-dimensional embedding
   - Creates sequences of shape: `[num_samples, sequence_length=4, embedding_dim=64]`

2. **POS Tag Processing**:
   - Maps 46 original POS tags to 4 classes:
     - Class 1 (Noun): Tags 22-28
     - Class 2 (Verb): Tags 37-42
     - Class 3 (Adj/Adv): Tags 16-18, 32-34
     - Class 4 (Others): All remaining tags

3. **Data Shapes**:
   ```
   Train tokens: (203621, 4, 64), Train tags: (203621,)
   Valid tokens: (51362, 4, 64), Valid tags: (51362,)
   Test tokens: (46435, 4, 64), Test tags: (46435,)
   ```
   Training set unique tokens: 21009
   Training set missing embeddings: 0
   Validation set unique tokens: 9002
   Validation set missing embeddings: 2640
   Test set unique tokens: 8548
   Test set missing embeddings: 0

   Missing tokens in the validation set is mapped to <unknown>, we use the same np.zeros(64) for <start> and <unknown>.


## S4 Model Implementation

Our implementation follows key components from the S4 paper:

1. **HiPPO Matrix Construction** (Section 2.2 of paper):
   ```
   n = (0:N-1)';
   A = (-diag(n) - tril(2 * ones(N), -1));
   B = sqrt(2 * n + 1);   % [N x 1]
   C = sqrt(2 * n + 1);   % [N x 1]
   ```
   - Implements equation (2) from paper
   - Creates the structured state matrix A with HiPPO-LegS parameterization
   - B and C vectors defined according to HiPPO theory

2. **Discretization** (Section 2.3):
   ```
   dt = 0.01;  % Step size
   I = eye(N);
   Ad = (I + dt/2 * A) / (I - dt/2 * A);
   Bd = (I - dt/2 * A) \ (dt * B);
   ```
   - Implements the bilinear transform for discretization
   - Converts continuous-time SSM to discrete-time representation

3. **NPLR Parameterization** (Section 3.2):
   ```
   [V, D_eig] = eig(Ad);
   Lambda = diag(D_eig);
   P = V \ Bd;
   Q = (V') \ C;
   ```
   - Implements diagonal plus low-rank decomposition
   - Converts A matrix to Λ - PQ* form for efficient computation

4. **SSM Convolution Kernel** (Sections 2.4 & 3.3):
   My implementation computes the convolution kernel with the following steps:
 
  ```
   % Compute at roots of unity
   omega = exp(-2*pi*1i*(0:L-1)'/L);
   
   % Expand dimensions for vectorized computation
   Lambda_expanded = Lambda * ones(1, L);
   omega_expanded = ones(N, 1) * omega.';
   D_mat = omega_expanded - Lambda_expanded;
   inv_D = 1 ./ D_mat;
   
   % Compute kernel components
   PQ = P .* Q;
   CB = C .* B;
   CQ = C .* Q;
   PB = P .* B;
   
   % Compute intermediate terms for kernel
   s = sum((PQ .* inv_D), 1);
   numerator = sum((CB .* inv_D), 1);
   v = sum((CQ .* inv_D), 1);
   u = sum((PB .* inv_D), 1);
   
   % Final kernel computation
   s_inv = 1 ./ (1 + s);
   K_omega = numerator - (v .* s_inv .* u);
   K = ifft(K_omega);
   ```

   This implements Algorithm 1 from the paper:
   - Uses roots of unity for frequency domain computation
   - Applies Woodbury identity in a vectorized form for efficient vectorized computation
   - Computes inverse FFT to obtain final convolution kernel


## Model Architecture Details

```
Input [batch_size, 4, 64] 
  → S4 Layer (State Size 64)
  → Hidden Layer (Size 128, ReLU)
  → Output Layer (Size 4, Softmax)
  → Cross Entropy Loss
```

The model consists of:
1. **Input Layer**: Takes sequences of shape [batch_size, 4, 64]
2. **S4 Layer**: 
   - State size N = 64 (matching embedding dimension)
   - Sequence length L = 4
   - Uses NPLR parameterization for efficient computation
3. **Classification Head**:
   ```matlab
   hidden_size = 128;
   W1 = rand(hidden_size, D) * 2 * limit - limit;
   W2 = rand(num_classes, hidden_size) * 2 * limit - limit;
   ```
   - Single hidden layer with ReLU activation
   - Output layer with softmax for 4-class classification


## Training Process

The implementation uses:

1. **Optimization**:
   - Standard Gradient Descent with Fixed Learning rate: 1e-4
   - Batch size: 128
   - Number of epochs: 50

2. **Forward Pass**:
   - Computes SSM convolution using NPLR parameterization
   - Applies MLP classification head
   - Uses cross-entropy loss

3. **Backward Pass**:
   - Implements gradient computation through SSM layer
   - Updates NPLR parameters (Lambda, P, Q, B, C)
   - Updates classification layer parameters


This implementation follows the core principles of the S4 paper while adapting it for the specific task of POS tagging with simplified architecture choices appropriate for the sequence length and classification requirements.

