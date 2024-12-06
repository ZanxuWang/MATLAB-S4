% S4_SSM.m
% MATLAB script implementing the S4 layer with gradient computations,
% and reporting of losses and accuracies during training.
% At the end, it reports the accuracy, precision, and recall for each class on all three datasets.

% Clear workspace and command window
clear; clc;

%% Load the Data
load('processed_data.mat');

% Ensure data is in double precision
train_tokens = double(train_tokens);
train_tags = double(train_tags);
valid_tokens = double(valid_tokens);
valid_tags = double(valid_tags);
test_tokens = double(test_tokens);
test_tags = double(test_tags);

%% Initialize Model Parameters
N = 64;             % State size (embedding size)
L = size(train_tokens, 2);  % Sequence length, 4
D = size(train_tokens, 3);  % Embedding dimension, 64
num_classes = 4;    % Number of POS tags

rng('default');  % For reproducibility

% Construct the HiPPO-LegS matrix
n = (0:N-1)';
A = (-diag(n) - tril(2 * ones(N), -1));
B = sqrt(2 * n + 1);   % [N x 1]
C = sqrt(2 * n + 1);   % [N x 1]

% Discretize A and B
dt = 0.01;  % Smaller step size for stability
I = eye(N);
Ad = (I + dt/2 * A) / (I - dt/2 * A);  % Discretized A
Bd = (I - dt/2 * A) \ (dt * B);        % Discretized B

% NPLR parameterization (A = diag(Lambda) + P*Q^T)
[V, D_eig] = eig(Ad);
Lambda = diag(D_eig);        % Eigenvalues

P = V \ Bd;                  % P = V^{-1} * Bd
Q = (V') \ C;                % Q = (V')^{-1} * C

% Ensure Lambda has negative real parts for stability
Lambda = Lambda - max(real(Lambda)) - 0.5;

% Initialize classification layer parameters
hidden_size = 128;

% Xavier/Glorot initialization for weights
limit = sqrt(6 / (hidden_size + D));
W1 = rand(hidden_size, D) * 2 * limit - limit;
b1 = zeros(hidden_size, 1);

limit = sqrt(6 / (num_classes + hidden_size));
W2 = rand(num_classes, hidden_size) * 2 * limit - limit;
b2 = zeros(num_classes, 1);

%% Training Loop
learning_rate = 1e-4;  % Reduced learning rate
num_epochs = 50;
batch_size = 128;
num_samples = size(train_tokens, 1);
num_batches = ceil(num_samples / batch_size);

train_tags = train_tags(:);

% Define omega once before the training loop
omega = exp(-2*pi*1i*(0:L-1)'/L);  % [L x 1] roots of unity

fprintf('Starting training...\n');
for epoch = 1:num_epochs
    fprintf('\nEpoch %d/%d\n', epoch, num_epochs);
    
    % Shuffle the training data
    idx = randperm(num_samples);
    train_tokens = train_tokens(idx, :, :);
    train_tags = train_tags(idx);
    
    total_loss = 0;
    total_correct = 0;
    for batch = 1:num_batches
        % Get batch data
        batch_start = (batch - 1) * batch_size + 1;
        batch_end = min(batch * batch_size, num_samples);
        actual_batch_size = batch_end - batch_start + 1;
        x_batch = train_tokens(batch_start:batch_end, :, :);  % [batch_size x L x D]
        y_batch = train_tags(batch_start:batch_end);          % [batch_size x 1]
        
        % Reshape x_batch to [batch_size x D x L]
        x_batch = permute(x_batch, [1, 3, 2]);  % [batch_size x D x L]
        
        %% Forward Pass
        % Compute denominator for each omega
        Lambda_expanded = Lambda * ones(1, L);    % [N x L]
        omega_expanded = ones(N, 1) * omega.';    % [N x L]
        D_mat = omega_expanded - Lambda_expanded; % [N x L]
        inv_D = 1 ./ D_mat;                       % [N x L]
        
        % Precompute products
        PQ = P .* Q;       % [N x 1]
        CB = C .* B;       % [N x 1]
        CQ = C .* Q;       % [N x 1]
        PB = P .* B;       % [N x 1]
        
        % Compute s, numerator, v, u
        s = sum((PQ .* inv_D), 1);         % [1 x L]
        numerator = sum((CB .* inv_D), 1); % [1 x L]
        v = sum((CQ .* inv_D), 1);         % [1 x L]
        u = sum((PB .* inv_D), 1);         % [1 x L]
        
        s_inv = 1 ./ (1 + s);              % [1 x L]
        K_omega = numerator - (v .* s_inv .* u);  % [1 x L]
        K = ifft(K_omega);                 % [1 x L]
        
        % Expand K for batch convolution
        K_expanded = reshape(K, [1, 1, L]);     % [1 x 1 x L] complex
        y_conv_batch = sum(x_batch .* real(K_expanded), 3);  % [batch_size x D]
        
        % First fully-connected layer with ReLU activation
        z1 = W1 * y_conv_batch' + b1;           % [hidden_size x batch_size]
        a1 = max(0, z1);                        % ReLU activation
        
        % Output layer
        z2 = W2 * a1 + b2;                      % [num_classes x batch_size]
        
        % Apply softmax to get probabilities
        p_batch = softmax(z2);                  % [num_classes x batch_size]
        
        % Compute loss
        batch_indices = 1:actual_batch_size;
        linear_indices = sub2ind(size(p_batch), y_batch', batch_indices);
        log_likelihood = -log(p_batch(linear_indices) + eps);
        batch_loss = sum(log_likelihood);
        total_loss = total_loss + batch_loss;
        
        % Compute predictions and accuracy
        [~, pred_classes] = max(p_batch, [], 1);  % [1 x batch_size]
        correct_predictions = sum(pred_classes' == y_batch);
        total_correct = total_correct + correct_predictions;
        
        %% Backpropagation
        % Output layer gradients
        dz2 = p_batch;
        dz2(linear_indices) = dz2(linear_indices) - 1;
        dz2 = dz2 / actual_batch_size;
        
        dW2 = dz2 * a1';
        db2 = sum(dz2, 2);
        
        % Hidden layer gradients
        da1 = W2' * dz2;
        dz1 = da1;
        dz1(z1 <= 0) = 0;
        
        dW1 = dz1 * y_conv_batch;
        db1 = sum(dz1, 2);
        
        % Gradient w.r.t y_conv_batch
        dy_conv_batch = W1' * dz1;  % [D x batch_size]
        dy_conv_batch = dy_conv_batch';  % [batch_size x D]
        
        % Compute gradients w.r.t K
        x_batch_reshaped = reshape(x_batch, [actual_batch_size * D, L]);  % [batch_size*D x L]
        dy_conv_batch_reshaped = reshape(dy_conv_batch, [actual_batch_size * D, 1]);  % [batch_size*D x 1]
        
        dK = (dy_conv_batch_reshaped' * x_batch_reshaped)';  % [L x 1]
        dK = dK / actual_batch_size;  % [L x 1]
        
        % Backpropagate through the inverse FFT to compute gradients w.r.t K_omega
        dK_omega = fft(dK);  % [L x 1]
        
        % Compute gradients w.r.t S4 parameters
        % Handle complex conjugates properly
        dK_omega_conj = conj(dK_omega).';  % Transpose to [1 x L]
        
        % Precompute terms
        inv_D_conj = conj(inv_D);        % [N x L]
        
        s_conj = conj(s);                % [1 x L]
        s_inv_conj = 1 ./ (1 + s_conj);  % [1 x L]
        s_inv2_conj = s_inv_conj.^2;     % [1 x L]
        
        v_conj = conj(v);                % [1 x L]
        u_conj = conj(u);                % [1 x L]
        
        % Gradients w.r.t P
        ds_dP = Q .* inv_D;              % [N x L]
        du_dP = B .* inv_D;              % [N x L]
        
        dK_omega_dP = v_conj .* (s_inv2_conj .* u_conj .* ds_dP) - v_conj .* (s_inv_conj .* du_dP);  % [N x L]
        
        dP = sum(dK_omega_dP .* dK_omega_conj, 2);  % [N x 1]
        
        % Gradients w.r.t Q
        ds_dQ = P .* inv_D;              % [N x L]
        dv_dQ = C .* inv_D;              % [N x L]
        
        dK_omega_dQ = - (dv_dQ .* s_inv_conj .* u_conj) + v_conj .* (s_inv2_conj .* u_conj .* ds_dQ);  % [N x L]
        
        dQ = sum(dK_omega_dQ .* dK_omega_conj, 2);  % [N x 1]
        
        % Gradients w.r.t B
        du_dB = P .* inv_D;              % [N x L]
        dnumerator_dB = C .* inv_D;      % [N x L]
        
        dK_omega_dB = dnumerator_dB - v_conj .* s_inv_conj .* du_dB;  % [N x L]
        
        dB = sum(dK_omega_dB .* dK_omega_conj, 2);  % [N x 1]
        
        % Gradients w.r.t C
        dv_dC = Q .* inv_D;              % [N x L]
        dnumerator_dC = B .* inv_D;      % [N x L]
        
        dK_omega_dC = dnumerator_dC - (s_inv_conj .* u_conj .* dv_dC);  % [N x L]
        
        dC = sum(dK_omega_dC .* dK_omega_conj, 2);    % [N x 1]
        
        % Gradients w.r.t Lambda
        % Compute derivatives of inv_D w.r.t Lambda
        dinv_D_dLambda = inv_D.^2;       % [N x L]
        ds_dLambda = - (P .* Q) .* dinv_D_dLambda;     % [N x L]
        du_dLambda = - (P .* B) .* dinv_D_dLambda;     % [N x L]
        dv_dLambda = - (C .* Q) .* dinv_D_dLambda;     % [N x L]
        dnumerator_dLambda = - (C .* B) .* dinv_D_dLambda;  % [N x L]
        
        % Compute dK_omega_dLambda
        term1 = dnumerator_dLambda;
        term2 = dv_dLambda .* s_inv_conj .* u_conj;
        term3 = v_conj .* (-s_inv2_conj .* ds_dLambda .* u_conj);
        term4 = v_conj .* s_inv_conj .* du_dLambda;
        
        dK_omega_dLambda = term1 - term2 - term3 - term4;  % [N x L]
        
        dLambda = sum(dK_omega_dLambda .* dK_omega_conj, 2);  % [N x 1]
        
        %% Update Parameters
        % Classification layer parameters
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        
        % S4 parameters
        P = P - learning_rate * dP;
        Q = Q - learning_rate * dQ;
        B = B - learning_rate * dB;
        C = C - learning_rate * dC;
        Lambda = Lambda - learning_rate * dLambda;
        
        if mod(batch, 100) == 0
            fprintf('Batch %d/%d processed.\n', batch, num_batches);
        end
    end
    % Compute average loss and accuracy for the epoch
    average_loss = total_loss / num_samples;
    train_accuracy = total_correct / num_samples;
    fprintf('Epoch %d/%d, Training Loss: %.4f, Training Accuracy: %.4f\n', epoch, num_epochs, average_loss, train_accuracy);
    
    % Evaluate on validation data
    [valid_loss, valid_accuracy] = evaluate_model(valid_tokens, valid_tags(:), batch_size, W1, b1, W2, b2, P, Q, B, C, Lambda, omega);
    fprintf('Validation Loss: %.4f, Validation Accuracy: %.4f\n', valid_loss, valid_accuracy);
end

%% After Training: Evaluate on All Datasets and Report Metrics
fprintf('\nTraining complete.\n');

% Evaluate on training data
fprintf('\nEvaluating on training data...\n');
[train_loss, train_accuracy, train_precision, train_recall] = evaluate_model_metrics(train_tokens, train_tags(:), batch_size, W1, b1, W2, b2, P, Q, B, C, Lambda, omega, num_classes);
fprintf('Training Loss: %.4f, Training Accuracy: %.4f\n', train_loss, train_accuracy);
for c = 1:num_classes
    fprintf('Class %d - Precision: %.4f, Recall: %.4f\n', c, train_precision(c), train_recall(c));
end

% Evaluate on validation data
fprintf('\nEvaluating on validation data...\n');
[valid_loss, valid_accuracy, valid_precision, valid_recall] = evaluate_model_metrics(valid_tokens, valid_tags(:), batch_size, W1, b1, W2, b2, P, Q, B, C, Lambda, omega, num_classes);
fprintf('Validation Loss: %.4f, Validation Accuracy: %.4f\n', valid_loss, valid_accuracy);
for c = 1:num_classes
    fprintf('Class %d - Precision: %.4f, Recall: %.4f\n', c, valid_precision(c), valid_recall(c));
end

% Evaluate on test data
fprintf('\nEvaluating on test data...\n');
[test_loss, test_accuracy, test_precision, test_recall] = evaluate_model_metrics(test_tokens, test_tags(:), batch_size, W1, b1, W2, b2, P, Q, B, C, Lambda, omega, num_classes);
fprintf('Test Loss: %.4f, Test Accuracy: %.4f\n', test_loss, test_accuracy);
for c = 1:num_classes
    fprintf('Class %d - Precision: %.4f, Recall: %.4f\n', c, test_precision(c), test_recall(c));
end

%% Helper Functions

function s = softmax(z)
    z = z - max(z, [], 1);
    exp_z = exp(z);
    s = exp_z ./ sum(exp_z, 1);
end

function [loss, accuracy] = evaluate_model(tokens, tags, batch_size, W1, b1, W2, b2, P, Q, B, C, Lambda, omega)
    N = length(Lambda);
    L = size(tokens, 2);
    num_samples = size(tokens, 1);
    num_batches = ceil(num_samples / batch_size);
    total_loss = 0;
    total_correct = 0;
    
    for batch = 1:num_batches
        batch_start = (batch - 1) * batch_size + 1;
        batch_end = min(batch * batch_size, num_samples);
        actual_batch_size = batch_end - batch_start + 1;
        
        x_batch = tokens(batch_start:batch_end, :, :);
        y_batch = tags(batch_start:batch_end);
        x_batch = permute(x_batch, [1, 3, 2]);  % [batch_size x D x L]
        
        % Forward pass
        Lambda_expanded = Lambda * ones(1, L);    % [N x L]
        omega_expanded = ones(N, 1) * omega.';    % [N x L]
        D_mat = omega_expanded - Lambda_expanded; % [N x L]
        inv_D = 1 ./ D_mat;                       % [N x L]
        
        % Precompute products
        PQ = P .* Q;       % [N x 1]
        CB = C .* B;       % [N x 1]
        CQ = C .* Q;       % [N x 1]
        PB = P .* B;       % [N x 1]
        
        % Compute s, numerator, v, u
        s = sum((PQ .* inv_D), 1);         % [1 x L]
        numerator = sum((CB .* inv_D), 1); % [1 x L]
        v = sum((CQ .* inv_D), 1);         % [1 x L]
        u = sum((PB .* inv_D), 1);         % [1 x L]
        
        s_inv = 1 ./ (1 + s);              % [1 x L]
        K_omega = numerator - (v .* s_inv .* u);  % [1 x L]
        K = ifft(K_omega);                 % [1 x L]
        
        K_expanded = reshape(K, [1, 1, L]);
        y_conv_batch = sum(x_batch .* real(K_expanded), 3);
        
        z1 = W1 * y_conv_batch' + b1;
        a1 = max(0, z1);
        
        z2 = W2 * a1 + b2;
        p_batch = softmax(z2);
        
        % Compute loss
        batch_indices = 1:actual_batch_size;
        linear_indices = sub2ind(size(p_batch), y_batch', batch_indices);
        log_likelihood = -log(p_batch(linear_indices) + eps);
        batch_loss = sum(log_likelihood);
        total_loss = total_loss + batch_loss;
        
        % Compute predictions and accuracy
        [~, pred_classes] = max(p_batch, [], 1);
        correct_predictions = sum(pred_classes' == y_batch);
        total_correct = total_correct + correct_predictions;
    end
    loss = total_loss / num_samples;
    accuracy = total_correct / num_samples;
end

function [loss, accuracy, precision, recall] = evaluate_model_metrics(tokens, tags, batch_size, W1, b1, W2, b2, P, Q, B, C, Lambda, omega, num_classes)
    N = length(Lambda);
    L = size(tokens, 2);
    num_samples = size(tokens, 1);
    num_batches = ceil(num_samples / batch_size);
    total_loss = 0;
    total_correct = 0;
    predictions = zeros(num_samples, 1);
    true_labels = zeros(num_samples, 1);
    
    for batch = 1:num_batches
        batch_start = (batch - 1) * batch_size + 1;
        batch_end = min(batch * batch_size, num_samples);
        actual_batch_size = batch_end - batch_start + 1;
        
        x_batch = tokens(batch_start:batch_end, :, :);
        y_batch = tags(batch_start:batch_end);
        x_batch = permute(x_batch, [1, 3, 2]);  % [batch_size x D x L]
        
        % Forward pass
        Lambda_expanded = Lambda * ones(1, L);    % [N x L]
        omega_expanded = ones(N, 1) * omega.';    % [N x L]
        D_mat = omega_expanded - Lambda_expanded; % [N x L]
        inv_D = 1 ./ D_mat;                       % [N x L]
        
        % Precompute products
        PQ = P .* Q;       % [N x 1]
        CB = C .* B;       % [N x 1]
        CQ = C .* Q;       % [N x 1]
        PB = P .* B;       % [N x 1]
        
        % Compute s, numerator, v, u
        s = sum((PQ .* inv_D), 1);         % [1 x L]
        numerator = sum((CB .* inv_D), 1); % [1 x L]
        v = sum((CQ .* inv_D), 1);         % [1 x L]
        u = sum((PB .* inv_D), 1);         % [1 x L]
        
        s_inv = 1 ./ (1 + s);              % [1 x L]
        K_omega = numerator - (v .* s_inv .* u);  % [1 x L]
        K = ifft(K_omega);                 % [1 x L]
        
        K_expanded = reshape(K, [1, 1, L]);
        y_conv_batch = sum(x_batch .* real(K_expanded), 3);
        
        z1 = W1 * y_conv_batch' + b1;
        a1 = max(0, z1);
        
        z2 = W2 * a1 + b2;
        p_batch = softmax(z2);
        
        % Compute loss
        batch_indices = 1:actual_batch_size;
        linear_indices = sub2ind(size(p_batch), y_batch', batch_indices);
        log_likelihood = -log(p_batch(linear_indices) + eps);
        batch_loss = sum(log_likelihood);
        total_loss = total_loss + batch_loss;
        
        % Compute predictions and accuracy
        [~, pred_classes] = max(p_batch, [], 1);
        correct_predictions = sum(pred_classes' == y_batch);
        total_correct = total_correct + correct_predictions;
        
        % Store predictions and true labels
        predictions(batch_start:batch_end) = pred_classes';
        true_labels(batch_start:batch_end) = y_batch;
    end
    loss = total_loss / num_samples;
    accuracy = total_correct / num_samples;
    
    % Compute per-class precision and recall
    precision = zeros(num_classes, 1);
    recall = zeros(num_classes, 1);
    for c = 1:num_classes
        TP = sum((predictions == c) & (true_labels == c));
        FP = sum((predictions == c) & (true_labels ~= c));
        FN = sum((predictions ~= c) & (true_labels == c));
        
        precision(c) = TP / (TP + FP + eps);
        recall(c) = TP / (TP + FN + eps);
    end
end
