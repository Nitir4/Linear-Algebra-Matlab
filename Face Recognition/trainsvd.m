n = input('Enter No. of Images for training: ');
L = input('Enter No. of Dominant Singular Values to keep: ');
if L > n
    error('L cannot exceed the number of training images (%d). Please choose a smaller L.', n);
end
M = 100; N = 90;
X = zeros(n, M*N);
T = zeros(n, L);

% Read and preprocess images
for count = 1:n
    I = imread(sprintf('%d.jpeg', count));
    I = rgb2gray(I);
    I = imresize(I, [M, N]);
    X(count, :) = reshape(I, [1, M*N]);
end

% Mean centering
Xb = X;
m = mean(X);
for i = 1:n
    X(i, :) = X(i, :) - m;
end

% Apply SVD
[U, S, V] = svd(X, 'econ'); % 'econ' for economy-sized SVD
PsVD = V(:, 1:L); % Right singular vectors as basis (equivalent to PCA eigenvectors)

% Project training data onto SVD basis
for i = 1:n
    T(i, :) = (Xb(i, :) - m) * PsVD;
end

% Save variables for testing
save('svddb.mat', 'PsVD', 'T', 'm', 'M', 'N', 'L');