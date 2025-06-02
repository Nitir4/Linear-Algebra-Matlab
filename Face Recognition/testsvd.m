clc;
load svddb.mat;
[filename, pathname] = uigetfile('*.*', 'Select the input image');
filewithpath = strcat(pathname, filename);
img = imread(filewithpath);
imgo = img;
img = rgb2gray(img);
img = imresize(img, [M, N]);
img = double(reshape(img, [1, M*N]));
imgsvd = (img - m) * PsVD; % Project test image onto SVD basis
distarray = zeros(n, 1);

% Compute Euclidean distances
for i = 1:n
    distarray(i) = sum(abs(T(i, :) - imgsvd));
end

[result, indx] = min(distarray);
resultimg = imread(sprintf('%d.jpeg', indx));

% Display results
subplot(121);
imshow(imgo);
title('Input face');
subplot(122);
imshow(resultimg);
title('Recognized face');