function ImageCompressionApp_mech505()

    global origImgPath; % Used to store loaded image's file path for later size stats

    % --- Step 1: Load Image ---
    % Use MATLAB's uigetfile dialog to select an image file interactively
    [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'});
    if isequal(file, 0) % If user cancels
        disp('No file selected.');
        return;
    end
    origImgPath = fullfile(path, file);
    origImage = imread(origImgPath); % Read image from disk (supports multiple formats)

    % --- Step 2: Ask for RGB or Grayscale Mode ---
    % The user can choose to work in "RGB" (keeps color) or "Gray" (single channel)
    mode = input('Enter mode ("RGB" or "Gray"): ', 's');
    if strcmpi(mode, 'Gray')
        origData = rgb2gray(origImage); % MATLAB's built-in function for grayscale conversion
        isRGB = false;
    elseif strcmpi(mode, 'RGB')
        origData = origImage; % Use the original color data
        isRGB = true;
    else
        disp('Invalid mode. Choose either "RGB" or "Gray".');
        return;
    end

    % --- Step 3: Display Original Image ---
    figure; imshow(origImage); title('Original Image');

    % --- Step 4: Show PCA and SVD max k info ---
    % k is the number of principal components (PCA) or singular values (SVD)
    if isRGB
        [m, n, ~] = size(origData);
    else
        [m, n] = size(origData);
    end
    maxPCAk = min(m, n);
    maxSVDk = min(m, n);
    fprintf('PCA max k: %d\n', maxPCAk);
    fprintf('SVD max k: %d\n', maxSVDk);

    % --- Step 5: Ask for Action ---
    % Main menu for user interaction: Compress, Compare or Analyze
    action = input('Choose action: "Compress", "Compare", or "Analyze": ', 's');
    switch lower(action)
        case 'compress'
            compressAndShow(origData, isRGB);
        case 'compare'
            compareMethods(origData, isRGB);
        case 'analyze'
            disp('1: Only PCA');
            disp('2: Only SVD');
            disp('3: Both PCA and SVD');
            algChoice = input('Which algorithm(s) to analyze? (1/2/3): ');
            analyzeMetricsOptimized(origData, isRGB, algChoice);
        otherwise
            disp('Invalid action. Choose "Compress", "Compare", or "Analyze".');
            return;
    end
end

function compressAndShow(data, isRGB)
    % compressAndShow - Compress the image using selected method and display results
    %
    % Inputs:
    %   data  - Image matrix (grayscale or RGB)
    %   isRGB - Boolean, true if data is color image
    
    global origImgPath;

    origInfo = dir(origImgPath);            % Get file info struct from path
    origSizeBytes = origInfo.bytes;         % File size in bytes on disk

    % --- Choose Method (PCA or SVD) ---
    method = input('Enter method ("PCA" or "SVD"): ', 's');
    if ~ismember(upper(method), {'PCA', 'SVD'})
        disp('Invalid method. Choose "PCA" or "SVD".');
        return;
    end

    % --- Choose k (number of components) ---
    k = input('Enter number of components k: ');

    % --- Compression Step ---
    tic; % Start timer
    [compImage, compressedData] = compressImageFull(data, k, method, isRGB);
    compTime = toc; % End timer, store elapsed time

    % Estimate uncompressed and compressed data size (for reporting ratio)
    if isRGB
        [m, n, ~] = size(data);
    else
        [m, n] = size(data);
    end
    origUncompressedSize = m * n * (isRGB * 2 + 1); % 3 for RGB, 1 for Gray (approximate, 1 byte/pixel/channel)
    compressedDataSize = getCompressedDataSize(compressedData, method, m, n, k); % Custom function

    compressionPercent = ((origUncompressedSize - compressedDataSize) / origUncompressedSize) * 100;
    compressionRatio = origUncompressedSize / compressedDataSize;

    % --- Compute Metrics ---
    mseVal = mseMetric(data, compImage);                 % Mean Squared Error
    psnrVal = psnrMetric(data, compImage);               % Peak Signal-to-Noise Ratio
    ssimVal = ssimMetric(data, compImage, isRGB);        % Structural Similarity Index

    % --- Visualization ---
    if strcmpi(method, 'PCA')
        % PCA: Show original, compressed projection, and reconstructed image + metrics
        showPCAVisualization(data, compressedData, compImage, isRGB, mseVal, psnrVal, ssimVal, compressionRatio, compTime);
    else
        % SVD: Show original and reconstructed (no "compressed" visualization)
        figure;
        subplot(1,3,1);
        imshow(uint8(data));
        title('Original Image');
        subplot(1,3,2);
        text(0.1,0.5,'No compressed visualization for SVD','FontSize',12); axis off;
        subplot(1,3,3);
        imshow(compImage);
        title('Reconstructed Image');
        % Add annotation with metrics
        annotation('textbox', [0.33 0.01 0.34 0.13], ...
            'String', sprintf('MSE: %.4f\nPSNR: %.2f dB\nSSIM: %.4f\nCR: %.2f\nTime: %.4fs', mseVal, psnrVal, ssimVal, compressionRatio, compTime), ...
            'FontSize', 10, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    end

    % --- Print metrics/info to command window ---
    fprintf('%s Compression:\n', upper(method));
    fprintf('Time: %.4fs\n', compTime);
    fprintf('MSE: %.4f\n', mseVal);
    fprintf('PSNR: %.2f dB\n', psnrVal);
    fprintf('SSIM: %.4f\n', ssimVal);
    fprintf('Original Size: %s\n', formatSize(origUncompressedSize));
    fprintf('Approx. Compressed Data Size: %s\n', formatSize(compressedDataSize));
    fprintf('Compression Percentage: %.2f%%\n', compressionPercent);
    fprintf('Compression Ratio: %.2f\n', compressionRatio);

    % --- Optionally Save the Reconstructed Image ---
    saveOpt = input('Save reconstructed image? (y/n): ', 's');
    if strcmpi(saveOpt, 'y')
        [saveFile, savePath] = uiputfile({'*.png', 'PNG Image (*.png)'; '*.jpg', 'JPEG Image (*.jpg)'; '*.bmp', 'Bitmap Image (*.bmp)'}, 'Save Image As');
        if isequal(saveFile, 0)
            disp('Save cancelled.');
        else
            saveFullPath = fullfile(savePath, saveFile);
            imwrite(compImage, saveFullPath); % MATLAB's image write function
            disp(['Image saved to ', saveFullPath]);
        end
    end
end

function [compImage, compressedData] = compressImageFull(data, k, method, isRGB)
    % compressImageFull - Compress an image (RGB or grayscale) with PCA or SVD.
    % Handles multi-channel images by processing each channel independently.
    %
    % Outputs:
    %   compImage      - The reconstructed (decompressed) image
    %   compressedData - The data actually used for compression (projection or SVD factors)
    if isRGB
        channels = size(data, 3);
        compChannels = cell(1, channels);
        compressedData = cell(1, channels);
        for c = 1:channels
            channelData = double(data(:,:,c)); % Always use double for math
            if strcmpi(method, 'PCA')
                [compChannels{c}, compressedData{c}] = customPCA(channelData, k);
            else
                [compChannels{c}, compressedData{c}] = customSVD(channelData, k);
            end
        end
        compImage = uint8(cat(3, compChannels{:})); % Reassemble RGB channels
    else
        channelData = double(data);
        if strcmpi(method, 'PCA')
            [compImage, compressedData] = customPCA(channelData, k);
        else
            [compImage, compressedData] = customSVD(channelData, k);
        end
        compImage = uint8(compImage);
    end
end

function [recon, compressedData] = customPCA(channel, k)
    % customPCA - Compress and reconstruct a 2D data channel using PCA.
    % Returns the reconstructed channel and the principal component projection.
    meanCols = mean(channel, 1);                  % Column mean for centering (MATLAB's mean)
    centered = channel - meanCols;                % Center the data (subtract mean)
    covMatrix = cov(centered);                    % Covariance matrix (MATLAB's cov)
    [eigvecs, eigvals] = eig(covMatrix);          % Get eigenvectors/values (MATLAB's eig)
    [eigvals, idx] = sort(diag(eigvals), 'descend'); % Sort eigenvalues descending
    eigvecs = eigvecs(:, idx);                    % Order eigenvectors to match
    V_k = eigvecs(:, 1:min(k, size(eigvecs, 2))); % Take first k eigenvectors
    proj = centered * V_k;                        % Project data
    recon = proj * V_k' + meanCols;               % Reconstruct approximation
    recon = max(min(recon, 255), 0);              % Clip to valid image range
    compressedData = proj;                        % Store only projected data
end

function [recon, compressedData] = customSVD(channel, k)
    % customSVD - Compress and reconstruct a 2D data channel using SVD.
    % Returns the reconstructed channel and the SVD factors.
    % SVD finds U, S, V such that channel = U*S*V'
    [m, n] = size(channel);
    if m >= n
        % Economical SVD for tall or square matrices
        A = channel' * channel;
        [V, D] = eig(A);
        [d, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        d = d(1:min(k, length(d)));
        V_k = V(:, 1:min(k, size(V, 2)));
        singularValues = sqrt(d);

        U_k = zeros(m, length(singularValues));
        for i = 1:length(singularValues)
            U_k(:, i) = (1/singularValues(i)) * channel * V_k(:, i);
        end
        S_k = diag(singularValues);

        recon = U_k * S_k * V_k';
        compressedData = struct('U', U_k, 'S', diag(S_k), 'V', V_k); % Store S as vector for compactness
    else
        % Fat matrix: switch SVD order
        A = channel * channel';
        [U, D] = eig(A);
        [d, idx] = sort(diag(D), 'descend');
        U = U(:, idx);
        d = d(1:min(k, length(d)));
        U_k = U(:, 1:min(k, size(U, 2)));
        singularValues = sqrt(d);

        V_k = zeros(n, length(singularValues));
        for i = 1:length(singularValues)
            V_k(:, i) = (1/singularValues(i)) * channel' * U_k(:, i);
        end
        S_k = diag(singularValues);

        recon = U_k * S_k * V_k';
        compressedData = struct('U', U_k, 'S', diag(S_k), 'V', V_k);
    end
    recon = max(min(recon, 255), 0); % Clip to valid image range
end

function sizeBytes = getCompressedDataSize(compressedData, method, m, n, k)
    % getCompressedDataSize - Estimate compressed data size in bytes for reporting
    % Assumes double (8 bytes per value)
    if iscell(compressedData)
        sizeBytes = 0;
        for i = 1:length(compressedData)
            if strcmpi(method, 'PCA')
                sizeBytes = sizeBytes + m * k * 8;
            elseif strcmpi(method, 'SVD')
                sizeBytes = sizeBytes + (m*k + n*k + k) * 8;
            end
        end
    elseif isstruct(compressedData)
        if strcmpi(method, 'SVD')
            sizeBytes = (m*k + n*k + k) * 8;
        else
            sizeBytes = numel(compressedData) * 8;
        end
    else % grayscale PCA
        sizeBytes = numel(compressedData) * 8;
    end
end

function sizeStr = formatSize(sizeBytes)
    % formatSize - Human-readable string for byte counts
    if sizeBytes < 1024
        sizeStr = sprintf('%d bytes', sizeBytes);
    elseif sizeBytes < 1024^2
        sizeStr = sprintf('%.2f kB', sizeBytes / 1024);
    else
        sizeStr = sprintf('%.2f MB', sizeBytes / 1024^2);
    end
end

function mseVal = mseMetric(orig, recon)
    % mseMetric - Compute mean squared error between two images (arrays)
    orig = double(orig);
    recon = double(recon);
    mseVal = mean((orig(:) - recon(:)).^2);
end

function psnrVal = psnrMetric(orig, recon)
    % psnrMetric - Compute Peak Signal-to-Noise Ratio (dB) between two images
    mseVal = mseMetric(orig, recon);
    if mseVal == 0
        psnrVal = Inf; % Perfect match
    else
        maxPixel = 255;
        psnrVal = 10 * log10(maxPixel^2 / mseVal);
    end
end

function ssimVal = ssimMetric(orig, recon, isRGB)
    % ssimMetric - Compute average SSIM between two images (RGB-aware)
    if isRGB
        ssimVal = 0;
        for c = 1:size(orig, 3)
            ssimVal = ssimVal + ssim(orig(:,:,c), recon(:,:,c));
        end
        ssimVal = ssimVal / size(orig, 3);
    else
        ssimVal = ssim(orig, recon);
    end
    % MATLAB's ssim() computes the Structural Similarity Index
end

function compareMethods(data, isRGB)
    % compareMethods - Compare PCA and SVD side-by-side for user-selected k
    k = input('Enter number of components k: ');

    if isRGB
        [m, n, ~] = size(data);
    else
        [m, n] = size(data);
    end
    origUncompressedSize = m * n * (isRGB * 2 + 1);

    % --- PCA ---
    [pcaRecon, pcaCompressed] = compressImageFull(data, k, 'PCA', isRGB);
    pcaMSE = mseMetric(data, pcaRecon);
    pcaPSNR = psnrMetric(data, pcaRecon);
    pcaSSIM = ssimMetric(data, pcaRecon, isRGB);
    pcaSize = getCompressedDataSize(pcaCompressed, 'PCA', m, n, k);
    pcaRatio = origUncompressedSize / pcaSize;

    % --- SVD ---
    [svdRecon, svdCompressed] = compressImageFull(data, k, 'SVD', isRGB);
    svdMSE = mseMetric(data, svdRecon);
    svdPSNR = psnrMetric(data, svdRecon);
    svdSSIM = ssimMetric(data, svdRecon, isRGB);
    svdSize = getCompressedDataSize(svdCompressed, 'SVD', m, n, k);
    svdRatio = origUncompressedSize / svdSize;

    % --- Display: Original | PCA | SVD ---
    figure;
    subplot(1,3,1);
    imshow(uint8(data));
    title('Original Image');

    subplot(1,3,2);
    imshow(uint8(pcaRecon));
    title({'PCA Reconstructed'; ...
        sprintf('MSE: %.4f', pcaMSE); ...
        sprintf('PSNR: %.2f dB', pcaPSNR); ...
        sprintf('SSIM: %.4f', pcaSSIM); ...
        sprintf('CR: %.2f', pcaRatio)});

    subplot(1,3,3);
    imshow(uint8(svdRecon));
    title({'SVD Reconstructed'; ...
        sprintf('MSE: %.4f', svdMSE); ...
        sprintf('PSNR: %.2f dB', svdPSNR); ...
        sprintf('SSIM: %.4f', svdSSIM); ...
        sprintf('CR: %.2f', svdRatio)});
end

function showPCAVisualization(orig, compressedData, recon, isRGB, mseVal, psnrVal, ssimVal, compressionRatio, compTime)
    % showPCAVisualization - Display original, compressed (projected), and reconstructed images and metrics
    figure;
    if isRGB
        k_proj = size(compressedData{1},2);
        m_proj = size(compressedData{1},1);
        pcaProjRGB = zeros(m_proj, k_proj, 3);
        for c = 1:3
            pc = compressedData{c};
            % Normalize for visualization
            pc = (pc - min(pc(:))) / (max(pc(:)) - min(pc(:)) + eps);
            pcaProjRGB(:,:,c) = pc;
        end
        subplot(1,3,1);
        imshow(uint8(orig));
        title('Original Image');
        subplot(1,3,2);
        imshow(pcaProjRGB);
        title('Compressed Image');
        subplot(1,3,3);
        imshow(recon);
        title('Reconstructed Image');
    else
        pc = compressedData;
        pc = (pc - min(pc(:))) / (max(pc(:)) - min(pc(:)) + eps);
        subplot(1,3,1);
        imshow(uint8(orig));
        title('Original Image');
        subplot(1,3,2);
        imshow(pc);
        title('Compressed Image');
        subplot(1,3,3);
        imshow(recon);
        title('Reconstructed Image');
    end

    % Show metrics annotation
    annotation('textbox', [0.33 0.01 0.34 0.15], ...
        'String', sprintf('MSE: %.4f\nPSNR: %.2f dB\nSSIM: %.4f\nCR: %.2f\nTime: %.4fs', mseVal, psnrVal, ssimVal, compressionRatio, compTime), ...
        'FontSize', 10, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end

function analyzeMetricsOptimized(data, isRGB, algChoice)
    % analyzeMetricsOptimized - Analyze (plot) metrics for PCA and/or SVD as k varies
    %
    % Plots time, MSE, PSNR, SSIM as a function of k for selected method(s).
    if nargin < 3
        algChoice = 3; % default to both
    end
    if isRGB
        [m, n, ~] = size(data);
        c = 3;
    else
        [m, n] = size(data);
        c = 1;
    end
    maxK = min(m, n);
    kValues = unique(round(linspace(1, maxK, min(20,maxK)))); % Spread k values linearly

    % --- Precompute SVD/PCA for all channels (to speed up k sweep) ---
    if algChoice == 1 || algChoice == 3
        % PCA
        MeanCols = cell(1,c); EigVecs = cell(1,c);
        for cc = 1:c
            if c==1
                chan = double(data);
            else
                chan = double(data(:,:,cc));
            end
            meanCols = mean(chan, 1);
            centered = chan - meanCols;
            covMatrix = cov(centered);
            [eigvecs, eigvals] = eig(covMatrix);
            [~, idx] = sort(diag(eigvals), 'descend');
            eigvecs = eigvecs(:, idx);
            MeanCols{cc} = meanCols;
            EigVecs{cc} = eigvecs;
        end
    end
    if algChoice == 2 || algChoice == 3
        % SVD
        Uall = cell(1,c); Sall = cell(1,c); Vall = cell(1,c);
        for cc = 1:c
            if c==1
                chan = double(data);
            else
                chan = double(data(:,:,cc));
            end
            [U, S, V] = svd(chan, 'econ');
            Uall{cc} = U; Sall{cc} = S; Vall{cc} = V;
        end
    end

    % --- Allocate metric arrays ---
    if algChoice == 1 || algChoice == 3
        pcaTimes = zeros(size(kValues));
        pcaMSEs = zeros(size(kValues));
        pcaPSNRs = zeros(size(kValues));
        pcaSSIMs = zeros(size(kValues));
    end
    if algChoice == 2 || algChoice == 3
        svdTimes = zeros(size(kValues));
        svdMSEs = zeros(size(kValues));
        svdPSNRs = zeros(size(kValues));
        svdSSIMs = zeros(size(kValues));
    end

    % --- Loop over k, reconstruct using precomputed SVD/PCA ---
    for i = 1:length(kValues)
        k = kValues(i);

        % PCA
        if algChoice == 1 || algChoice == 3
            tic;
            if c == 1
                V_k = EigVecs{1}(:,1:min(k,size(EigVecs{1},2)));
                proj = (double(data)-MeanCols{1}) * V_k;
                recon = proj * V_k' + MeanCols{1};
                recon = max(min(recon,255),0);
                pcaRecon = uint8(recon);
            else
                pcaRecon = zeros(m,n,3,'uint8');
                for cc = 1:3
                    V_k = EigVecs{cc}(:,1:min(k,size(EigVecs{cc},2)));
                    proj = (double(data(:,:,cc))-MeanCols{cc}) * V_k;
                    recon = proj * V_k' + MeanCols{cc};
                    recon = max(min(recon,255),0);
                    pcaRecon(:,:,cc) = uint8(recon);
                end
            end
            pcaTimes(i) = toc;
            pcaMSEs(i) = mseMetric(data, pcaRecon);
            pcaPSNRs(i) = psnrMetric(data, pcaRecon);
            pcaSSIMs(i) = ssimMetric(data, pcaRecon, isRGB);
        end

        % SVD
        if algChoice == 2 || algChoice == 3
            tic;
            if c == 1
                reconS = Uall{1}(:,1:k) * Sall{1}(1:k,1:k) * Vall{1}(:,1:k)';
                reconS = max(min(reconS,255),0);
                svdRecon = uint8(reconS);
            else
                svdRecon = zeros(m,n,3,'uint8');
                for cc = 1:3
                    reconS = Uall{cc}(:,1:k) * Sall{cc}(1:k,1:k) * Vall{cc}(:,1:k)';
                    reconS = max(min(reconS,255),0);
                    svdRecon(:,:,cc) = uint8(reconS);
                end
            end
            svdTimes(i) = toc;
            svdMSEs(i) = mseMetric(data, svdRecon);
            svdPSNRs(i) = psnrMetric(data, svdRecon);
            svdSSIMs(i) = ssimMetric(data, svdRecon, isRGB);
        end

        drawnow; % Update any open figures
        if mod(i,5)==0 || i==1 || i==length(kValues)
            fprintf('Processed k=%d/%d\n',k,maxK);
        end
    end

    % --- Plot results ---
    if algChoice == 1
        figure;
        subplot(2,2,1);
        plot(kValues, pcaTimes, 'b-', 'LineWidth', 2);
        legend('PCA Time');
        xlabel('k (components)'); ylabel('Time (seconds)'); title('PCA: Compression Time vs k');
        subplot(2,2,2);
        plot(kValues, pcaMSEs, 'b-', 'LineWidth', 2);
        legend('PCA MSE');
        xlabel('k (components)'); ylabel('MSE'); title('PCA: Reconstruction Error (MSE) vs k');
        subplot(2,2,3);
        plot(kValues, pcaPSNRs, 'b-', 'LineWidth', 2);
        legend('PCA PSNR');
        xlabel('k (components)'); ylabel('PSNR (dB)'); title('PCA: PSNR vs k');
        subplot(2,2,4);
        plot(kValues, pcaSSIMs, 'b-', 'LineWidth', 2);
        legend('PCA SSIM');
        xlabel('k (components)'); ylabel('SSIM'); title('PCA: SSIM vs k');
    elseif algChoice == 2
        figure;
        subplot(2,2,1);
        plot(kValues, svdTimes, 'r--', 'LineWidth', 2);
        legend('SVD Time');
        xlabel('k (components)'); ylabel('Time (seconds)'); title('SVD: Compression Time vs k');
        subplot(2,2,2);
        plot(kValues, svdMSEs, 'r--', 'LineWidth', 2);
        legend('SVD MSE');
        xlabel('k (components)'); ylabel('MSE'); title('SVD: Reconstruction Error (MSE) vs k');
        subplot(2,2,3);
        plot(kValues, svdPSNRs, 'r--', 'LineWidth', 2);
        legend('SVD PSNR');
        xlabel('k (components)'); ylabel('PSNR (dB)'); title('SVD: PSNR vs k');
        subplot(2,2,4);
        plot(kValues, svdSSIMs, 'r--', 'LineWidth', 2);
        legend('SVD SSIM');
        xlabel('k (components)'); ylabel('SSIM'); title('SVD: SSIM vs k');
    else % both
        figure;
        subplot(2,2,1);
        plot(kValues, pcaTimes, 'b-', 'LineWidth', 2); hold on;
        plot(kValues, svdTimes, 'r--', 'LineWidth', 2); hold off;
        legend('PCA Time', 'SVD Time');
        xlabel('k (components)'); ylabel('Time (seconds)'); title('Compression Time vs k');

        subplot(2,2,2);
        plot(kValues, pcaMSEs, 'b-', 'LineWidth', 2); hold on;
        plot(kValues, svdMSEs, 'r--', 'LineWidth', 2); hold off;
        legend('PCA MSE', 'SVD MSE');
        xlabel('k (components)'); ylabel('MSE'); title('Reconstruction Error (MSE) vs k');

        subplot(2,2,3);
        plot(kValues, pcaPSNRs, 'b-', 'LineWidth', 2); hold on;
        plot(kValues, svdPSNRs, 'r--', 'LineWidth', 2); hold off;
        legend('PCA PSNR', 'SVD PSNR');
        xlabel('k (components)'); ylabel('PSNR (dB)'); title('PSNR vs k');

        subplot(2,2,4);
        plot(kValues, pcaSSIMs, 'b-', 'LineWidth', 2); hold on;
        plot(kValues, svdSSIMs, 'r--', 'LineWidth', 2); hold off;
        legend('PCA SSIM', 'SVD SSIM');
        xlabel('k (components)'); ylabel('SSIM'); title('SSIM vs k');
    end
end
