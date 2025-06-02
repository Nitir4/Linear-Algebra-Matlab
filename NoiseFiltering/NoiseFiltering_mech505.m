function NoiseFilteringApp2()
    % PCA/SVD metrics vs k using customPCA and customSVD for both text and images
    % Features:
    % - Greyscale/RGB modes for both text and image
    % - User can input k or do full k sweep (metrics-vs-k)
    % - MSE/SSIM/PSNR (images), MSE only (text)
    % - No image display for textual mode, only metrics/plots
    % - No resizing unless user confirms (for large images)
    % - Progress feedback
    % - Computation time vs k
    % - "Optimum k" (first k beating noisy reference) detection and marking
    % - Efficient: SVD/PCA computed once per channel

    clc;
    disp('PCA/SVD metrics-vs-k using customPCA and customSVD (optimized, all features)');

    datatype = menu('Data Type','Textual Data','Image Data');
    use_mock = questdlg('Use mock data?','Data Source','Yes','No','Yes');
    use_mock = strcmp(use_mock,'Yes');

    % --- Textual Data ---
    if datatype==1
        mock_mode = menu('Mock textual data mode?','Greyscale','RGB');
        if use_mock
            if mock_mode == 1 % Greyscale
                X_true = uint8(make_low_rank_matrix_matlab(100, 50, 10, 0.1) * 10 + 128);
                is_rgb = false;
            else % RGB
                X_true = zeros(100,50,3,'uint8');
                for cc = 1:3
                    X_true(:,:,cc) = uint8(make_low_rank_matrix_matlab(100, 50, 10, 0.1) * 10 + 128);
                end
                is_rgb = true;
            end
        else
            [file, path] = uigetfile('*.csv');
            if isequal(file,0)
                disp('No file selected. Exiting.');
                return;
            end
            X_true = uint8(readmatrix(fullfile(path, file)));
            is_rgb = false; % CSV is 2D
        end
        sz = size(X_true);
        X_noisy = X_true + uint8(10*randn(sz));
        mask = rand(sz) < 0.01;
        tmp = X_noisy(mask) + uint8(50.*(2*(rand(nnz(mask),1)>0.5)-1));
        tmp(tmp > 255) = 255; tmp(tmp < 0) = 0;
        X_noisy(mask) = tmp;
        [m, n, ~] = size(X_true);
        kmax = min(m, n);

        prompt = sprintf('Enter k (1-%d) (leave empty for metrics-vs-k/optimum analysis): ', kmax);
        user_k = input(prompt);

        analyzeMetricsHybrid(X_noisy, X_true, kmax, is_rgb, user_k, 'text');
        return
    end

    % --- Image Data ---
    is_rgb = true; % default
    mode = menu('Image Mode?','RGB','Greyscale');
    is_rgb = (mode==1);

    if use_mock
        img = imread('peppers.png'); % NO RESIZE unless too big
    else
        [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif','Image files'});
        if isequal(file,0)
            disp('No file selected. Exiting.');
            return;
        end
        img = imread(fullfile(path,file)); % NO RESIZE unless too big
    end

    % Greyscale/RGB selection
    if ~is_rgb && ndims(img)==3
        img = rgb2gray(img);
    elseif is_rgb && ndims(img)==2
        img = repmat(img,1,1,3);
    end

    % Resize if large
    [m, n, ~] = size(img);
    kmax = min(m,n);
    max_dim = 128; % You can tighten this for speed
    if max(m,n) > max_dim
        resp = questdlg(sprintf('Image is large (%dx%d). Resize to %dx%d for faster computation?',m,n,max_dim,max_dim),'Resize?','Yes','No','Yes');
        if strcmp(resp,'Yes')
            img = imresize(img, [max_dim max_dim]);
            [m, n, ~] = size(img); kmax = min(m,n);
        end
    end

    img_orig = img;
    sz = size(img);
    img_noisy = img + uint8(randn(sz)*10);
    mask = rand(sz) < 0.01;
    tmp = img_noisy(mask) + uint8(50.*(2*(rand(nnz(mask),1)>0.5)-1));
    tmp(tmp > 255) = 255; tmp(tmp < 0) = 0;
    img_noisy(mask) = tmp;

    prompt = sprintf('Enter k (1-%d) (leave empty for metrics-vs-k/optimum analysis): ', kmax);
    user_k = input(prompt);

    analyzeMetricsHybrid(img_noisy, img_orig, kmax, is_rgb, user_k, 'image');
end

function X = make_low_rank_matrix_matlab(n_samples, n_features, effective_rank, tail_strength)
    A = randn(n_samples, effective_rank);
    B = randn(effective_rank, n_features);
    X = A * B;
    X = X + tail_strength*randn(n_samples, n_features);
end

function analyzeMetricsHybrid(noisy, orig, kmax, isRGB, user_k, datatype)
    if nargin < 4, isRGB = false; end
    if nargin < 5, user_k = []; end
    if nargin < 6, datatype = 'image'; end
    if ndims(noisy)==3
        c = size(noisy,3);
    else
        c = 1;
    end
    m = size(noisy,1); n = size(noisy,2);

    % Calculate reference errors (due to noise)
    ref_mse = mseMetric(orig, noisy);

    if strcmp(datatype,'image')
        ref_psnr = psnrMetric(orig, noisy);
        ref_ssim = ssimMetric(orig, noisy, isRGB);
    end

    if isempty(user_k)
        kValues = 1:kmax;
        mse_pca = zeros(size(kValues));
        mse_svd = zeros(size(kValues));
        t_pca = zeros(size(kValues));
        t_svd = zeros(size(kValues));
        if strcmp(datatype,'image')
            psnr_pca = zeros(size(kValues));
            psnr_svd = zeros(size(kValues));
            ssim_pca = zeros(size(kValues));
            ssim_svd = zeros(size(kValues));
        end

        % Compute full decomposition ONCE for each channel
        % --- SVD ---
        Uall = cell(1,c); Sall = cell(1,c); Vall = cell(1,c);
        for cc = 1:c
            if c==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,cc));
            end
            [U,S,V] = customSVD_full(chan);
            Uall{cc} = U; Sall{cc} = S; Vall{cc} = V;
        end

        % --- PCA ---
        MeanCols = cell(1,c); EigVecs = cell(1,c); EigVals = cell(1,c);
        for cc = 1:c
            if c==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,cc));
            end
            meanCols = mean(chan, 1);
            centered = chan - meanCols;
            covMatrix = cov(centered);
            [eigvecs, eigvals] = eig(covMatrix);
            [eigvals, idx] = sort(diag(eigvals), 'descend');
            eigvecs = eigvecs(:, idx);
            MeanCols{cc} = meanCols;
            EigVecs{cc} = eigvecs;
            EigVals{cc} = eigvals;
        end

        % --- Sweep k ---
        for i = 1:length(kValues)
            k = kValues(i);

            % SVD reconstruct
            tic;
            if c == 1
                recon = Uall{1}(:,1:k) * Sall{1}(1:k,1:k) * Vall{1}(:,1:k)';
                recon = max(min(recon,255),0);
                svdRecon = uint8(recon);
            else
                svdRecon = zeros(size(noisy),'double');
                for cc = 1:c
                    recon = Uall{cc}(:,1:k) * Sall{cc}(1:k,1:k) * Vall{cc}(:,1:k)';
                    recon = max(min(recon,255),0);
                    svdRecon(:,:,cc) = recon;
                end
                svdRecon = uint8(svdRecon);
            end
            t_svd(i) = toc;

            % PCA reconstruct
            tic;
            if c == 1
                V_k = EigVecs{1}(:,1:min(k,size(EigVecs{1},2)));
                proj = (double(noisy)-MeanCols{1}) * V_k;
                recon = proj * V_k' + MeanCols{1};
                recon = max(min(recon,255),0);
                pcaRecon = uint8(recon);
            else
                pcaRecon = zeros(size(noisy),'double');
                for cc = 1:c
                    V_k = EigVecs{cc}(:,1:min(k,size(EigVecs{cc},2)));
                    proj = (double(noisy(:,:,cc))-MeanCols{cc}) * V_k;
                    recon = proj * V_k' + MeanCols{cc};
                    recon = max(min(recon,255),0);
                    pcaRecon(:,:,cc) = recon;
                end
                pcaRecon = uint8(pcaRecon);
            end
            t_pca(i) = toc;

            mse_pca(i) = mseMetric(orig, pcaRecon);
            mse_svd(i) = mseMetric(orig, svdRecon);

            if strcmp(datatype,'image')
                psnr_pca(i) = psnrMetric(orig, pcaRecon);
                psnr_svd(i) = psnrMetric(orig, svdRecon);
                ssim_pca(i) = ssimMetric(orig, pcaRecon, isRGB);
                ssim_svd(i) = ssimMetric(orig, svdRecon, isRGB);
            end

            if mod(i,10)==0 || i==1 || i==length(kValues)
                fprintf('Processed k=%d/%d\n',k,kmax);
            end
        end

        % Find optimum k (first k beating noisy reference)
        opt_pca = find(mse_pca < ref_mse, 1, 'first');
        opt_svd = find(mse_svd < ref_mse, 1, 'first');
        if strcmp(datatype,'image')
            opt_pca_psnr = find(psnr_pca > ref_psnr, 1, 'first');
            opt_svd_psnr = find(psnr_svd > ref_psnr, 1, 'first');
            opt_pca_ssim = find(ssim_pca > ref_ssim, 1, 'first');
            opt_svd_ssim = find(ssim_svd > ref_ssim, 1, 'first');
        end

        % ---- Plot MSE ----
        figure;
        set(gcf,'Position',[200 200 700 550]);
        plot(kValues, mse_pca, 'b-', 'LineWidth',2); hold on
        plot(kValues, mse_svd, 'r-', 'LineWidth',2);
        yline(ref_mse, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Noisy Reference');
        if ~isempty(opt_pca), plot(kValues(opt_pca), mse_pca(opt_pca), 'bo', 'MarkerSize',10, 'LineWidth',2, 'DisplayName', 'Optimum PCA'); end
        if ~isempty(opt_svd), plot(kValues(opt_svd), mse_svd(opt_svd), 'ro', 'MarkerSize',10, 'LineWidth',2, 'DisplayName', 'Optimum SVD'); end
        hold off
        legend('PCA','SVD','Noisy','Optimum PCA','Optimum SVD','Location','northeast');
        xlabel('k (components)');
        ylabel('MSE (uint8)');
        title('MSE (uint8) vs k');
        grid on;

        % ---- Plot computation time ----
        figure;
        set(gcf,'Position',[950 200 700 550]);
        plot(kValues, t_pca, 'b-', 'LineWidth',2); hold on
        plot(kValues, t_svd, 'r-', 'LineWidth',2);
        hold off
        legend('PCA','SVD','Location','northwest');
        xlabel('k (components)');
        ylabel('Computation Time (seconds)');
        title('Computation Time vs k');
        grid on;

        % ---- Image: plot PSNR/SSIM ----
        if strcmp(datatype,'image')
            % PSNR
            figure;
            plot(kValues, psnr_pca, 'b-', 'LineWidth',2); hold on
            plot(kValues, psnr_svd, 'r-', 'LineWidth',2);
            yline(ref_psnr, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Noisy Reference');
            if ~isempty(opt_pca_psnr), plot(kValues(opt_pca_psnr), psnr_pca(opt_pca_psnr), 'bo', 'MarkerSize',10, 'LineWidth',2); end
            if ~isempty(opt_svd_psnr), plot(kValues(opt_svd_psnr), psnr_svd(opt_svd_psnr), 'ro', 'MarkerSize',10, 'LineWidth',2); end
            hold off
            legend('PCA','SVD','Noisy','Optimum PCA','Optimum SVD','Location','southeast');
            xlabel('k (components)'); ylabel('PSNR (dB)');
            title('PSNR vs k');
            grid on;
            % SSIM
            figure;
            plot(kValues, ssim_pca, 'b-', 'LineWidth',2); hold on
            plot(kValues, ssim_svd, 'r-', 'LineWidth',2);
            yline(ref_ssim, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Noisy Reference');
            if ~isempty(opt_pca_ssim), plot(kValues(opt_pca_ssim), ssim_pca(opt_pca_ssim), 'bo', 'MarkerSize',10, 'LineWidth',2); end
            if ~isempty(opt_svd_ssim), plot(kValues(opt_svd_ssim), ssim_svd(opt_svd_ssim), 'ro', 'MarkerSize',10, 'LineWidth',2); end
            hold off
            legend('PCA','SVD','Noisy','Optimum PCA','Optimum SVD','Location','southeast');
            xlabel('k (components)'); ylabel('SSIM');
            title('SSIM vs k');
            grid on;
        end

        % ---- Print optimum k ----
        fprintf('\n-------- Optimum k (first k that beats noisy reference) --------\n');
        fprintf('Metric      |  PCA   |  SVD\n');
        fprintf('MSE         |  %4d   |  %4d\n', opt_pca, opt_svd);
        if strcmp(datatype,'image')
            fprintf('PSNR        |  %4d   |  %4d\n', opt_pca_psnr, opt_svd_psnr);
            fprintf('SSIM        |  %4d   |  %4d\n', opt_pca_ssim, opt_svd_ssim);
        end
        fprintf('---------------------------------------------------------------\n');

    else
        k = user_k;
        if k < 1 || k > kmax
            error('k must be between 1 and %d', kmax);
        end

        % --- SVD ---
        svdRecon = zeros(size(noisy));
        for cc = 1:c
            if c==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,cc));
            end
            [U,S,V] = customSVD_full(chan);
            recon = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
            recon = max(min(recon,255),0);
            if c==1
                svdRecon = recon;
            else
                svdRecon(:,:,cc) = recon;
            end
        end
        svdRecon = uint8(svdRecon);

        % --- PCA ---
        pcaRecon = zeros(size(noisy));
        for cc = 1:c
            if c==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,cc));
            end
            meanCols = mean(chan, 1);
            centered = chan - meanCols;
            covMatrix = cov(centered);
            [eigvecs, eigvals] = eig(covMatrix);
            [eigvals, idx] = sort(diag(eigvals), 'descend');
            eigvecs = eigvecs(:, idx);
            V_k = eigvecs(:,1:min(k,size(eigvecs,2)));
            proj = centered * V_k;
            recon = proj * V_k' + meanCols;
            recon = max(min(recon,255),0);
            if c==1
                pcaRecon = recon;
            else
                pcaRecon(:,:,cc) = recon;
            end
        end
        pcaRecon = uint8(pcaRecon);

        % Metrics
        fprintf('\nSelected k: %d\n', k);
        fprintf('Noisy  - MSE: %.4g\n', mseMetric(orig, noisy));
        fprintf('PCA    - MSE: %.4g\n', mseMetric(orig, pcaRecon));
        fprintf('SVD    - MSE: %.4g\n', mseMetric(orig, svdRecon));
        if strcmp(datatype,'image')
            fprintf('Noisy  - PSNR: %.4g, SSIM: %.4g\n', psnrMetric(orig, noisy), ssimMetric(orig, noisy, isRGB));
            fprintf('PCA    - PSNR: %.4g, SSIM: %.4g\n', psnrMetric(orig, pcaRecon), ssimMetric(orig, pcaRecon, isRGB));
            fprintf('SVD    - PSNR: %.4g, SSIM: %.4g\n', psnrMetric(orig, svdRecon), ssimMetric(orig, svdRecon, isRGB));
        end

        % Display (image only)
        if strcmp(datatype,'image')
            figure;
            if ndims(orig) == 2
                subplot(1,4,1); imagesc(orig); axis image off; title('Original');
                subplot(1,4,2); imagesc(noisy); axis image off; title('Noisy');
                subplot(1,4,3); imagesc(pcaRecon); axis image off; title('PCA');
                subplot(1,4,4); imagesc(svdRecon); axis image off; title('SVD');
                colormap gray;
            else
                subplot(1,4,1); imshow(orig); title('Original');
                subplot(1,4,2); imshow(noisy); title('Noisy');
                subplot(1,4,3); imshow(pcaRecon); title('PCA');
                subplot(1,4,4); imshow(svdRecon); title('SVD');
            end
        end
    end
end

% --- Efficient SVD (full, ONCE) ---
function [U, S, V] = customSVD_full(channel)
    [m, n] = size(channel);
    if m >= n
        A = channel' * channel; 
        [V, D] = eig(A);
        [d, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        d(d < 0) = 0; % Numerical fix
        singularValues = sqrt(d);
        U = zeros(m, length(singularValues));
        for i = 1:length(singularValues)
            if singularValues(i) > 1e-12
                U(:,i) = (1/singularValues(i)) * channel * V(:,i);
            else
                U(:,i) = zeros(m,1);
            end
        end
        S = diag(singularValues);
    else
        A = channel * channel';
        [U, D] = eig(A);
        [d, idx] = sort(diag(D), 'descend');
        U = U(:, idx);
        d(d < 0) = 0;
        singularValues = sqrt(d);
        V = zeros(n, length(singularValues));
        for i = 1:length(singularValues)
            if singularValues(i) > 1e-12
                V(:,i) = (1/singularValues(i)) * channel' * U(:,i);
            else
                V(:,i) = zeros(n,1);
            end
        end
        S = diag(singularValues);
    end
    % Truncate U/S/V to min(m,n) if needed
    minmn = min(m,n);
    U = U(:,1:minmn); S = S(1:minmn,1:minmn); V = V(:,1:minmn);
end

% --- Metrics ---
function val = mseMetric(A, B)
    A = double(A); B = double(B);
    val = mean((A(:) - B(:)).^2);
end

function val = psnrMetric(A, B)
    mse = mseMetric(A, B);
    if mse == 0
        val = 99;
    else
        val = 10*log10(255^2/mse);
    end
end

function val = ssimMetric(A, B, isRGB)
    try
        if isRGB
            val = ssim(uint8(B), uint8(A));
        else
            val = ssim(uint8(B), uint8(A));
        end
    catch
        val = 1 - mseMetric(A,B)/mean([var(double(A(:))), var(double(B(:)))]);
    end
end
