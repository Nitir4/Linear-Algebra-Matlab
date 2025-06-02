function ImageCompressionApp_mech505_GUI()
% PCA/SVD image compression GUI, with pop-up windows for all results
% (compare, compress, analyze), and main UI always stays the same as in terminal.
% Compress with PCA also displays the compressed (projected) image.

    fig = uifigure('Name', 'Image Compression App', 'Position', [100 100 950 700]);
    appdata = struct();
    appdata.origImage = [];
    appdata.origImgPath = '';
    appdata.isRGB = true;
    appdata.origData = [];
    appdata.lastCompImage = []; % For saving

    % Only the loaded image is shown in the main window
    ax = uiaxes(fig, 'Position', [330 340 600 340]);
    ax.Visible = 'on'; % Only used for preview

    btnLoad = uibutton(fig, 'Position', [30 630 210 30], ...
        'Text', 'Load Image', 'ButtonPushedFcn', @(btn, event) onLoadImage());

    lblMode = uilabel(fig, 'Position', [30 590 80 25], 'Text', 'Mode:');
    bgMode = uibuttongroup(fig, 'Position', [100 585 120 40]);
    rbRGB = uiradiobutton(bgMode, 'Text', 'RGB', 'Position', [10 10 50 22], 'Value', 1);
    rbGray = uiradiobutton(bgMode, 'Text', 'Gray', 'Position', [70 10 60 22]);

    lblAction = uilabel(fig, 'Position', [30 540 80 25], 'Text', 'Action:');
    ddAction = uidropdown(fig, 'Position', [100 540 120 25], ...
        'Items', {'Compress', 'Compare', 'Analyze'}, 'Value', 'Compress');

    lblMethod = uilabel(fig, 'Position', [30 495 80 25], 'Text', 'Method:');
    ddMethod = uidropdown(fig, 'Position', [100 495 120 25], ...
        'Items', {'PCA', 'SVD'}, 'Value', 'PCA');

    lblK = uilabel(fig, 'Position', [30 450 80 25], 'Text', 'k:');
    editK = uieditfield(fig, 'numeric', 'Position', [100 450 120 25], 'Value', 20, 'Limits', [1 500]);

    lblAlg = uilabel(fig, 'Position', [30 410 80 25], 'Text', 'Analyze:');
    ddAlg = uidropdown(fig, 'Position', [100 410 120 25], ...
        'Items', {'Only PCA', 'Only SVD', 'Both PCA and SVD'}, 'Value', 'Both PCA and SVD');

    btnRun = uibutton(fig, 'Position', [30 350 210 40], ...
        'Text', 'Run', 'FontWeight', 'bold', 'ButtonPushedFcn', @(btn, event) onRunAction());

    btnSave = uibutton(fig, 'Position', [30 300 210 30], ...
        'Text', 'Save Reconstructed Image', 'Enable', 'off', 'ButtonPushedFcn', @(btn, event) onSaveImage());

    txtMetrics = uitextarea(fig, 'Position', [330 30 600 260], ...
        'Editable', 'off', 'FontSize', 12, 'Value', {'Metrics and logs will appear here.'});

    %% --- CALLBACKS ---
    function onLoadImage()
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'});
        if isequal(file, 0)
            txtMetrics.Value = {'No file selected.'};
            return;
        end
        appdata.origImgPath = fullfile(path, file);
        appdata.origImage = imread(appdata.origImgPath);
        cla(ax); % clear previous loaded image
        imshow(appdata.origImage, 'Parent', ax);
        title(ax, 'Original Image');
        txtMetrics.Value = {['Loaded image: ', file]};
        appdata.origData = [];
        btnSave.Enable = 'off';
    end

    function onRunAction()
        if isempty(appdata.origImage)
            txtMetrics.Value = {'Please load an image first.'};
            return;
        end
        mode = bgMode.SelectedObject.Text;
        if strcmpi(mode, 'Gray')
            appdata.origData = rgb2gray(appdata.origImage);
            appdata.isRGB = false;
        else
            appdata.origData = appdata.origImage;
            appdata.isRGB = true;
        end
        action = ddAction.Value;
        k = round(editK.Value);
        method = ddMethod.Value;
        algChoice = find(strcmp(ddAlg.Items, ddAlg.Value));
        if appdata.isRGB
            [m, n, ~] = size(appdata.origData);
        else
            [m, n] = size(appdata.origData);
        end
        maxK = min(m, n);
        editK.Limits = [1 maxK];
        if k > maxK
            txtMetrics.Value = {'k is too large for this image.'};
            return;
        end

        switch lower(action)
            case 'compress'
                [compImage, compressedData, metrics, compTime] = compressAndShow(appdata.origData, k, method, appdata.isRGB);
                txtMetrics.Value = metrics;
                appdata.lastCompImage = compImage;
                btnSave.Enable = 'on';
            case 'compare'
                [pcaRecon, svdRecon, metrics] = compareMethods(appdata.origData, k, appdata.isRGB);
                popupCompare(uint8(appdata.origData), pcaRecon, svdRecon); % <--- popup only, UI is unchanged
                txtMetrics.Value = metrics;
                btnSave.Enable = 'off';
            case 'analyze'
                analyzeMetricsOptimized(appdata.origData, appdata.isRGB, algChoice, txtMetrics);
                btnSave.Enable = 'off';
        end
    end

    function onSaveImage()
        if isfield(appdata, 'lastCompImage') && ~isempty(appdata.lastCompImage)
            [file, path] = uiputfile({'*.png', 'PNG (*.png)'; '*.jpg', 'JPEG (*.jpg)'; '*.bmp', 'Bitmap (*.bmp)'}, 'Save Image As');
            if isequal(file, 0)
                txtMetrics.Value = {'Save cancelled.'};
                return;
            end
            imwrite(appdata.lastCompImage, fullfile(path, file));
            txtMetrics.Value = {['Image saved to ', fullfile(path, file)]};
        end
    end
end

%% --- POPUP DISPLAY HELPERS ---

function popupCompare(origIm, pcaIm, svdIm)
    % Show a popup window with three labeled images: original, PCA, SVD
    f = figure('Name','Compare: Original | PCA | SVD','NumberTitle','off','Position',[100 100 1200 400]);
    t = tiledlayout(f,1,3,'TileSpacing','Compact','Padding','Compact');
    ax1 = nexttile(t,1); imshow(origIm, 'Parent', ax1); title(ax1,'Original');
    ax2 = nexttile(t,2); imshow(pcaIm, 'Parent', ax2); title(ax2,'PCA reconstructed');
    ax3 = nexttile(t,3); imshow(svdIm, 'Parent', ax3); title(ax3,'SVD reconstructed');
end

function popupCompress(origIm, compProj, reconIm, method, isRGB)
    % Show a popup window for compress with three panels and clear titles
    f = figure('Name',['Compression: ',method],'NumberTitle','off','Position',[100 100 1300 400]);
    t = tiledlayout(f,1,3,'TileSpacing','Compact','Padding','Compact');
    % Original
    ax1 = nexttile(t,1); imshow(origIm, 'Parent', ax1); title(ax1,'Original Image');
    % Compressed
    ax2 = nexttile(t,2);
    if strcmpi(method,'PCA')
        % Show PCA projection as normalized image
        if iscell(compProj) % RGB
            k_proj = size(compProj{1},2);
            m_proj = size(compProj{1},1);
            pcaProjRGB = zeros(m_proj, k_proj, 3);
            for c = 1:3
                pc = compProj{c};
                pc = (pc - min(pc(:))) / (max(pc(:)) - min(pc(:)) + eps);
                pcaProjRGB(:,:,c) = pc;
            end
            imshow(pcaProjRGB, 'Parent', ax2);
        else % Gray
            pc = compProj;
            pc = (pc - min(pc(:))) / (max(pc(:)) - min(pc(:)) + eps);
            imshow(pc, 'Parent', ax2);
        end
        title(ax2, 'Compressed Image (PCA compressed)');
    else
        axis(ax2, 'off'); text(0.5,0.5,'','HorizontalAlignment','center','FontSize',12);
    end
    % Reconstructed
    ax3 = nexttile(t,3); imshow(reconIm, 'Parent', ax3); title(ax3,'Reconstructed Image');
end

function popupAnalyze(kValues, pcaTimes, svdTimes, pcaMSEs, svdMSEs, pcaPSNRs, svdPSNRs, pcaSSIMs, svdSSIMs, algChoice)
    % Show a popup window with all four metric plots
    f = figure('Name','Analysis: Compression Metrics vs k','NumberTitle','off','Position',[100 100 900 700]);
    t = tiledlayout(f,2,2,'TileSpacing','Compact','Padding','Compact');
    % 1: Time
    ax1 = nexttile(t,1);
    hold(ax1,'on');
    if algChoice == 1
        plot(ax1, kValues, pcaTimes, 'b-o', 'LineWidth', 2); legend(ax1,'PCA Time');
    elseif algChoice == 2
        plot(ax1, kValues, svdTimes, 'r--s', 'LineWidth', 2); legend(ax1,'SVD Time');
    else
        plot(ax1, kValues, pcaTimes, 'b-o', kValues, svdTimes, 'r--s', 'LineWidth', 2);
        legend(ax1,'PCA Time', 'SVD Time');
    end
    hold(ax1,'off'); xlabel(ax1,'k'); ylabel(ax1,'Time (s)'); title(ax1,'Compression Time vs k');
    % 2: MSE
    ax2 = nexttile(t,2);
    hold(ax2,'on');
    if algChoice == 1
        plot(ax2, kValues, pcaMSEs, 'b-o', 'LineWidth', 2); legend(ax2,'PCA MSE');
    elseif algChoice == 2
        plot(ax2, kValues, svdMSEs, 'r--s', 'LineWidth', 2); legend(ax2,'SVD MSE');
    else
        plot(ax2, kValues, pcaMSEs, 'b-o', kValues, svdMSEs, 'r--s', 'LineWidth', 2);
        legend(ax2,'PCA MSE', 'SVD MSE');
    end
    hold(ax2,'off'); xlabel(ax2,'k'); ylabel(ax2,'MSE'); title(ax2,'MSE vs k');
    % 3: PSNR
    ax3 = nexttile(t,3);
    hold(ax3,'on');
    if algChoice == 1
        plot(ax3, kValues, pcaPSNRs, 'b-o', 'LineWidth', 2); legend(ax3,'PCA PSNR');
    elseif algChoice == 2
        plot(ax3, kValues, svdPSNRs, 'r--s', 'LineWidth', 2); legend(ax3,'SVD PSNR');
    else
        plot(ax3, kValues, pcaPSNRs, 'b-o', kValues, svdPSNRs, 'r--s', 'LineWidth', 2);
        legend(ax3,'PCA PSNR', 'SVD PSNR');
    end
    hold(ax3,'off'); xlabel(ax3,'k'); ylabel(ax3,'PSNR (dB)'); title(ax3,'PSNR vs k');
    % 4: SSIM
    ax4 = nexttile(t,4);
    hold(ax4,'on');
    if algChoice == 1
        plot(ax4, kValues, pcaSSIMs, 'b-o', 'LineWidth', 2); legend(ax4,'PCA SSIM');
    elseif algChoice == 2
        plot(ax4, kValues, svdSSIMs, 'r--s', 'LineWidth', 2); legend(ax4,'SVD SSIM');
    else
        plot(ax4, kValues, pcaSSIMs, 'b-o', kValues, svdSSIMs, 'r--s', 'LineWidth', 2);
        legend(ax4,'PCA SSIM', 'SVD SSIM');
    end
    hold(ax4,'off'); xlabel(ax4,'k'); ylabel(ax4,'SSIM'); title(ax4,'SSIM vs k');
end

%% --- LOGIC ---

function [compImage, compressedData, metrics, compTime] = compressAndShow(data, k, method, isRGB)
    tStart = tic;
    [compImage, compressedData] = compressImageFull(data, k, method, isRGB);
    compTime = toc(tStart);
    if isRGB
        [m, n, ~] = size(data);
    else
        [m, n] = size(data);
    end
    origUncompressedSize = m * n * (isRGB * 2 + 1);
    compressedDataSize = getCompressedDataSize(compressedData, method, m, n, k);
    compressionPercent = ((origUncompressedSize - compressedDataSize) / origUncompressedSize) * 100;
    compressionRatio = origUncompressedSize / compressedDataSize;
    mseVal = mseMetric(data, compImage);
    psnrVal = psnrMetric(data, compImage);
    ssimVal = ssimMetric(data, compImage, isRGB);

    % Popup visualization
    popupCompress(uint8(data), compressedData, compImage, method, isRGB);

    metrics = {...
        sprintf('%s Compression:', upper(method)), ...
        sprintf('Time: %.4fs', compTime), ...
        sprintf('MSE: %.4f', mseVal), ...
        sprintf('PSNR: %.2f dB', psnrVal), ...
        sprintf('SSIM: %.4f', ssimVal), ...
        sprintf('Orig. Size: %s', formatSize(origUncompressedSize)), ...
        sprintf('Compressed Size: %s', formatSize(compressedDataSize)), ...
        sprintf('Compression %%: %.2f%%', compressionPercent), ...
        sprintf('Compression Ratio: %.2f', compressionRatio)};
end

function [pcaRecon, svdRecon, metrics] = compareMethods(data, k, isRGB)
    if isRGB
        [m, n, ~] = size(data);
    else
        [m, n] = size(data);
    end
    origUncompressedSize = m * n * (isRGB * 2 + 1);
    [pcaRecon, pcaCompressed] = compressImageFull(data, k, 'PCA', isRGB);
    pcaMSE = mseMetric(data, pcaRecon);
    pcaPSNR = psnrMetric(data, pcaRecon);
    pcaSSIM = ssimMetric(data, pcaRecon, isRGB);
    pcaSize = getCompressedDataSize(pcaCompressed, 'PCA', m, n, k);
    pcaRatio = origUncompressedSize / pcaSize;
    [svdRecon, svdCompressed] = compressImageFull(data, k, 'SVD', isRGB);
    svdMSE = mseMetric(data, svdRecon);
    svdPSNR = psnrMetric(data, svdRecon);
    svdSSIM = ssimMetric(data, svdRecon, isRGB);
    svdSize = getCompressedDataSize(svdCompressed, 'SVD', m, n, k);
    svdRatio = origUncompressedSize / svdSize;
    metrics = {...
        sprintf('PCA:    MSE=%.4f  PSNR=%.2f dB  SSIM=%.4f  CR=%.2f', pcaMSE, pcaPSNR, pcaSSIM, pcaRatio), ...
        sprintf('SVD:    MSE=%.4f  PSNR=%.2f dB  SSIM=%.4f  CR=%.2f', svdMSE, svdPSNR, svdSSIM, svdRatio)};
end

function analyzeMetricsOptimized(data, isRGB, algChoice, txtMetrics)
    if isRGB
        [m, n, ~] = size(data);
        c = 3;
    else
        [m, n] = size(data);
        c = 1;
    end
    maxK = min(m, n);
    kValues = unique(round(linspace(1, maxK, min(20,maxK))));
    if algChoice == 1 || algChoice == 3
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
    for i = 1:length(kValues)
        k = kValues(i);
        if algChoice == 1 || algChoice == 3
            t1 = tic;
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
            pcaTimes(i) = toc(t1);
            pcaMSEs(i) = mseMetric(data, pcaRecon);
            pcaPSNRs(i) = psnrMetric(data, pcaRecon);
            pcaSSIMs(i) = ssimMetric(data, pcaRecon, isRGB);
        end
        if algChoice == 2 || algChoice == 3
            t2 = tic;
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
            svdTimes(i) = toc(t2);
            svdMSEs(i) = mseMetric(data, svdRecon);
            svdPSNRs(i) = psnrMetric(data, svdRecon);
            svdSSIMs(i) = ssimMetric(data, svdRecon, isRGB);
        end
        txtMetrics.Value = {sprintf('Processing k = %d/%d', k, maxK)};
        drawnow;
    end
    popupAnalyze(kValues, getOrNaN(pcaTimes,algChoice,1), getOrNaN(svdTimes,algChoice,2), ...
        getOrNaN(pcaMSEs,algChoice,1), getOrNaN(svdMSEs,algChoice,2), ...
        getOrNaN(pcaPSNRs,algChoice,1), getOrNaN(svdPSNRs,algChoice,2), ...
        getOrNaN(pcaSSIMs,algChoice,1), getOrNaN(svdSSIMs,algChoice,2), algChoice);
    txtMetrics.Value = {'Analysis complete. See popup for all metrics.'};
end

function arr = getOrNaN(arr, algChoice, idx)
    % If not used, returns NaN array for plotting compatibility
    if (algChoice==1 && idx==2) || (algChoice==2 && idx==1)
        arr = nan(size(arr));
    end
end

% --- Helper logic ---
function [compImage, compressedData] = compressImageFull(data, k, method, isRGB)
    if isRGB
        channels = size(data, 3);
        compChannels = cell(1, channels);
        compressedData = cell(1, channels);
        for c = 1:channels
            channelData = double(data(:,:,c));
            if strcmpi(method, 'PCA')
                [compChannels{c}, compressedData{c}] = customPCA(channelData, k);
            else
                [compChannels{c}, compressedData{c}] = customSVD(channelData, k);
            end
        end
        compImage = uint8(cat(3, compChannels{:}));
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
    meanCols = mean(channel, 1);
    centered = channel - meanCols;
    covMatrix = cov(centered);
    [eigvecs, eigvals] = eig(covMatrix);
    [eigvals, idx] = sort(diag(eigvals), 'descend');
    eigvecs = eigvecs(:, idx);
    V_k = eigvecs(:, 1:min(k, size(eigvecs, 2)));
    proj = centered * V_k;
    recon = proj * V_k' + meanCols;
    recon = max(min(recon, 255), 0);
    compressedData = proj;
end

function [recon, compressedData] = customSVD(channel, k)
    [m, n] = size(channel);
    if m >= n
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
        compressedData = struct('U', U_k, 'S', diag(S_k), 'V', V_k);
    else
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
    recon = max(min(recon, 255), 0);
end

function sizeBytes = getCompressedDataSize(compressedData, method, m, n, k)
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
    else
        sizeBytes = numel(compressedData) * 8;
    end
end

function sizeStr = formatSize(sizeBytes)
    if sizeBytes < 1024
        sizeStr = sprintf('%d bytes', sizeBytes);
    elseif sizeBytes < 1024^2
        sizeStr = sprintf('%.2f kB', sizeBytes / 1024);
    else
        sizeStr = sprintf('%.2f MB', sizeBytes / 1024^2);
    end
end

function mseVal = mseMetric(orig, recon)
    orig = double(orig);
    recon = double(recon);
    mseVal = mean((orig(:) - recon(:)).^2);
end

function psnrVal = psnrMetric(orig, recon)
    mseVal = mseMetric(orig, recon);
    if mseVal == 0
        psnrVal = Inf;
    else
        maxPixel = 255;
        psnrVal = 10 * log10(maxPixel^2 / mseVal);
    end
end

function ssimVal = ssimMetric(orig, recon, isRGB)
    if isRGB
        ssimVal = 0;
        for c = 1:size(orig, 3)
            ssimVal = ssimVal + ssim(orig(:,:,c), recon(:,:,c));
        end
        ssimVal = ssimVal / size(orig, 3);
    else
        ssimVal = ssim(orig, recon);
    end
end
