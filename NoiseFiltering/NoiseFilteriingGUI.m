function NoiseFiltering_mech505_GUI()
    fig = uifigure('Name','Noise Filtering App 5','Position',[200 200 500 450]);
    movegui(fig,'center');
    
    app = struct();
    app.mode = 1; % 1=text/csv, 2=image
    app.useDemo = true;
    app.isRGB = false;
    app.k = [];
    app.data_loaded = false;
    app.clean = [];
    app.noisy = [];
    app.kmax = [];
    app.fname = '';
    app.noiseType = 1;
    
    lblType = uilabel(fig, 'Position', [30, 400, 110, 22], 'Text', 'Data Type:');
    ddType = uidropdown(fig, 'Position', [140, 400, 130, 22], ...
        'Items', {'Text (matrix) / CSV', 'Image'}, ...
        'ValueChangedFcn', @(src,evt) setType());
    
    cbDemo = uicheckbox(fig, 'Position', [300, 400, 140, 22], ...
        'Text', 'Use Demo Data', 'Value', true, ...
        'ValueChangedFcn', @(src,evt) setDemo());
    
    lblNoise = uilabel(fig, 'Position', [30, 360, 100, 22], 'Text', 'Noise type:');
    ddNoise = uidropdown(fig, 'Position', [140, 360, 130, 22], ...
        'Items', {'Impulse', 'Gaussian'}, ...
        'ValueChangedFcn', @(src,evt) setNoiseType());
    
    btnLoad = uibutton(fig, 'Position', [30, 320, 120, 30], ...
        'Text', 'Load Data', 'ButtonPushedFcn', @(src,evt) loadData());
    
    btnNoise = uibutton(fig, 'Position', [170, 320, 120, 30], ...
        'Text', 'Add Noise', 'Enable', 'off', 'ButtonPushedFcn', @(src,evt) addNoise());
    
    lblK = uilabel(fig, 'Position', [30, 280, 170, 22], 'Text', 'k (empty for sweep):');
    edK = uieditfield(fig, 'numeric', 'Position', [200, 280, 80, 22], ...
        'Limits', [1 Inf], 'AllowEmpty', true, 'Value', []);
    
    btnRun = uibutton(fig, 'Position', [30, 230, 120, 30], ...
        'Text', 'Run Denoising', 'Enable', 'off', 'ButtonPushedFcn', @(src,evt) runDenoising());
    
    txtResults = uitextarea(fig, 'Position', [30, 20, 440, 200], ...
        'Editable', 'off', 'Value', {'Results will appear here.'});
    
    function setType()
        app.mode = ddType.Value=="Text (matrix) / CSV";
        app.data_loaded = false;
        app.clean = [];
        app.noisy = [];
        app.kmax = [];
        app.fname = '';
        btnNoise.Enable = 'off';
        btnRun.Enable = 'off';
        txtResults.Value = {'Results will appear here.'};
    end
    function setDemo()
        app.useDemo = cbDemo.Value;
        app.data_loaded = false;
        app.clean = [];
        app.noisy = [];
        app.kmax = [];
        app.fname = '';
        btnNoise.Enable = 'off';
        btnRun.Enable = 'off';
        txtResults.Value = {'Results will appear here.'};
    end
    function setNoiseType()
        app.noiseType = ddNoise.Value=="Impulse";
    end

    function loadData()
        txtResults.Value = {'Loading data...'};
        drawnow;
        if app.mode == 1 % Text/CSV
            if app.useDemo
                txtType = questdlg('Demo mode: Greyscale or RGB?','Data','Greyscale','RGB','Greyscale');
                if strcmp(txtType,'Greyscale')
                    app.clean = makeDemoMat(100,50,10,0.1)*10 + 128;
                    app.isRGB = false;
                else
                    app.clean = zeros(100,50,3);
                    for c=1:3
                        app.clean(:,:,c) = makeDemoMat(100,50,10,0.1)*10 + 128;
                    end
                    app.isRGB = true;
                end
                app.fname = 'Demo low-rank matrix';
            else
                [file, path] = uigetfile('*.csv');
                if isequal(file,0)
                    txtResults.Value = {'No file selected.'};
                    return;
                end
                app.clean = readmatrix(fullfile(path, file));
                app.isRGB = false;
                app.fname = fullfile(path, file);
            end
            app.clean = double(app.clean); % Always double for math!
            [m, n, d3] = size(app.clean);
            app.kmax = min(m,n);
            % --- Visualize textual data ---
            figure('Name','Loaded Matrix Data');
            if ndims(app.clean)==2
                imagesc(app.clean); axis image; colorbar; title('Loaded Matrix Data');
            elseif ndims(app.clean)==3
                d3 = size(app.clean,3);
                for c=1:d3
                    subplot(1,d3,c);
                    imagesc(app.clean(:,:,c)); axis image; colorbar;
                    title(sprintf('Channel %d',c));
                end
            end
        else % Image
            imgType = questdlg('Image type?','Image','RGB','Greyscale','RGB');
            app.isRGB = strcmp(imgType,'RGB');
            if app.useDemo
                app.clean = imread('peppers.png');
                app.fname = 'peppers.png';
            else
                [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif','Images'});
                if isequal(file,0)
                    txtResults.Value = {'No file selected.'};
                    return;
                end
                app.clean = imread(fullfile(path, file));
                app.fname = fullfile(path, file);
            end
            if ~app.isRGB && ndims(app.clean)==3
                app.clean = rgb2gray(app.clean);
            elseif app.isRGB && ndims(app.clean)==2
                app.clean = repmat(app.clean,1,1,3);
            end
            app.clean = double(app.clean); % Always double for math!
            [m, n, ~] = size(app.clean);
            app.kmax = min(m,n);
            figure('Name','Loaded Image Data');
            if app.isRGB
                imshow(uint8(app.clean)); title('Loaded Image (RGB)');
            else
                imagesc(app.clean); axis image; colormap gray; colorbar; title('Loaded Image (Greyscale)');
            end
        end
        txtResults.Value = {['Loaded: ' app.fname], sprintf('Data size: %s', mat2str(size(app.clean)))};
        btnNoise.Enable = 'on';
        btnRun.Enable = 'off';
    end

    function addNoise()
        txtResults.Value = {'Adding noise...'};
        drawnow;
        sz = size(app.clean);
        base = double(app.clean);
        if app.noiseType
            noisy = base + randn(sz)*10;
            mask = rand(sz)<0.01;
            tmp = noisy(mask) + 50.*(2*(rand(nnz(mask),1)>0.5)-1);
            noisy(mask) = tmp;
        else
            noisy = base + randn(sz)*20;
        end
        app.noisy = noisy;
        if app.mode == 1
            figure('Name','Noisy Matrix Data');
            if ndims(app.noisy)==2
                imagesc(app.noisy); axis image; colorbar; title('Noisy Matrix Data');
            elseif ndims(app.noisy)==3
                d3 = size(app.noisy,3);
                for c=1:d3
                    subplot(1,d3,c);
                    imagesc(app.noisy(:,:,c)); axis image; colorbar;
                    title(sprintf('Noisy Channel %d',c));
                end
            end
        else
            figure('Name','Noisy Image Data');
            if app.isRGB
                imshow(uint8(app.noisy)); title('Noisy Image (RGB)');
            else
                imagesc(uint8(app.noisy)); axis image; colormap gray; colorbar; title('Noisy Image (Greyscale)');
            end
        end
        txtResults.Value = [txtResults.Value; {'Noise added.'}];
        btnRun.Enable = 'on';
    end

    function runDenoising()
        txtResults.Value = {'Running denoising...'};
        drawnow;
        userk = edK.Value;
        if isempty(userk)
            userk = [];
        end
        if app.mode == 1
            [bestPCA, bestSVD, noisyMat, origMat, info] = runMatrixDenoiseCurves(app.noisy, app.clean, app.kmax, userk);
            addtxt = {};
            if isfield(info, 'optimumPCA_k') && ~isempty(info.optimumPCA_k)
                addtxt{end+1} = sprintf('Optimum PCA k: %d', info.optimumPCA_k);
            end
            if isfield(info, 'optimumSVD_k') && ~isempty(info.optimumSVD_k)
                addtxt{end+1} = sprintf('Optimum SVD k: %d', info.optimumSVD_k);
            end
            addtxt = addtxt(:); % ensure column cell
            txtblock = [ ...
                {'Denoising complete.'}; ...
                addtxt; ...
                {sprintf('Original (first 5x5):\n%s',mat2str(origMat(1:min(5,end),1:min(5,end))))}; ...
                {sprintf('Noisy (first 5x5):\n%s',mat2str(noisyMat(1:min(5,end),1:min(5,end))))}; ...
                {sprintf('PCA (first 5x5):\n%s',mat2str(bestPCA(1:min(5,end),1:min(5,end))))}; ...
                {sprintf('SVD (first 5x5):\n%s',mat2str(bestSVD(1:min(5,end),1:min(5,end))))} ...
            ];
            txtResults.Value = txtblock;
        else
            info = runDenoiseCurves(uint8(app.noisy), uint8(app.clean), app.kmax, app.isRGB, userk, 'image');
            addtxt = {};
            if isfield(info,'optimumPCA_k') && ~isempty(info.optimumPCA_k)
                addtxt{end+1} = sprintf('Optimum PCA k: %d', info.optimumPCA_k);
            end
            if isfield(info,'optimumSVD_k') && ~isempty(info.optimumSVD_k)
                addtxt{end+1} = sprintf('Optimum SVD k: %d', info.optimumSVD_k);
            end
            if isfield(info,'bestPCA_k') && ~isempty(info.bestPCA_k)
                addtxt{end+1} = sprintf('Best PCA k (max SSIM): %d', info.bestPCA_k);
            end
            if isfield(info,'bestSVD_k') && ~isempty(info.bestSVD_k)
                addtxt{end+1} = sprintf('Best SVD k (max SSIM): %d', info.bestSVD_k);
            end
            addtxt = addtxt(:); % ensure column
            txtResults.Value = [txtResults.Value; {'Denoising complete. See figures.'}; addtxt];
        end
    end
end

function X = makeDemoMat(ns, nf, rank, tail)
    X = randn(ns,rank)*randn(rank,nf);
    X = X + tail*randn(ns,nf);
end

function [bestPCA, bestSVD, noisyMat, origMat, info] = runMatrixDenoiseCurves(noisy, clean, kmax, userk)
    [m, n, C] = size(clean);
    origMat = clean; noisyMat = noisy; info = struct;
    if isempty(userk)
        ks = 1:kmax;
        msePCA = zeros(size(ks)); mseSVD = zeros(size(ks));
        [U,S,V] = mySVD(noisy);
        m1 = mean(noisy,1); cent = noisy - m1;
        covM = cov(cent); [E,ev] = eig(covM); [~,idx]=sort(diag(ev),'descend'); E=E(:,idx);
        bestPCA = []; bestSVD = []; bestPCA_k = []; bestSVD_k = []; bestPCA_mse = inf; bestSVD_mse = inf;
        for i=1:length(ks)
            k = ks(i);
            k_c = min([k, size(U,2), size(S,1), size(V,2)]);
            reconSVD = U(:,1:k_c)*S(1:k_c,1:k_c)*V(:,1:k_c)';
            mseSVD(i) = mean((clean(:)-reconSVD(:)).^2);
            Ek = E(:,1:min(k,size(E,2))); proj = cent*Ek; reconPCA = proj*Ek'+m1;
            msePCA(i) = mean((clean(:)-reconPCA(:)).^2);
            if msePCA(i) < bestPCA_mse, bestPCA_mse = msePCA(i); bestPCA = reconPCA; bestPCA_k = k; end
            if mseSVD(i) < bestSVD_mse, bestSVD_mse = mseSVD(i); bestSVD = reconSVD; bestSVD_k = k; end
        end
        refMSE = mean((clean(:)-noisy(:)).^2);
        info.optimumPCA_k = find(msePCA < refMSE, 1, 'first');
        info.optimumSVD_k = find(mseSVD < refMSE, 1, 'first');
        figure('Name','Matrix mode: MSE vs k','Units','normalized','Position',[.1 .3 .6 .4]);
        plot(ks, msePCA, 'b-', ks, mseSVD, 'r-', 'LineWidth', 2); grid on; hold on;
        legend('PCA','SVD','Location','northeast');
        xlabel('k'); ylabel('MSE'); title('Matrix Denoising: MSE vs k');
        yline(refMSE,'k--','Noisy','LineWidth',1.5);
        if ~isempty(info.optimumPCA_k), plot(ks(info.optimumPCA_k), msePCA(info.optimumPCA_k), 'bd', 'MarkerSize',10, 'LineWidth',2); end
        if ~isempty(info.optimumSVD_k), plot(ks(info.optimumSVD_k), mseSVD(info.optimumSVD_k), 'rd', 'MarkerSize',10, 'LineWidth',2); end
        plot(bestPCA_k, bestPCA_mse, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
        plot(bestSVD_k, bestSVD_mse, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;
        bestPCA = round(bestPCA,4); bestSVD = round(bestSVD,4); noisyMat = round(noisyMat,4); origMat = round(origMat,4);
    else
        k = userk; if k<1||k>kmax, error('k out of range'); end
        [U,S,V] = mySVD(noisy); 
        k_c = min([k, size(U,2), size(S,1), size(V,2)]);
        reconSVD = U(:,1:k_c)*S(1:k_c,1:k_c)*V(:,1:k_c)';
        m1 = mean(noisy,1); cent = noisy - m1;
        covM = cov(cent); [E,ev] = eig(covM); [~,idx]=sort(diag(ev),'descend'); E=E(:,idx);
        Ek = E(:,1:min(k,size(E,2))); proj = cent*Ek; reconPCA = proj*Ek'+m1;
        bestPCA = round(reconPCA,4); bestSVD = round(reconSVD,4); noisyMat = round(noisy,4); origMat = round(clean,4);
        info = struct;
        fprintf('k=%d\n',k);
        fprintf('Noisy: MSE=%.4g\n',mean((clean(:)-noisy(:)).^2));
        fprintf(' PCA : MSE=%.4g\n',mean((clean(:)-reconPCA(:)).^2));
        fprintf(' SVD : MSE=%.4g\n',mean((clean(:)-reconSVD(:)).^2));
    end
end

function info = runDenoiseCurves(noisy, clean, kmax, isRGB, userk, dtype)
    if nargin<4, isRGB=false; end
    if nargin<5, userk=[]; end
    if nargin<6, dtype='image'; end
    info = struct;
    if ndims(noisy)==3, C=size(noisy,3); else, C=1; end
    refMSE = mseMetric(clean,noisy);
    if strcmp(dtype,'image')
        refPSNR = psnrMetric(clean,noisy);
        refSSIM = ssimMetric(clean,noisy,isRGB);
    end
    if isempty(userk)
        ks = 1:kmax; msePCA = zeros(size(ks)); mseSVD = zeros(size(ks));
        tPCA = zeros(size(ks)); tSVD = zeros(size(ks));
        if strcmp(dtype,'image')
            psnrPCA = zeros(size(ks)); psnrSVD = zeros(size(ks));
            ssimPCA = zeros(size(ks)); ssimSVD = zeros(size(ks));
        end
        Uc=cell(1,C); Sc=cell(1,C); Vc=cell(1,C);
        for c=1:C
            if C==1 && ndims(noisy)==2, chan = double(noisy);
            else, chan = double(noisy(:,:,c)); end
            [U,S,V] = mySVD(chan); Uc{c}=U; Sc{c}=S; Vc{c}=V;
        end
        Mc=cell(1,C); Ec=cell(1,C);
        for c=1:C
            if C==1 && ndims(noisy)==2, chan = double(noisy);
            else, chan = double(noisy(:,:,c)); end
            m1 = mean(chan,1); cent = chan-m1;
            covM = cov(cent); [E,ev] = eig(covM); [~,idx]=sort(diag(ev),'descend'); E=E(:,idx);
            Mc{c}=m1; Ec{c}=E;
        end
        for i=1:length(ks)
            k=ks(i);
            tic;
            if C==1
                k_c = min([k, size(Uc{1},2), size(Sc{1},1), size(Vc{1},2)]);
                recon = Uc{1}(:,1:k_c)*Sc{1}(1:k_c,1:k_c)*Vc{1}(:,1:k_c)';
                recon = min(max(recon,0),255); svdRecon = uint8(recon);
            else
                svdRecon = zeros(size(noisy),'double');
                for c=1:C
                    k_c = min([k, size(Uc{c},2), size(Sc{c},1), size(Vc{c},2)]);
                    recon = Uc{c}(:,1:k_c)*Sc{c}(1:k_c,1:k_c)*Vc{c}(:,1:k_c)';
                    recon = min(max(recon,0),255);
                    [m, n] = size(noisy(:,:,c));
                    recon = reshape(recon, m, n);
                    svdRecon(:,:,c) = recon;
                end
                svdRecon=uint8(svdRecon);
            end
            tSVD(i) = toc;
            tic;
            if C==1
                Ek = Ec{1}(:,1:min(k,size(Ec{1},2)));
                proj = (double(noisy)-Mc{1})*Ek;
                recon = proj*Ek' + Mc{1};
                recon = min(max(recon,0),255); pcaRecon = uint8(recon);
            else
                pcaRecon = zeros(size(noisy),'double');
                for c=1:C
                    Ek = Ec{c}(:,1:min(k,size(Ec{c},2)));
                    proj = (double(noisy(:,:,c))-Mc{c})*Ek;
                    recon = proj*Ek' + Mc{c};
                    recon = min(max(recon,0),255);
                    [m, n] = size(noisy(:,:,c));
                    recon = reshape(recon, m, n);
                    pcaRecon(:,:,c) = recon;
                end
                pcaRecon=uint8(pcaRecon);
            end
            tPCA(i) = toc;
            msePCA(i) = mseMetric(clean,pcaRecon);
            mseSVD(i) = mseMetric(clean,svdRecon);
            if strcmp(dtype,'image')
                psnrPCA(i) = psnrMetric(clean,pcaRecon);
                psnrSVD(i) = psnrMetric(clean,svdRecon);
                ssimPCA(i) = ssimMetric(clean,pcaRecon,isRGB);
                ssimSVD(i) = ssimMetric(clean,svdRecon,isRGB);
            end
            if mod(i,10)==0 || i==1 || i==length(ks)
                fprintf('k=%d/%d done\n',k,kmax);
            end
        end
        info.optimumPCA_k = find(msePCA < refMSE, 1, 'first');
        info.optimumSVD_k = find(mseSVD < refMSE, 1, 'first');
        if strcmp(dtype,'image')
            info.optimumPCA_k_ssim = find(ssimPCA > refSSIM, 1, 'first');
            info.optimumSVD_k_ssim = find(ssimSVD > refSSIM, 1, 'first');
            [~,info.bestPCA_k]=max(ssimPCA);
            [~,info.bestSVD_k]=max(ssimSVD);
        end
        figure('Name','All metrics vs k','Units','normalized','Position',[.1 .2 .7 .6]);
        subplot(2,2,1);
        plot(ks,msePCA,'b-',ks,mseSVD,'r-','LineWidth',2); hold on
        yline(refMSE,'k--','LineWidth',1.5);
        if strcmp(dtype,'image')
            plot(ks(info.bestPCA_k),msePCA(info.bestPCA_k),'bo','MarkerSize',10,'LineWidth',2);
            plot(ks(info.bestSVD_k),mseSVD(info.bestSVD_k),'ro','MarkerSize',10,'LineWidth',2);
            if ~isempty(info.optimumPCA_k), plot(ks(info.optimumPCA_k), msePCA(info.optimumPCA_k), 'bd', 'MarkerSize',10, 'LineWidth',2); end
            if ~isempty(info.optimumSVD_k), plot(ks(info.optimumSVD_k), mseSVD(info.optimumSVD_k), 'rd', 'MarkerSize',10, 'LineWidth',2); end
            legend('PCA','SVD','Noisy','PCA best','SVD best','PCA opt','SVD opt','Location','northeast'); 
        else
            legend('PCA','SVD','Noisy','Location','northeast');
        end
        xlabel('k'); ylabel('MSE'); title('MSE vs k'); grid on; hold off
        subplot(2,2,2);
        plot(ks,tPCA,'b-',ks,tSVD,'r-','LineWidth',2);
        legend('PCA','SVD'); xlabel('k'); ylabel('Time (s)'); title('Time vs k'); grid on;
        if strcmp(dtype,'image')
            subplot(2,2,3);
            plot(ks,psnrPCA,'b-',ks,psnrSVD,'r-','LineWidth',2); hold on
            yline(refPSNR,'k--','LineWidth',1.5);
            plot(ks(info.bestPCA_k),psnrPCA(info.bestPCA_k),'bo','MarkerSize',10,'LineWidth',2);
            plot(ks(info.bestSVD_k),psnrSVD(info.bestSVD_k),'ro','MarkerSize',10,'LineWidth',2);
            legend('PCA','SVD','Noisy','PCA best','SVD best','Location','southeast');
            xlabel('k'); ylabel('PSNR (dB)'); title('PSNR vs k'); grid on; hold off
            subplot(2,2,4);
            plot(ks,ssimPCA,'b-',ks,ssimSVD,'r-','LineWidth',2); hold on
            yline(refSSIM,'k--','LineWidth',1.5);
            plot(ks(info.bestPCA_k),ssimPCA(info.bestPCA_k),'bo','MarkerSize',10,'LineWidth',2);
            plot(ks(info.bestSVD_k),ssimSVD(info.bestSVD_k),'ro','MarkerSize',10,'LineWidth',2);
            if ~isempty(info.optimumPCA_k_ssim), plot(ks(info.optimumPCA_k_ssim), ssimPCA(info.optimumPCA_k_ssim), 'bd', 'MarkerSize',10, 'LineWidth',2); end
            if ~isempty(info.optimumSVD_k_ssim), plot(ks(info.optimumSVD_k_ssim), ssimSVD(info.optimumSVD_k_ssim), 'rd', 'MarkerSize',10, 'LineWidth',2); end
            legend('PCA','SVD','Noisy','PCA best','SVD best','PCA opt','SVD opt','Location','southeast');
            xlabel('k'); ylabel('SSIM'); title('SSIM vs k'); grid on; hold off
        else
            subplot(2,2,3); axis off; text(0.1,0.5,'PSNR not for text mode','FontSize',14);
            subplot(2,2,4); axis off; text(0.1,0.5,'SSIM not for text mode','FontSize',14);
        end
    else
        k = userk; if k<1||k>kmax, error('k out of range'); end
        svdRecon = zeros(size(noisy),'double');
        for c=1:C
            if C==1 && ndims(noisy)==2, chan = double(noisy);
            else, chan = double(noisy(:,:,c)); end
            [U,S,V]=mySVD(chan);
            k_c = min([k, size(U,2), size(S,1), size(V,2)]);
            r = U(:,1:k_c)*S(1:k_c,1:k_c)*V(:,1:k_c)';
            r = min(max(r,0),255);
            [m, n] = size(noisy(:,:,c));
            r = reshape(r, m, n);
            if C==1
                svdRecon = r;
            else
                svdRecon(:,:,c)=r;
            end
        end
        svdRecon=uint8(svdRecon);
        pcaRecon = zeros(size(noisy),'double');
        for c=1:C
            if C==1 && ndims(noisy)==2, chan = double(noisy);
            else, chan = double(noisy(:,:,c)); end
            m1=mean(chan,1); cent=chan-m1;
            covM = cov(cent); [E,ev]=eig(covM); [~,idx]=sort(diag(ev),'descend'); E=E(:,idx);
            Ek = E(:,1:min(k,size(E,2))); proj = cent*Ek; r = proj*Ek'+m1;
            r = min(max(r,0),255);
            [m, n] = size(noisy(:,:,c));
            r = reshape(r, m, n);
            if C==1
                pcaRecon = r;
            else
                pcaRecon(:,:,c)=r;
            end
        end
        pcaRecon=uint8(pcaRecon);
        fprintf('k=%d\n',k);
        fprintf('Noisy: MSE=%.4g\n',mseMetric(clean,noisy));
        fprintf(' PCA : MSE=%.4g\n',mseMetric(clean,pcaRecon));
        fprintf(' SVD : MSE=%.4g\n',mseMetric(clean,svdRecon));
        if strcmp(dtype,'image')
            fprintf('Noisy: PSNR=%.4g, SSIM=%.4g\n',psnrMetric(clean,noisy),ssimMetric(clean,noisy,isRGB));
            fprintf(' PCA : PSNR=%.4g, SSIM=%.4g\n',psnrMetric(clean,pcaRecon),ssimMetric(clean,pcaRecon,isRGB));
            fprintf(' SVD : PSNR=%.4g, SSIM=%.4g\n',psnrMetric(clean,svdRecon),ssimMetric(clean,svdRecon,isRGB));
            figure;
            if ndims(clean)==2
                subplot(1,4,1); imagesc(clean); axis image off; title('Original');
                subplot(1,4,2); imagesc(noisy); axis image off; title('Noisy');
                subplot(1,4,3); imagesc(pcaRecon); axis image off; title('PCA');
                subplot(1,4,4); imagesc(svdRecon); axis image off; title('SVD'); colormap gray;
            else
                subplot(1,4,1); imshow(clean); title('Original');
                subplot(1,4,2); imshow(noisy); title('Noisy');
                subplot(1,4,3); imshow(pcaRecon); title('PCA');
                subplot(1,4,4); imshow(svdRecon); title('SVD');
            end
        end
    end
end

function [U,S,V]=mySVD(A)
    [m,n]=size(A);
    if m>=n
        [V,D]=eig(A'*A);
        [d,idx]=sort(diag(D),'descend');
        V=V(:,idx); d(d<0)=0; s=sqrt(d);
        U=zeros(m,length(s));
        for i=1:length(s)
            if s(i)>1e-12
                U(:,i) = (1/s(i))*A*V(:,i);
            else
                U(:,i)=zeros(m,1);
            end
        end
        S=diag(s);
    else
        [U,D]=eig(A*A');
        [d,idx]=sort(diag(D),'descend');
        U=U(:,idx); d(d<0)=0; s=sqrt(d);
        V=zeros(n,length(s));
        for i=1:length(s)
            if s(i)>1e-12
                V(:,i) = (1/s(i))*A'*U(:,i);
            else
                V(:,i)=zeros(n,1);
            end
        end
        S=diag(s);
    end
    minmn = min(m,n);
    U=U(:,1:minmn); S=S(1:minmn,1:minmn); V=V(:,1:minmn);
end

function v=mseMetric(A,B), v=mean((double(A(:))-double(B(:))).^2); end
function v=psnrMetric(A,B), m=mseMetric(A,B); v=99*(m==0)+10*log10(255^2/m)*(m~=0); end
function v=ssimMetric(A,B,isRGB)
    try
        v = ssim(uint8(B),uint8(A));
    catch
        v = 1-mseMetric(A,B)/mean([var(double(A(:))),var(double(B(:)))]);
    end
end
