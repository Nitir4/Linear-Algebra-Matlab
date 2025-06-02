% Denoising numerical matrices or images with PCA & SVD, metrics-vs-k.
% Greyscale/RGB for demo data only. User-provided CSV data is always treated as a matrix.
% After denoising a CSV ("Text (matrix)"), display the original, noisy, PCA, and SVD denoised matrices.
% For large data, ask the user before compressing to k=50.

function NoiseFilteringApp5()
    clc;
    disp('--- Noise filtering with PCA & SVD ---');

    % Ask data type and source
    modeType = menu('What do you want to denoise?','Text (matrix) / CSV','Image');
    useDemo = questdlg('Use demo data?','Source','Yes','No','Yes');
    useDemo = strcmp(useDemo,'Yes');

    % Choose noise type
    noiseType = menu('Noise type?','Impulse noise (random spikes)','Gaussian white noise');

    if modeType == 1
        % --------- TEXT DATA (matrix) ---------
        if useDemo
            txtType = menu('Text: Grey or RGB?','Greyscale','RGB');
            if txtType == 1
                clean = makeDemoMat(100,50,10,0.1)*10 + 128;
                isRGB = false;
            else
                clean = zeros(100,50,3);
                for c=1:3
                    clean(:,:,c) = makeDemoMat(100,50,10,0.1)*10 + 128;
                end
                isRGB = true;
            end
        else
            [f, p] = uigetfile('*.csv');
            if isequal(f,0)
                disp('No file picked. Bye!');
                return;
            end
            clean = readmatrix(fullfile(p,f));
            isRGB = false;
        end

        sz = size(clean);

        switch noiseType
            case 1 % Impulse
                noisy = clean + randn(sz)*10;
                mask = rand(sz)<0.01;
                tmp = noisy(mask) + 50.*(2*(rand(nnz(mask),1)>0.5)-1);
                noisy(mask) = tmp;
            case 2 % Gaussian
                noisy = clean + randn(sz)*20;
        end

        [m, n, ~] = size(clean);
        kmax = min(m,n);

        % Optionally compress large data
        if max(m,n) > 100
            compressAns = questdlg(sprintf('Data is %dx%d. Compress to k=50 for speed?', m, n), ...
                'Compress?', 'Yes', 'No', 'Yes');
            if strcmp(compressAns, 'Yes')
                kmax = min(kmax, 50);
            end
        end

        userk = input(sprintf('Pick k (1-%d) or leave blank for sweep: ',kmax));
        [bestPCA, bestSVD, noisyMat, origMat] = runMatrixDenoiseCurves(noisy, clean, kmax, userk);

        % Show the original, noisy, PCA, and SVD matrices
        fprintf('\n--- Matrix Denoising Results ---\n');
        disp('Original matrix:');
        disp(origMat);
        disp('Noisy matrix:');
        disp(noisyMat);
        disp('Denoised (PCA):');
        disp(bestPCA);
        disp('Denoised (SVD):');
        disp(bestSVD);
        return
    else
        % --------- IMAGE DATA ---------
        isRGB = menu('Image type?','RGB','Greyscale') == 1;
        if useDemo
            img = imread('peppers.png');
        else
            [f,p] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif','Images'});
            if isequal(f,0)
                disp('No img. Bye!');
                return;
            end
            img = imread(fullfile(p,f));
        end
        if ~isRGB && ndims(img)==3
            img = rgb2gray(img);
        elseif isRGB && ndims(img)==2
            img = repmat(img,1,1,3);
        end
        [m,n,~] = size(img); kmax = min(m,n);

        % Optionally compress large data
        if max(m,n)>128
            compressAns = questdlg(sprintf('Img is %dx%d. Compress to 128x128 for speed?',m,n),'Compress?','Yes','No','Yes');
            if strcmp(compressAns,'Yes')
                img = imresize(img,[128 128]);
                [m,n,~] = size(img); kmax = min(m,n);
            end
        end

        clean = double(img);
        sz = size(img);

        switch noiseType
            case 1 % Impulse
                noisy = double(img) + randn(sz)*10;
                mask = rand(sz)<0.01;
                tmp = noisy(mask) + 50.*(2*(rand(nnz(mask),1)>0.5)-1);
                noisy(mask) = tmp;
            case 2 % Gaussian
                noisy = double(img) + randn(sz)*20;
        end

        userk = input(sprintf('Pick k (1-%d) or blank for sweep: ',kmax));
        runDenoiseCurves(uint8(noisy), uint8(clean), kmax, isRGB, userk, 'image');
        return
    end
end

function X = makeDemoMat(ns, nf, rank, tail)
    X = randn(ns,rank)*randn(rank,nf);
    X = X + tail*randn(ns,nf);
end

function [bestPCA, bestSVD, noisyMat, origMat] = runMatrixDenoiseCurves(noisy, clean, kmax, userk)
    [m, n, C] = size(clean);
    origMat = clean;
    noisyMat = noisy;

    if isempty(userk)
        ks = 1:kmax;
        msePCA = zeros(size(ks));
        mseSVD = zeros(size(ks));
        % SVD decomp
        [U,S,V] = svd(noisy, 'econ');
        % PCA decomp
        m1 = mean(noisy,1);
        cent = noisy - m1;
        covM = cov(cent);
        [E,ev] = eig(covM);
        [~,idx]=sort(diag(ev),'descend');
        E=E(:,idx);

        bestPCA = [];
        bestSVD = [];
        bestPCA_k = [];
        bestSVD_k = [];
        bestPCA_mse = inf;
        bestSVD_mse = inf;

        for i=1:length(ks)
            k = ks(i);
            % SVD
            reconSVD = U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
            mseSVD(i) = mean((clean(:)-reconSVD(:)).^2);

            % PCA
            Ek = E(:,1:min(k,size(E,2)));
            proj = cent*Ek;
            reconPCA = proj*Ek'+m1;
            msePCA(i) = mean((clean(:)-reconPCA(:)).^2);

            % Track best (lowest MSE)
            if msePCA(i) < bestPCA_mse
                bestPCA_mse = msePCA(i);
                bestPCA = reconPCA;
                bestPCA_k = k;
            end
            if mseSVD(i) < bestSVD_mse
                bestSVD_mse = mseSVD(i);
                bestSVD = reconSVD;
                bestSVD_k = k;
            end
        end

        % Plot MSE vs k
        figure('Name','Matrix mode: MSE vs k','Units','normalized','Position',[.1 .3 .6 .4]);
        plot(ks, msePCA, 'b-', ks, mseSVD, 'r-', 'LineWidth', 2); grid on; hold on;
        legend('PCA','SVD','Location','northeast');
        xlabel('k'); ylabel('MSE'); title('Matrix Denoising: MSE vs k');
        yline(mean((clean(:)-noisy(:)).^2),'k--','Noisy','LineWidth',1.5);
        plot(bestPCA_k, bestPCA_mse, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
        plot(bestSVD_k, bestSVD_mse, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;

        bestPCA = round(bestPCA,4);
        bestSVD = round(bestSVD,4);
        noisyMat = round(noisyMat,4);
        origMat = round(origMat,4);

    else
        k = userk;
        if k<1||k>kmax, error('k out of range'); end
        [U,S,V] = svd(noisy, 'econ');
        reconSVD = U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
        m1 = mean(noisy,1);
        cent = noisy - m1;
        covM = cov(cent);
        [E,ev] = eig(covM);
        [~,idx]=sort(diag(ev),'descend');
        E=E(:,idx);
        Ek = E(:,1:min(k,size(E,2)));
        proj = cent*Ek;
        reconPCA = proj*Ek'+m1;

        bestPCA = round(reconPCA,4);
        bestSVD = round(reconSVD,4);
        noisyMat = round(noisy,4);
        origMat = round(clean,4);

        fprintf('k=%d\n',k);
        fprintf('Noisy: MSE=%.4g\n',mean((clean(:)-noisy(:)).^2));
        fprintf(' PCA : MSE=%.4g\n',mean((clean(:)-reconPCA(:)).^2));
        fprintf(' SVD : MSE=%.4g\n',mean((clean(:)-reconSVD(:)).^2));
    end
end

function runDenoiseCurves(noisy, clean, kmax, isRGB, userk, dtype)
    % This is the original image denoising function from your prior code.
    if nargin<4, isRGB=false; end
    if nargin<5, userk=[]; end
    if nargin<6, dtype='image'; end

    if ndims(noisy)==3, C=size(noisy,3); else, C=1; end
    refMSE = mseMetric(clean,noisy);

    if strcmp(dtype,'image')
        refPSNR = psnrMetric(clean,noisy);
        refSSIM = ssimMetric(clean,noisy,isRGB);
    end

    if isempty(userk)
        ks = 1:kmax;
        msePCA = zeros(size(ks));
        mseSVD = zeros(size(ks));
        tPCA = zeros(size(ks));
        tSVD = zeros(size(ks));
        if strcmp(dtype,'image')
            psnrPCA = zeros(size(ks));
            psnrSVD = zeros(size(ks));
            ssimPCA = zeros(size(ks));
            ssimSVD = zeros(size(ks));
        end

        % SVD decomp once per channel
        Uc=cell(1,C); Sc=cell(1,C); Vc=cell(1,C);
        for c=1:C
            if C==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,c));
            end
            [U,S,V] = mySVD(chan);
            Uc{c}=U; Sc{c}=S; Vc{c}=V;
        end
        % PCA decomp once per channel
        Mc=cell(1,C); Ec=cell(1,C);
        for c=1:C
            if C==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,c));
            end
            m1 = mean(chan,1);
            cent = chan-m1;
            covM = cov(cent);
            [E,ev] = eig(covM);
            [~,idx]=sort(diag(ev),'descend');
            E=E(:,idx);
            Mc{c}=m1; Ec{c}=E;
        end

        for i=1:length(ks)
            k=ks(i);

            % SVD
            tic;
            if C==1
                recon = Uc{1}(:,1:k)*Sc{1}(1:k,1:k)*Vc{1}(:,1:k)';
                recon = min(max(recon,0),255);
                svdRecon = uint8(recon);
            else
                svdRecon = zeros(size(noisy),'double');
                for c=1:C
                    recon = Uc{c}(:,1:k)*Sc{c}(1:k,1:k)*Vc{c}(:,1:k)';
                    recon = min(max(recon,0),255);
                    svdRecon(:,:,c) = recon;
                end
                svdRecon=uint8(svdRecon);
            end
            tSVD(i) = toc;

            % PCA
            tic;
            if C==1
                Ek = Ec{1}(:,1:min(k,size(Ec{1},2)));
                proj = (double(noisy)-Mc{1})*Ek;
                recon = proj*Ek' + Mc{1};
                recon = min(max(recon,0),255);
                pcaRecon = uint8(recon);
            else
                pcaRecon = zeros(size(noisy),'double');
                for c=1:C
                    Ek = Ec{c}(:,1:min(k,size(Ec{c},2)));
                    proj = (double(noisy(:,:,c))-Mc{c})*Ek;
                    recon = proj*Ek' + Mc{c};
                    recon = min(max(recon,0),255);
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

        % Optimum k (first k beating noisy reference)
        optimumPCA_k = find(msePCA < refMSE, 1, 'first');
        optimumSVD_k = find(mseSVD < refMSE, 1, 'first');
        if strcmp(dtype,'image')
            optimumPCA_k_ssim = find(ssimPCA > refSSIM, 1, 'first');
            optimumSVD_k_ssim = find(ssimSVD > refSSIM, 1, 'first');
        end

        % Best k (max SSIM)
        if strcmp(dtype,'image')
            [~,bestPCA_k]=max(ssimPCA);
            [~,bestSVD_k]=max(ssimSVD);
        end

        % ---- All metrics in one 2x2 figure ----
        figure('Name','All metrics vs k','Units','normalized','Position',[.1 .2 .7 .6]);
        subplot(2,2,1);
        plot(ks,msePCA,'b-',ks,mseSVD,'r-','LineWidth',2); hold on
        yline(refMSE,'k--','LineWidth',1.5);
        if strcmp(dtype,'image')
            plot(ks(bestPCA_k),msePCA(bestPCA_k),'bo','MarkerSize',10,'LineWidth',2);
            plot(ks(bestSVD_k),mseSVD(bestSVD_k),'ro','MarkerSize',10,'LineWidth',2);
            if ~isempty(optimumPCA_k), plot(ks(optimumPCA_k), msePCA(optimumPCA_k), 'bd', 'MarkerSize',10, 'LineWidth',2); end
            if ~isempty(optimumSVD_k), plot(ks(optimumSVD_k), mseSVD(optimumSVD_k), 'rd', 'MarkerSize',10, 'LineWidth',2); end
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
            plot(ks(bestPCA_k),psnrPCA(bestPCA_k),'bo','MarkerSize',10,'LineWidth',2);
            plot(ks(bestSVD_k),psnrSVD(bestSVD_k),'ro','MarkerSize',10,'LineWidth',2);
            legend('PCA','SVD','Noisy','PCA best','SVD best','Location','southeast');
            xlabel('k'); ylabel('PSNR (dB)'); title('PSNR vs k'); grid on; hold off

            subplot(2,2,4);
            plot(ks,ssimPCA,'b-',ks,ssimSVD,'r-','LineWidth',2); hold on
            yline(refSSIM,'k--','LineWidth',1.5);
            plot(ks(bestPCA_k),ssimPCA(bestPCA_k),'bo','MarkerSize',10,'LineWidth',2);
            plot(ks(bestSVD_k),ssimSVD(bestSVD_k),'ro','MarkerSize',10,'LineWidth',2);
            if ~isempty(optimumPCA_k_ssim), plot(ks(optimumPCA_k_ssim), ssimPCA(optimumPCA_k_ssim), 'bd', 'MarkerSize',10, 'LineWidth',2); end
            if ~isempty(optimumSVD_k_ssim), plot(ks(optimumSVD_k_ssim), ssimSVD(optimumSVD_k_ssim), 'rd', 'MarkerSize',10, 'LineWidth',2); end
            legend('PCA','SVD','Noisy','PCA best','SVD best','PCA opt','SVD opt','Location','southeast');
            xlabel('k'); ylabel('SSIM'); title('SSIM vs k'); grid on; hold off
        else
            subplot(2,2,3); axis off; text(0.1,0.5,'PSNR not for text mode','FontSize',14);
            subplot(2,2,4); axis off; text(0.1,0.5,'SSIM not for text mode','FontSize',14);
        end

        % ---- Show best and optimum reconstructions ----
        if strcmp(dtype,'image')
            reconPCA_best = getRecon(Uc,Sc,Vc,Mc,Ec,noisy,C,bestPCA_k,'pca');
            reconSVD_best = getRecon(Uc,Sc,Vc,Mc,Ec,noisy,C,bestSVD_k,'svd');
            reconPCA_opt = [];
            reconSVD_opt = [];
            if exist('optimumPCA_k_ssim','var') && ~isempty(optimumPCA_k_ssim)
                reconPCA_opt = getRecon(Uc,Sc,Vc,Mc,Ec,noisy,C,optimumPCA_k_ssim,'pca');
            end
            if exist('optimumSVD_k_ssim','var') && ~isempty(optimumSVD_k_ssim)
                reconSVD_opt = getRecon(Uc,Sc,Vc,Mc,Ec,noisy,C,optimumSVD_k_ssim,'svd');
            end
            figure('Name','Reconstructed Images','Units','normalized','Position',[.15 .3 .7 .3]);
            if ndims(clean)==2
                subplot(1,4,1); imagesc(clean); axis image off; title('Original'); colormap gray;
                subplot(1,4,2); imagesc(noisy); axis image off; title('Noisy'); colormap gray;
                subplot(1,4,3); imagesc(reconPCA_best); axis image off; title(sprintf('PCA best (k=%d)',bestPCA_k)); colormap gray;
                subplot(1,4,4); imagesc(reconSVD_best); axis image off; title(sprintf('SVD best (k=%d)',bestSVD_k)); colormap gray;
            else
                subplot(1,4,1); imshow(clean); title('Original');
                subplot(1,4,2); imshow(noisy); title('Noisy');
                subplot(1,4,3); imshow(reconPCA_best); title(sprintf('PCA best (k=%d)',bestPCA_k));
                subplot(1,4,4); imshow(reconSVD_best); title(sprintf('SVD best (k=%d)',bestSVD_k));
            end
            % Print both best and optimum
            if exist('ssimPCA','var')
                fprintf('Best SSIM for PCA:    k=%d, SSIM=%.4f\n',bestPCA_k,ssimPCA(bestPCA_k));
                fprintf('Best SSIM for SVD:    k=%d, SSIM=%.4f\n',bestSVD_k,ssimSVD(bestSVD_k));
                if exist('optimumPCA_k_ssim','var') && ~isempty(optimumPCA_k_ssim)
                    fprintf('Optimum SSIM for PCA: k=%d, SSIM=%.4f\n',optimumPCA_k_ssim,ssimPCA(optimumPCA_k_ssim));
                end
                if exist('optimumSVD_k_ssim','var') && ~isempty(optimumSVD_k_ssim)
                    fprintf('Optimum SSIM for SVD: k=%d, SSIM=%.4f\n',optimumSVD_k_ssim,ssimSVD(optimumSVD_k_ssim));
                end
            end
        end
    else
        k = userk;
        if k<1||k>kmax, error('k out of range'); end
        svdRecon = zeros(size(noisy));
        for c=1:C
            if C==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,c));
            end
            [U,S,V]=mySVD(chan);
            r = U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
            r = min(max(r,0),255);
            if C==1, svdRecon=r; else, svdRecon(:,:,c)=r; end
        end
        svdRecon=uint8(svdRecon);
        pcaRecon = zeros(size(noisy));
        for c=1:C
            if C==1 && ndims(noisy)==2
                chan = double(noisy);
            else
                chan = double(noisy(:,:,c));
            end
            m1=mean(chan,1);
            cent=chan-m1;
            covM = cov(cent);
            [E,ev]=eig(covM);
            [~,idx]=sort(diag(ev),'descend');
            E=E(:,idx);
            Ek = E(:,1:min(k,size(E,2)));
            proj = cent*Ek;
            r = proj*Ek'+m1;
            r = min(max(r,0),255);
            if C==1, pcaRecon=r; else, pcaRecon(:,:,c)=r; end
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

function recon = getRecon(Uc,Sc,Vc,Mc,Ec,noisy,C,k,mode)
    if strcmp(mode,'svd')
        if C==1
            recon = Uc{1}(:,1:k)*Sc{1}(1:k,1:k)*Vc{1}(:,1:k)';
            recon = min(max(recon,0),255);
            recon = uint8(recon);
        else
            recon = zeros(size(noisy),'double');
            for c=1:C
                r = Uc{c}(:,1:k)*Sc{c}(1:k,1:k)*Vc{c}(:,1:k)';
                r = min(max(r,0),255);
                recon(:,:,c)=r;
            end
            recon=uint8(recon);
        end
    else
        if C==1
            Ek = Ec{1}(:,1:min(k,size(Ec{1},2)));
            proj = (double(noisy)-Mc{1})*Ek;
            recon = proj*Ek'+Mc{1};
            recon = min(max(recon,0),255);
            recon = uint8(recon);
        else
            recon = zeros(size(noisy),'double');
            for c=1:C
                Ek = Ec{c}(:,1:min(k,size(Ec{c},2)));
                proj = (double(noisy(:,:,c))-Mc{c})*Ek;
                r = proj*Ek'+Mc{c};
                r = min(max(r,0),255);
                recon(:,:,c)=r;
            end
            recon=uint8(recon);
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
