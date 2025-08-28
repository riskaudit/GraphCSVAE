%% Initialize
clear all
close all
clc

%%
[X,R] = readgeoraster("data/SHOCK/20170528 Bangladesh Cyclone Mora/Google25D/ZE/2020.tif");

%%
fp_dir = "data/SHOCK/20170528 Bangladesh Cyclone Mora/Google25D/ZV/";
listing  = struct2table(dir(fp_dir));
fp_label = string(listing.name(listing.isdir==0));
fp_label = erase(fp_label, ".tif");
fp_label(fp_label==".DS_Store") = [];

fp_data = zeros([size(X)';length(fp_label)]');
for i = 1:length(fp_label)
    fp_data(:,:,i) = readgeoraster(fp_dir+fp_label(i)+".tif");
end

%% Create Tiles

createTrainTiles = 0;
if createTrainTiles == 1
    mask = (X>0) & (sum(fp_data>0,3)>0);
    H = X.*mask;
    labels = fp_data.*mask;
    
    tileSize = 450;
    stride = 0; 
    connectDist = 1.5;
    
    if stride == 0
        stride = tileSize;
    end
    
    numRows = floor((size(H,1) - tileSize) / stride) + 1;
    numCols = floor((size(H,2) - tileSize) / stride) + 1;
    
    labelsCells = {};
    nodesGlobalCells = {};
    featuresCells = {};
    adjacencyCells = {};
    
    for i = 1:numRows
        for j = 1:numCols
            disp([i,numRows,j,numCols])
            rowStart = (i-1)*stride + 1;
            rowEnd   = rowStart + tileSize - 1;
            colStart = (j-1)*stride + 1;
            colEnd   = colStart + tileSize - 1;
    
            tile = H(rowStart:rowEnd, colStart:colEnd);
    
            [r, c, v] = find(tile);
            if numel(v) < 2
                continue;
            end
    
            globalRows = r + (rowStart - 1);
            globalCols = c + (colStart - 1);
            nodes_global = [globalRows, globalCols];

            n = size(tile,1) * size(tile,2);
            idx = find(tile);
            bw = sparse(idx, idx, 1, n, n);
            B = sparse_dilate_ultra(bw);
            C = B - bw;      
            [ii, jj, ~] = find(C);
            Bresult = sparse(ii, jj, 1, n, n);
            adjacency = sparse(logical(Bresult(idx, idx)));

            if nnz(adjacency) == 0
                continue; 
            end
    
            labelPatch = labels(rowStart:rowEnd, colStart:colEnd, :);
    
            numNodes = numel(r);
            labelVectors = zeros(numNodes, size(labels, 3));
            for idx = 1:numNodes
                labelVectors(idx, :) = squeeze(labelPatch(r(idx), c(idx), :))';
            end
    
            labelsCells{end+1} = labelVectors;
            nodesGlobalCells{end+1} = nodes_global;
            featuresCells{end+1} = v;
            adjacencyCells{end+1} = adjacency;
        end
    end
    save('data/SHOCK/20170528 Bangladesh Cyclone Mora/createTiles.mat', ...
         "labelsCells","nodesGlobalCells","featuresCells","adjacencyCells")
else
    load('data/SHOCK/20170528 Bangladesh Cyclone Mora/createTiles.mat', ...
         "labelsCells","nodesGlobalCells","featuresCells","adjacencyCells")
end

%% DataSplit
datasplit = 0;
fn = (1:size(labelsCells,2))';
if datasplit == 1
    diversity = zeros(length(fn),length(fp_label));
    for ifn = 1:size(labelsCells,2)
        diversity(ifn,:) = sum(labelsCells{1,ifn});
    end
    
    TVT = zeros(length(fn),3);
    [~,p] = sort(sum(diversity>0),'ascend');
    r = 1:length(sum(diversity>0));
    r(p) = r;
    for iR = 1:length(r)
        if (iR == 1)
            tmp = diversity(:,(r==iR));
            iGreaterThanZeroT = find(tmp>0);
            [trainInd,valInd,testInd] = dividerand(length(iGreaterThanZeroT),0.7,0.15,0.15);
            TVT(iGreaterThanZeroT(trainInd),:) = repmat([1 0 0],length(trainInd),1);
            TVT(iGreaterThanZeroT(valInd),:)   = repmat([0 1 0],length(valInd),1);
            TVT(iGreaterThanZeroT(testInd),:)  = repmat([0 0 1],length(testInd),1);
        else % now considers running placement of 1s
            tmp = diversity(:,(r==iR));
            iGreaterThanZeroT = find(tmp>0 & sum(TVT,2)~=1);
            if ~isempty(iGreaterThanZeroT)
                [trainInd,valInd,testInd] = dividerand(length(iGreaterThanZeroT),0.7,0.15,0.15);
                TVT(iGreaterThanZeroT(trainInd),:) = repmat([1 0 0],length(trainInd),1);
                TVT(iGreaterThanZeroT(valInd),:)   = repmat([0 1 0],length(valInd),1);
                TVT(iGreaterThanZeroT(testInd),:)  = repmat([0 0 1],length(testInd),1);
            end
        end
    end
    iNotYet = find(sum(TVT,2)~=1);
    [trainInd,valInd,testInd] = dividerand(length(iNotYet),0.7,0.15,0.15);
    TVT(iNotYet(trainInd),:) = repmat([1 0 0],length(trainInd),1);
    TVT(iNotYet(valInd),:)   = repmat([0 1 0],length(valInd),1);
    TVT(iNotYet(testInd),:)  = repmat([0 0 1],length(testInd),1);
    T = table((1:size(labelsCells,2))', TVT(:,1), TVT(:,2), TVT(:,3));
    T.Properties.VariableNames = {'fn', 'train', 'valid', 'test'};
    filename = 'data/SHOCK/20170528 Bangladesh Cyclone Mora/datasplit.csv';
    writetable(T, filename);
else
    T = readtable('data/SHOCK/20170528 Bangladesh Cyclone Mora/datasplit.csv');
end
fnTrain = fn(T.train==1);
fnValid = fn(T.valid==1);
fnTest  = fn(T.test==1);

%% PrepareData and Normalize X
XTrain = vertcat(featuresCells{1,fnTrain});
muX = mean(log(XTrain));
sigsqX = var(log(XTrain),1);
XTrainNorm = (log(XTrain) - muX)./sqrt(sigsqX);
minXNorm = min(XTrainNorm);
maxXNorm = max(XTrainNorm);
XTrainNormMinMax = (XTrainNorm-minXNorm)./(maxXNorm-minXNorm);

for iFN = 1:length(fnTrain)
    TE.OV.XTMM.Train{iFN,1} = ((log(featuresCells{1,fnTrain(iFN)}) - muX)./sqrt(sigsqX)-minXNorm)./(maxXNorm-minXNorm);
end
for iFN = 1:length(fnValid)
    TE.OV.XTMM.Valid{iFN,1} = ((log(featuresCells{1,fnValid(iFN)}) - muX)./sqrt(sigsqX)-minXNorm)./(maxXNorm-minXNorm);
end
for iFN = 1:length(fnTest)
    TE.OV.XTMM.Test{iFN,1} = ((log(featuresCells{1,fnTest(iFN)}) - muX)./sqrt(sigsqX)-minXNorm)./(maxXNorm-minXNorm);
end
    
labelsCellsNorm = cell(size(labelsCells,2),1);
for iFN = 1:size(labelsCells,2)
    tmp = labelsCells{:,iFN};
    normtmp = tmp./sum(tmp,2);
    normtmp(isnan(normtmp)) = 0;
    labelsCellsNorm{iFN,:} = normtmp;
end
for iFN = 1:size(adjacencyCells,2)
    iFN
    tmp = adjacencyCells{:,iFN};
    adjacencyCells{:,iFN} = sparse(logical(tmp));
end

%% subGraphs
subGraphs = adjacencyCells(1,fnTrain);
n = numel(subGraphs);
totalSize = 0;            
rowIndices = [];          
colIndices = [];          
values = [];              
offset = 0;
for i = 1:n
    A = subGraphs{i};
    [r, c, v] = find(A);  
    r = r + offset;
    c = c + offset;

    rowIndices = [rowIndices; r];
    colIndices = [colIndices; c];
    values     = [values; v];

    offset = offset + size(A, 1);
end
totalSize = offset;
finalMatrix = sparse(rowIndices, colIndices, values, totalSize, totalSize);
aXTrain = logical(finalMatrix); 

subGraphs = adjacencyCells(1,fnValid);
n = numel(subGraphs);
totalSize = 0;            
rowIndices = [];          
colIndices = [];         
values = [];             
offset = 0;
for i = 1:n
    A = subGraphs{i};
    [r, c, v] = find(A);  
    r = r + offset;
    c = c + offset;

    rowIndices = [rowIndices; r];
    colIndices = [colIndices; c];
    values     = [values; v];

    offset = offset + size(A, 1);
end
totalSize = offset;
finalMatrix = sparse(rowIndices, colIndices, values, totalSize, totalSize);
aXValid = logical(finalMatrix);  

subGraphs = adjacencyCells(1,fnTest);
n = numel(subGraphs);
totalSize = 0;            
rowIndices = [];          
colIndices = [];          
values = [];              
offset = 0;
for i = 1:n
    A = subGraphs{i};
    [r, c, v] = find(A);  
    r = r + offset;
    c = c + offset;

    rowIndices = [rowIndices; r];
    colIndices = [colIndices; c];
    values     = [values; v];

    offset = offset + size(A, 1);
end
totalSize = offset;
finalMatrix = sparse(rowIndices, colIndices, values, totalSize, totalSize);
aXTest = logical(finalMatrix); 

%% Training
doOVTraining = 1;
[fig31,fig32,lineReconLoss,lineKL] = initializeOVfig();
epoch = 0; iteration = 0;
trailingAvgSqE = []; trailingAvgE = [];
trailingAvgD = []; trailingAvgSqD = [];
[pOV_encoder,pOV_decoder] = initializePOV(length(fp_label),1);
numEpochs = 50; numBatches = 10; 
numNodes = length(double(vertcat(TE.OV.XTMM.Train{:})));
while mod(numNodes, numBatches) ~= 0
    numNodes = numNodes - 1;
end
numSubGraph = numNodes./numBatches;
p_dropout = 0.2;
learnRate = 1e-3;
start = tic;

%%
inXTrain = double(vertcat(TE.OV.XTMM.Train{:}));
inLTrain = vertcat(labelsCellsNorm{fnTrain(:),1});

inXValid = double(vertcat(TE.OV.XTMM.Valid{:}));
inLValid = vertcat(labelsCellsNorm{fnValid(:),1});

inXTest = double(vertcat(TE.OV.XTMM.Test{:}));
inLTest = vertcat(labelsCellsNorm{fnTest(:),1});


%%
while epoch < numEpochs 
    epoch = epoch + 1; ibatch = 0;
    bG = reshape(randperm(numNodes), ...
                 numSubGraph, numBatches);
    learnRate = ((epoch<numEpochs./2)*1e-2)+((epoch>=numEpochs./2)*1e-3);
    while ibatch < numBatches

        startIT = tic;

        iteration = iteration + 1; ibatch = ibatch + 1; disp([epoch, ibatch, iteration])
        xfn = bG(:,ibatch); 

        upper_triangle = triu(aXTrain(xfn,xfn));
        
        [i_edges0, j_edges0, v_edges0] = find(upper_triangle);
        num_edges = length(v_edges0);
        
        num_to_drop = ceil(num_edges * p_dropout);
        iDrop = randperm(num_edges, num_to_drop);
        
        i_keep = i_edges0;
        j_keep = j_edges0;
        v_keep = v_edges0;
        i_keep(iDrop) = [];
        j_keep(iDrop) = [];
        v_keep(iDrop) = [];
        
        new_upper_triangle = sparse(i_keep, j_keep, v_keep, size(aXTrain(xfn,xfn),1), size(aXTrain(xfn,xfn),2));
        
        aX = new_upper_triangle + triu(new_upper_triangle, 1)';

        % Train: Evaluate the model loss and gradients. 
        [grad_pOV_encoder, grad_pOV_decoder, ...
         reconLossTrain(iteration), KLTrain(iteration), ~, ~] = ...
        dlfeval(@modelOVlossSHOCK, pOV_encoder, pOV_decoder, inXTrain(xfn,:), ...
                aX, inLTrain(xfn,:));

        D = duration(0,0,toc(start),Format="hh:mm:ss");
        figure(fig31); title("Recon, Epoch: " + epoch + ", Iteration: " + iteration + ", Elapsed: " + string(D))
        addpoints(lineReconLoss.Train,iteration,double(reconLossTrain(iteration))); drawnow
        figure(fig32); title("KL, Epoch: " + epoch + ", Iteration: " + iteration + ", Elapsed: " + string(D))
        addpoints(lineKL.Train,iteration,double(KLTrain(iteration))); drawnow 

        % Valid and Test: Evaluate the model loss.
        if (iteration==1) || (mod(iteration, 10) == 0)

            itmp = randperm(length(inXValid),ceil(length(inXValid)./100));
            [~, ~, reconLossValid(iteration), KLValid(iteration), ~, ~] = ...
            dlfeval(@modelOVlossSHOCK, pOV_encoder, pOV_decoder, inXValid(itmp,:), ...
                aXValid(itmp,itmp), inLValid(itmp,:) );

            itmp = randperm(length(inXTest),ceil(length(inXTest)./100));
            [~, ~, reconLossTest(iteration), KLTest(iteration), ~, ~] = ...
            dlfeval(@modelOVlossSHOCK, pOV_encoder, pOV_decoder, inXTest(itmp,:), ...
                aXTest(itmp,itmp), inLTest(itmp,:) );

            figure(fig31); title("Recon, Epoch: " + epoch + ", Iteration: " + iteration + ", Elapsed: " + string(D))
            addpoints(lineReconLoss.Valid,iteration,double(reconLossValid(iteration))); drawnow
            addpoints(lineReconLoss.Test,iteration,double(reconLossTest(iteration))); drawnow
            figure(fig32); title("KL, Epoch: " + epoch + ", Iteration: " + iteration + ", Elapsed: " + string(D))
            addpoints(lineKL.Valid,iteration,double(KLValid(iteration))); drawnow 
            addpoints(lineKL.Test,iteration,double(KLTest(iteration))); drawnow 

        end

        % Update learnable parameters.  
        [pOV_encoder,trailingAvgE,trailingAvgSqE] = adamupdate(pOV_encoder, ...
            grad_pOV_encoder,trailingAvgE,trailingAvgSqE,iteration,learnRate);
        [pOV_decoder,trailingAvgD,trailingAvgSqD] = adamupdate(pOV_decoder, ...
            grad_pOV_decoder,trailingAvgD,trailingAvgSqD,iteration,learnRate);
        disp('---------------------------------------')

    end
    save('data/SHOCK/20170528 Bangladesh Cyclone Mora/GraphVSSM_OVparameters.mat', "pOV_decoder","pOV_encoder")
    savefig(fig31,'data/SHOCK/20170528 Bangladesh Cyclone Mora/OV_ReconLoss.fig'); 
    savefig(fig32,'data/SHOCK/20170528 Bangladesh Cyclone Mora/OV_KL.fig'); 
end

%% Inference

fp_data_norm = fp_data./sum(fp_data,3);
fp_data_norm(isnan(fp_data_norm)) = 0;

for yr = 2016:2023
    yr
    tmp = readgeoraster("data/SHOCK/20170528 Bangladesh Cyclone Mora/Google25D/ZE/"+string(yr)+".tif");
    tmp = tmp(r1:r2, c1:c2);
    tmp(tmp<=0)                       = 0;
    TE.OV.Index.Infer{1,(yr-2015)}    = find( (tmp>0) );
    TE.OV.X.Infer{1,(yr-2015)}        = tmp(TE.OV.Index.Infer{1,(yr-2015)});

    TE.OV.Adj.Infer{iFN,(yr-2015)} = sparse([]); 
    n = size(tmp,1) * size(tmp,2);
    idx = TE.OV.Index.Infer{1,(yr-2015)};
    bw = sparse(idx, idx, 1, n, n);
    B = sparse_dilate_ultra(bw);
    C = B - bw;
    [i, j, ~] = find(C);
    Bresult = sparse(i, j, 1, n, n);
    TE.OV.Adj.Infer{1,(yr-2015)} = Bresult(idx, idx);

    TE.OV.prior.Infer{1,(yr-2015)} = []; 
    tmp3 = [];
    for iP = 1:length(fp_label)
        tmp31 = fp_data_norm(:,:,iP);
        tmp3 = [tmp3; tmp31(TE.OV.Index.Infer{1,(yr-2015)})'];
    end
    TE.OV.prior.Infer{1,(yr-2015)} = tmp3';

end

%%
for yr = 2016:2023
    yr
    TE.OV.XTMM.Infer{1,(yr-2015)} = ...
        ((log(double(TE.OV.X.Infer{1,(yr-2015)})) - muX)./sqrt(sigsqX)-minXNorm)./(maxXNorm-minXNorm);
end

%%
[~,~] = mkdir("data/SHOCK/20170528 Bangladesh Cyclone Mora/Google25D/FINAL");
fnInfer = 1;
for yr = 2016:2023
    yr
    OV.p.Infer = zeros([size(tmp,1), size(tmp,2), length(fp_label)]);

    p = ones(size(vertcat(TE.OV.prior.Infer{:,(yr-2015)}))).*1./length(fp_label);

    [theta, sampled_theta] = encoderOV(pOV_encoder, ...
                              double(vertcat(TE.OV.XTMM.Infer{:,(yr-2015)})), ...
                              blkdiag(TE.OV.Adj.Infer{:,(yr-2015)}), ...
                              p);
    [row,col]   = ind2sub([size(tmp,1), size(tmp,2)],vertcat(TE.OV.Index.Infer{:,(yr-2015)}));
    idx4Da = sub2ind(size(OV.p.Infer), ...
                     repmat(row,[length(fp_label),1]), ...
                     repmat(col,[length(fp_label),1]), ...
                     repelem(1:length(fp_label),size(theta,1))' ...
                     );
    OV.p.Infer(idx4Da) = reshape(extractdata(sampled_theta),[],1);
    for i = 1:length(fp_label)
        disp([yr,fp_label(i)])
        geotiffwrite("data/SHOCK/20170528 Bangladesh Cyclone Mora/Google25D/FINAL"+'/'+string(yr)+"_"+string(fp_label(i))+'.tif', ...
            OV.p.Infer(:,:,i), R, 'CoordRefSysCode', 32646);
    end
end