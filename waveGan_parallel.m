delete(gcp('nocreate'));clear;clc
addpath('./helperFunctions');
addpath('./trainingData');
getWorkers(2);

%% Settings

maxepochs = 200;
batchSize = 16;
genFilt = 32;
discFilt = 64;

%% Initialize Variables 
[gen,genStats] = initGen(genFilt);
[dis,disStats] = initDis(discFilt);
spmd
    settings = standardSettings(batchSize,maxepochs);
   
    paramsGen = gen; stGen = genStats;
    paramsDis = dis; stDis = disStats;
    
    avgG.Dis = []; avgGS.Dis = [];
    avgG.Gen = []; avgGS.Gen = [];
    
    %[{generator},{discriminator}]
    batchLoss = cell(batchSize,2);
    batchGrad = cell(batchSize,2);
    batchSt   = cell(batchSize,2);
   
    input = []; output = []; 
    
    l = 1;
end
clear gen genStats dis disStats
modelFcn = {[]};
out = false; epoch = 0; globalIter = 0;
otherlab = [2 1];
% validationRMSE = [];

%% Load Data and transfer to workers
testData = 'trainingData\256_30_inputOutput.mat';
if ~exist('XData','var')||~exist('YData','var')
    if ~exist('inputData','var')||~exist('outputData','var')
        load(testData)
        for i = 1:size(inputData,4)
            for j = 1:size(inputData,3)
                inputData(:,:,j,i) = scaleData(inputData(:,:,j,i));
                outputData(:,:,j,i) = scaleData(outputData(:,:,j,i));
            end
        end
    end
    
    spmd 
        XData = [];YData = [];
        gpuDevice(labindex);
    end
    
    validationSamples = randi(size(inputData,4),[maxepochs,1]);
    validationXData = inputData(:,:,:,validationSamples);
    validationYData = outputData(:,:,:,validationSamples);
    
    numIterations = size(inputData,4)/2;
    h1 = [1:numIterations];h2 = [numIterations+1:2*numIterations];
    
    XData{1} = inputData(:,:,:,h1);inputData = inputData(:,:,:,h1);
    XData{2} = inputData;clear inputData
    YData{1} = outputData(:,:,:,h1);outputData = outputData(:,:,:,h2);
    YData{2} = outputData;clear outputData
    clear h1 h2
    spmd
        %consider gpudl instead of gpuArray + dlarray(sample)
        gpuXData = gpuArray(XData);
        gpuYData = gpuArray(YData);
        
%         generatorLoss = cell(size(gpuXData,4),maxepochs);
%         discriminatorLoss = cell(size(gpuXData,4),maxepochs);
%         rmse = cell(size(gpuXData,4),maxepochs);
    end
end

%% Train
tic;
totalTime = toc;
while ~out
    epoch = epoch+1;
    fprintf("Epoch %d\n\n",epoch)
    i = 1;
    
    spmd
        shuffleID = randperm(numIterations);
    end
    
    while i <= numIterations
        globalIter = globalIter+1;
        j = 1;
        for i = i + [0:batchSize-1]
            if i > numIterations; break;end
            spmd 
                % Process Batch
                input = dlarray(gpuXData(:,:,:,shuffleID(i)),'SSCB');
                output = dlarray(gpuYData(:,:,:,shuffleID(i)),'SSCB');
                [batchGrad{j,1},batchGrad{j,2},batchSt{j,1},batchSt{j,2},...
                    batchLoss{j,1},batchLoss{j,2},~] = ...
                    dlfeval(@modelGradients,input,output,paramsGen,paramsDis,stGen,stDis);
%                 generatorLoss{i,epoch} = batchLoss{j,1};
%                 discriminatorLoss{i,epoch} = batchLoss{j,2};
            end
            j = j+1;
        end
        
        %Combine Batch Stats
        for j = 1:2
            if j == 1
                modelFcn = [{@addGradGen},{@min}];
            else
                modelFcn = [{@addGradDis},{@min}];
            end
            
            spmd
                loss = modelFcn{2}(obj2mat(batchLoss(j,1)));
                if batchSize > 1
                    for l = 2:batchSize
                        batchGrad{1,j} = modelFcn{1}(batchGrad{1,j},batchGrad{l,j});
                        batchSt{1,j} = addSt(batchSt{1,j},batchSt{l,j});
                    end
                end
                batchGrad{1,j} = modelFcn{1}(batchGrad{1,j},...
                    labSendReceive(otherlab(labindex),otherlab(labindex),batchGrad{1,j}));
                batchSt{1,j} = addSt(batchSt{1,j},...
                    labSendReceive(otherlab(labindex),otherlab(labindex),batchSt{1,j}));
            end
        end
        
        %Compute new parameters
        spmd
            % Update Generator Network Parameters
            [paramsGen,avgG.Gen,avgGS.Gen] = ...
                adamupdate(paramsGen,batchGrad{1,1},avgG.Gen,avgGS.Gen,globalIter,...
                settings.lrG,settings.beta1,settings.beta2);
            
                    
            % Update Discriminator Network Parameters
            [paramsDis,avgG.Dis,avgGS.Dis] = ...
                adamupdate(paramsDis,batchGrad{1,2},avgG.Dis, avgGS.Dis, globalIter, ...
                settings.lrD, settings.beta1, settings.beta2);
            
        end
    end
        
    elapsedTime = toc-totalTime;
    totalTime = toc;
    disp("Epoch"+epoch+". Time taken for epoch = "+elapsedTime + "s")
    if epoch == maxepochs
        out = true;
    end    
end
totalTime = toc;
fprintf("total time : %.2f\n",totalTime);

%% Helper Functions
function matrix = obj2mat(object)
    %assumes one dimensional cell array
    matrix = zeros(size(object));
    for i = 1:length(matrix)
        matrix(i) = object{i};
    end
end

function settings = standardSettings(batchsize,maxepochs)
    settings.disc_patch = [16 16 1];
    settings.batch_size = batchsize;
    settings.image_size = [256 256 7];%<-------[important]
    settings.lrD = 0.5e-3;
    settings.lrG = 0.5e-3;
    settings.beta1 = 0.5;
    settings.beta2 = 0.999;
    settings.maxepochs = maxepochs;
end

function normData = scaleData(Data)
    normData = Data - min(Data,[],'all');
    normData = normData/max(normData,[],'all');
    normData = 2*(normData-0.5);
end

function params = addGradGen(params,params2)
    for i = 1:12
        params.BNo{i} = params.BNo{i}+params2.BNo{i};
        params.BNs{i} = params.BNs{i}+params2.BNs{i};
        if i < 8
            params.CNw{i} = params.CNw{i}+params2.CNw{i};
            params.CNb{i} = params.CNb{i}+params2.CNb{i};
            params.TCw{i} = params.TCw{i}+params2.TCw{i};
            params.TCb{i} = params.TCb{i}+params2.TCb{i};
        end
    end
end

function params = addGradDis(params,params2)
    for i = 1:5
        params.CNw{i} = params.CNw{i}+params2.CNw{i};
        params.CNb{i} = params.CNb{i}+params2.CNb{i};
        if i < 4
            params.BNo{i} = params.BNo{i} + params2.BNo{i};
            params.BNs{i} = params.BNs{i} + params2.BNs{i};
        end
    end
end

function params = addSt(params,params2)
    for i = 1:size(params.BN)
        params.BN{i}.mu = params.BN{i}.mu + params2.BN{i}.mu;
        params.BN{i}.sig = params.BN{i}.sig + params2.BN{i}.sig;
    end
end