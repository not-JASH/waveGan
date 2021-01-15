addpath('./helperFunctions');
getWorkers(8);
windowSize = 240;
numLvls = 7;    

testData = '256_30_inputOutput.mat';
if ~exist('inputData','var')||~exist('outputData','var')
    load(testData)
end

%% hotfix 
% coeffs = zeros(size(inputData,3),2,size(inputData,4));
%[level, input-output, sample]

for j = 1:size(inputData,4)
%     progressbar(j/size(inputData,4))
    for i = 1:size(inputData,3)
        inputData(:,:,i,j) = scaleData(inputData(:,:,i,j));
        outputData(:,:,i,j) = scaleData(outputData(:,:,i,j));
    end
end

%% settings 
gf = 32; df = 64; 
settings.disc_patch = [16 16 1];
settings.batch_size = 1; %% increase this number till 100% gpu utilization
settings.image_size = [windowSize windowSize numLvls];
settings.lrD = 2e-4; settings.lrG = 2e-4; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 200;

%% Generator (conv->deconv + skips)
[paramsGen,stGen] = initGen(gf);

%% Discriminator (conv)
[paramsDis,stDis] = initDis(df);

%% Train

avgG.Dis = []; avgGS.Dis = []; 
avgG.Gen = []; avgGS.Gen = [];
numIterations = size(inputData,4);
out = false; epoch = 0; global_iter = 0;

generatorLoss = cell(numIterations,settings.maxepochs);
discriminatorLoss = cell(numIterations,settings.maxepochs);
RMSE = zeros(numIterations,settings.maxepochs);
tic
totalTime = toc;
while ~out 
    shuffleID = randperm(size(inputData,4));
    
    fprintf("Epoch %d\n",epoch)
    for i = 1:settings.batch_size:numIterations 
        global_iter = global_iter + 1;
        
        XBatch = gpudl(inputData(:,:,:,shuffleID(i)),'SSCB');
        YBatch = gpudl(outputData(:,:,:,shuffleID(i)),'SSCB');
        
        % Evaluate Model Gradients
        [GradGen,GradDis,stGen,stDis,...
            generatorLoss{i,epoch+1},discriminatorLoss{i,epoch+1},...
            RMSE(i,epoch+1)] = ...
            dlfeval(@modelGradients,XBatch,YBatch,...
            paramsGen,paramsDis,stGen,stDis);
        
        % Update Discriminator Network Parameters
        [paramsDis,avgG.Dis,avgGS.Dis] = ...
            adamupdate(paramsDis,GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            settings.lrD, settings.beta1, settings.beta2);
        
        % Update Generator Network Parameters
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen,GradGen,...
            avgG.Gen, avgGS.Gen, global_iter,...
            settings.lrG, settings.beta1, settings.beta2);
        
        if i == 1 || rem(i,10) == 0
            idXplot = randi(size(inputData,4));
            xPlot = gpudl(inputData(:,:,:,idXplot),'SSCB');
            yPlot = gpudl(outputData(:,:,:,idXplot),'SSCB');
          
%             RMSE(global_iter) = progressplot(xPlot,gatext(yPlot),paramsGen,stGen,1);
        end
    end
    
    elapsedTime = toc-totalTime;
    totalTime = toc;
    disp("Epoch"+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end
end
totalTime = toc;
fprintf("Total Time : %.2f\n",totalTime);
save('tempStorage.mat','paramsGen','stGen','generatorLoss','discriminatorLoss','RMSE');

            

function normData = scaleData(Data)
    normData = Data - min(Data,[],'all');
    normData = normData/max(normData,[],'all');
    normData = 2*(normData-0.5);
end

























            