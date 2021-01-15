if ~exist('inputData','var')||~exist('outputData','var')
    load('C:\projectData\surfGan\trainingData\256_30_inputOutput.mat')
    load('C:\projectData\surfGan\trainedNets\trained_net_3_30.mat')
end

if ~exist('inputData_sc','var')||~exist('outputData_sc','var')
    inputData_sc = inputData;
    outputData_sc = outputData;

    for i = 1:size(inputData,4)
        for j = 1:size(inputData,3)
            inputData_sc(:,:,j,i) = scaleData(inputData(:,:,j,i));
            outputData_sc(:,:,j,i) = scaleData(outputData(:,:,j,i));
        end
    end
end

nSamples = 50;
sampleIndex = randi(size(inputData,4),[nSamples 1]);
sampleInputs = inputData_sc(:,:,:,sampleIndex);
referenceOutputs = outputData_sc(:,:,:,sampleIndex);
sampleOutputs = zeros(size(sampleInputs));

[meanRMSE,rmse] = evaluateModel(paramsGen,stGen,inputData_sc,outputData_sc);
fprintf("mean rmse : %f\n",meanRMSE)

for i = 1:nSamples
    sample = gpudl(sampleInputs(:,:,:,i),'SSCB');
    sampleOutputs(:,:,:,i) = gatext(Generator(sample,paramsGen,stGen));
end

function normData = scaleData(Data)
    normData = Data - min(Data,[],'all');
    normData = normData/max(normData,[],'all');
    normData = 2*(normData-0.5);
end
    