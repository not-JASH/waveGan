clc;clear;
windowSize = 256;
predictionLength = 60;
numSamples = 1000;
files = dir('binanceData');
files(1:2) = [];
numFiles = size(files,1)-1;

data = cell(numFiles,1);
for i = 1:numFiles
    data{i} = binance_textLoad(fullfile('binanceData',files(i).name));
    data{i} = data{i}(:,5)-data{i}(:,2);
end
i = 1;
while i <= size(data,1)
    if isempty(data{i})
        data(i) = [];
    else
        i = i+1;
    end
end
numFiles = size(data,1);
sampleFiles = randi(numFiles,[numSamples 1]);

jobs = cell(numSamples,1);
getWorkers(10);

for i = 1:numSamples
    jobs{i} = parfeval(@getSample,4,data{sampleFiles(i)},...
        windowSize,predictionLength);
end

sample = fetchOutputs(jobs{1});
sampleSize = size(sample);
clear sample

inputTemp = zeros([sampleSize,numSamples]);
outputTemp = inputTemp;
signals = cell(numSamples,2);

for i = 1:numSamples
    progressbar(i/numSamples);
    [inputTemp(:,:,:,:,i),outputTemp(:,:,:,:,i),signals{i,1},signals{i,2}] = fetchOutputs(jobs{i});
end

inputData = zeros([[1 1 2].*sampleSize(1:3),numSamples]);
inputData = reshapeData(inputData,inputTemp);
clear inputTemp
outputData = zeros([[1 1 2].*sampleSize(1:3),numSamples]);
outputData = reshapeData(outputData,outputTemp);
clear outputTemp

    
filename = append(num2str(windowSize),'_',num2str(predictionLength),'_inputOutput');
save(fullfile('trainingData',filename),'inputData','outputData','signals','-v7.3');
clearvars -except inputData


function data = reshapeData(data,data1)
    for i = 1:size(data,4)
        for j = 1:size(data,3)
            if rem(j,2)~=0
                data(:,:,j,i) = data1(:,:,ceil(j/2),1,i);
            else
                data(:,:,j,i) = data1(:,:,ceil(j/2),2,i);
            end
        end
    end
end
            