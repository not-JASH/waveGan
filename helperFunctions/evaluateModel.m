function [meanRMSE,RMSE] = evaluateModel(params,st,inputData,outputData)
    RMSE = cell(size(inputData,4),1);
    for i = 1:size(inputData,4)
    %     progressbar(i/size(inputData,4));
        sampleIn = gpudl(inputData(:,:,:,i),'SSCB');
        sampleOut = Generator(sampleIn,params,st);
        RMSE{i} = sqrt(sum(power(outputData(:,:,:,i)-gatext(sampleOut),2),'all')/numel(sampleOut));
    end
    meanRMSE = mean(cell2mat(RMSE));
end
