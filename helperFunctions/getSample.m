function [X,Y,xSignal,ySignal] = getSample(data,ws,predictionLength)
% WindowSize [ws]
    startpoint = randi(round(size(data,1)-2.1*ws-predictionLength),1);
    [scale,wave] = dualtree(zeros(ws,1));
    noLvl = size(wave,1);
    if rem(ws,2) == 0
        noLvl = noLvl-1;
        [scale,wave] = dualtree(zeros(ws,1),'level',noLvl);
    end
    xTemp = cell(noLvl+1,1);
    xTemp{1} = zeros(ws,size(scale,1));
    for i = 2:noLvl+1
        xTemp{i} = zeros(ws,size(wave{i-1},1));
    end
    yTemp = xTemp;

    k = startpoint;xSample=zeros(ws,1);ySample=zeros(ws,1);
    while true 
        %look ma im stuck in a loop!
        xSample(1) = []; xSample = [xSample;data(k)];
        ySample(1) = []; ySample = [ySample;data(k+predictionLength)];
        k = k+1;
        %plot(xSample);f = getframe;
        if xSample(1)==0;continue;end
        yTemp = waveDec(yTemp,ySample);
        xTemp = waveDec(xTemp,xSample);

        if ~all(xTemp{end}(1,:)==0)
            break
        end
    end
    
    X = zeros(ws,ws,noLvl,2);Y = zeros(ws,ws,noLvl,2);
    xTemp=regularize(xTemp);yTemp=regularize(yTemp);
    xTemp=cell2nDarray(xTemp);yTemp=cell2nDarray(yTemp);
    X(:,:,:,1) = real(xTemp);X(:,:,:,2) = imag(xTemp);
    Y(:,:,:,1) = real(yTemp);Y(:,:,:,2) = imag(yTemp);
    xSignal=xSample;ySignal=ySample;
    
    function array = cell2nDarray(temp)
        array = zeros(ws,ws,noLvl);
        %discarding scaling coefficients
        for i = 1:noLvl
            array(:,:,i) = temp{i+1};
        end
    end

    function temp = waveDec(temp,sample)
        [scale,wave] = dualtree(sample,'level',noLvl);
        temp{1}(1,:) = [];temp{1} = [temp{1};scale'];
        for i = 2:noLvl+1
            temp{i}(1,:) = [];
            temp{i} = [temp{i};wave{i-1}'];
        end
    end

    function temp = regularize(temp)
        y12 = [1:ws];
        for i = 1:noLvl+1
            display(size(temp{i}))
            x1 = [1:size(temp{i},2)]';
            x2 = linspace(1,size(temp{i},2),ws)';
            temp{i} = interp2(x1,y12,temp{i},x2,y12,'cubic');
        end
    end
end
