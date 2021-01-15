function rmse = progressplot(xData,yData,paramsGen,stGen,i)
    persistent x y genImg p f
    if isempty(p)
        p = panel;
        p.pack(3,14);
        p.de.margin = 0.1;
    end
    x = gpudl(xData(:,:,:,i),'SSCB');
    y = yData(:,:,:,i);
    genImg = Generator(x,paramsGen,stGen);
    rmse = sqrt(sum(power(y-genImg,2),'all'))/numel(y);
%     [rmse,~] = evaluateModel(paramsGen,stGen,xData,yData);
    fprintf("RMSE : %.8f\n",rmse);
   
    for i = 1:3
        for j = 1:14
            p(i,j).select()
            if i == 1
                surf(gatext(x(:,:,j)),'linestyle','none')
            elseif i == 2
                
                surf(gatext(genImg(:,:,j)),'linestyle','none')
            else 
                surf(y(:,:,j),'linestyle','none')
            end
            view(2)
            xlim([0 256]) %<------[256]!!
            ylim([0 256])
            set(gca,'xticklabel',[],'yticklabel',[]);
        end
    end
    f = getframe;
end