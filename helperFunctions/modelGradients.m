function [GradGen,GradDis,stGen,stDis,g_loss,d_loss,rmse] = modelGradients(x,y,paramsGen,paramsDis,stGen,stDis)
%     persistent pairYes fakeA pairNo d_loss g_loss
    pairYes = Discriminator(cat(4,y,x),paramsDis,stDis);
    [fakeA,stGen] = Generator(x,paramsGen,stGen);
    
    rmse = sum(power(gatext(y-fakeA),2),'all')/numel(fakeA);
    
    [pairNo,stDis] = Discriminator(cat(4,fakeA,x),paramsDis,stDis);
    
    d_loss = -.5*(mean(log(pairYes+eps) + log(1-pairNo+eps),'all'));
    g_loss = -.5*mean(log(pairNo+eps),'all')+100*mean(abs(y-fakeA),'all');
    
    GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
    GradDis = dlgradient(d_loss,paramsDis);
end