function [dly,st] = Discriminator(dlx,params,st)
    dly = dlconv(dlx,params.CNw{1},params.CNb{1},...
        'Stride',2,'padding','same');
%     dly = leakyrelu(dly,.2);
    dly = tanh(dly);
    for i = 2:4
        dly = dlconv(dly,params.CNw{i},params.CNb{i},...
            'Stride',2,'padding','same');
%         dly = leakyrelu(dly,.2);
        dly = tanh(dly);
        [dly,st] = batchnormwrap(dly,params,st,i-1);
    end
    dly = dlconv(dly,params.CNw{5},params.CNb{5},...
        'Stride',1,'Padding','same');
    dly = sigmoid(dly);
end