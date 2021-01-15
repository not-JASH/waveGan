function [dly,st] = Generator(dlx,params,st)
    persistent d u
    if isempty(d)
        d = cell(7,1);u=d;
    end

    dly = dlconv(dlx,params.CNw{1},params.CNb{1},...
        "Stride",2,"padding",'same');
    d{1} = tanh(dly);
%     d{1} = dly;
    for i = 2:7
        convBlock;
    end

    u{1} = dltranspconv(d{7},params.TCw{1},params.TCb{1},...
        "Stride",2,"cropping",'same');
    [u{1},st] = batchnormwrap(u{1},params,st,7);
    u{1} = cat(3,u{1},d{6});
%     fprintf("here are the skips\n") 
    for i = 2:6
        tConvBlock;
    end
    u{7} = dltranspconv(u{6},params.TCw{7},params.TCb{7},...
        "Stride",2,"cropping",'same');
    dly = tanh(u{7});
    
%     dly = u{7};
%     for i = 1:14
%         dly(:,:,i) = dly(:,:,i) - min(dly(:,:,i),[],'all');
%         dly(:,:,i) = dly(:,:,i)/max(dly(:,:,i),[],'all');
%     end

    function convBlock
        d{i} = dlconv(d{i-1},params.CNw{i},params.CNb{i},...
            "Stride",2,"padding",'same');
        d{i} = tanh(d{i});
        [d{i},st] = batchnormwrap(d{i},params,st,i-1);
    end
        
        
    function tConvBlock 
        u{i} = dltranspconv(u{i-1},params.TCw{i},params.TCb{i},...
            "Stride",2,"cropping",'same');
        u{i} = tanh(u{i});
        [u{i},st] = batchnormwrap(u{i},params,st,i+6);
        u{i} = cat(3,u{i},d{7-i});
    end
end